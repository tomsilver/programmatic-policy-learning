"""Util for LLM-based Primitives Generation."""

import ast
import builtins
import json
import logging
import re
from typing import Any, Callable

from prpl_llm_utils.reprompting import RepromptCheck, create_reprompt_from_error_message
from prpl_llm_utils.structs import Query, Response

from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    get_dsl_functions_dict,
)


class JSONStructureRepromptCheck(RepromptCheck):
    """Check whether the LLM's response contains valid JSON with required
    fields."""

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        try:
            llm_output = json.loads(response.text)
        except json.JSONDecodeError as e:
            error_msg = f"The response is not valid JSON: {str(e)}"
            return create_reprompt_from_error_message(query, response, error_msg)

        # Check for required fields
        required_fields = ["proposal", "updated_grammar"]
        missing_fields = [field for field in required_fields if field not in llm_output]
        if missing_fields:
            error_msg = f"The response JSON is missing required fields:\
                {', '.join(missing_fields)}"
            return create_reprompt_from_error_message(query, response, error_msg)

        return None


class SemanticJSONVerifierReprompt(RepromptCheck):
    """Checks semantic and structural validity of an LLM-generated grammar
    JSON."""

    VALID_NONTERMINALS = {"START", "LOCAL_PROGRAM", "CONDITION", "DIRECTION", "VALUE"}

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        # I assume it's been checked before to be a valid json file
        llm_output = json.loads(response.text)

        proposal = llm_output.get("proposal", {})
        grammar = llm_output.get("updated_grammar", {})

        # 1. Nonterminal validity
        nonterm = proposal.get("pcfg_insertion", {}).get("nonterminal")
        if not isinstance(nonterm, str) or "|" in nonterm:
            return create_reprompt_from_error_message(
                query,
                response,
                "The pcfg_insertion->'nonterminal'\
                must be a single valid name (no '|').",
            )
        if nonterm not in self.VALID_NONTERMINALS:
            return create_reprompt_from_error_message(
                query,
                response,
                f"Invalid nonterminal '{nonterm}'.\
                    Must be one of {sorted(self.VALID_NONTERMINALS)}.",
            )

        # 2. Variable naming consistency
        stub = proposal.get("semantics_py_stub", "")
        if re.search(r"\blambda\s+cell\s*,\s*o\b", stub) or " o:" in stub:
            return create_reprompt_from_error_message(
                query,
                response,
                "Use 'cell, obs' (not 'o') in all lambdas and function definitions.",
            )

        # 3. Trailing commas or syntax typos
        prod = proposal.get("pcfg_insertion", {}).get("production", "")

        if re.search(r",\s*\|", prod) or prod.strip().endswith(","):
            return create_reprompt_from_error_message(
                query,
                response,
                "Production contains stray commas or misplaced '|'. Clean up syntax.",
            )

        # 4. Infinite recursion detection
        # e.g., CONDITION -> conditional_action(CONDITION, ...)
        prods = grammar.get("productions", {})
        for nt, rule in prods.items():
            if f"{nt}(" in rule:
                return create_reprompt_from_error_message(
                    query,
                    response,
                    f"Rule for '{nt}' calls itself directly;\
                        avoid recursive definitions.",
                )

        # 5. Missing terminals consistency
        terminals = grammar.get("terminals", [])
        if proposal.get("name") not in terminals:
            return create_reprompt_from_error_message(
                query,
                response,
                f"New terminal '{proposal.get('name')}' not added to 'terminals' list.",
            )

        # 6. Argument consistency across stub, args, and production ---
        func_name = proposal.get("name", "")

        # Extract argument list from stub
        func_match = re.search(rf"def\s+{re.escape(func_name)}\s*\(([^)]*)\):", stub)
        if func_match:
            func_args = [a.strip() for a in func_match.group(1).split(",") if a.strip()]
        else:
            func_args = []

        # Extract argument list from first call in production
        prod_match = re.search(rf"{re.escape(func_name)}\s*\(([^)]*)\)", prod)
        if prod_match:
            prod_args = [a.strip() for a in prod_match.group(1).split(",") if a.strip()]
        else:
            prod_args = []

        # Compare argument counts (ignore lambda params)
        if func_args and prod_args and len(func_args) != len(prod_args):
            return create_reprompt_from_error_message(
                query,
                response,
                f"The primitive '{func_name}' defines {len(func_args)}\
                arguments in its stub, but is called with {len(prod_args)}\
                    in production. Ensure the counts match.",
            )

        # 7. Ensure primitive is inserted only once across the grammar
        count = 0
        for nt, rhs in prods.items():
            # look for the primitive name followed by "("
            if re.search(rf"\b{re.escape(func_name)}\s*\(", rhs):
                count += 1

        if count > 1:
            return create_reprompt_from_error_message(
                query,
                response,
                f"The primitive '{func_name}' appears in {count}\
                different grammar rules. It must be inserted in\
                exactly one nonterminal, as specified in\
                    'pcfg_insertion.nonterminal'.",
            )

        # 8: Nonterminal + variable scope validation
        ALLOWED_RUNTIME_VARS = {"cell", "obs", "s", "a"}
        ALLOWED_KEYWORDS = {"lambda", "True", "False", "None"}
        declared_nts = set(grammar.get("nonterminals", []))
        declared_terms = terminals

        # Collect every identifier that looks like a Python name
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", prod)

        invalid = []
        for tok in tokens:
            if (
                tok not in ALLOWED_RUNTIME_VARS
                and tok not in ALLOWED_KEYWORDS
                and tok not in declared_nts
                and tok not in declared_terms
                and not tok.isupper()  # sometimes DSL constants like TPN.EMPTY
            ):
                invalid.append(tok)

        if invalid:
            return create_reprompt_from_error_message(
                query,
                response,
                f"Production contains undefined symbols: \
                    {', '.join(sorted(set(invalid)))}. "
                "Use only declared nonterminals, DSL terminals, \
                or bound variables (cell, obs, s, a).",
            )

        return None


class Scope:
    """Represents a variable scope in Python code, tracking defined, global,
    and nonlocal variables."""

    def __init__(self, parent: "Scope | None" = None) -> None:
        """Initialize a new Scope instance."""
        self.parent = parent
        self.defined: set[str] = set()
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()

    def is_defined(self, name: str) -> bool:
        """Check if a variable is defined in the current or parent scopes."""
        if name in self.defined:
            return True
        if name in self.globals:
            root = self
            while root.parent:
                root = root.parent
            return name in root.defined
        if name in self.nonlocals and self.parent:
            return self.parent.is_defined(name)
        return self.parent.is_defined(name) if self.parent else False


class UndefinedVisitor(ast.NodeVisitor):
    """AST visitor to identify undefined variable names in Python code."""

    DEFAULT_ALLOWED_IMPORTS = {
        "math",
        "statistics",
        "itertools",
        "functools",
        "collections",
        "re",
        "json",
        "typing",
        "dataclasses",
        "heapq",
        "bisect",
        "operator",
        "random",
    }

    def __init__(
        self,
        provided_globals: set[str] | None = None,
        allowed_imports: set[str] | None = None,
    ) -> None:
        """Initialize the UndefinedVisitor."""
        self.issues: set[str] = set()
        self.scope: Scope = Scope()
        self.provided: set[str] = set(provided_globals or [])
        self.allowed_imports: set[str] = allowed_imports or self.DEFAULT_ALLOWED_IMPORTS
        # builtins available by default
        self.scope.defined |= set(dir(builtins))
        # map alias -> top-level module (e.g, self.alias_to_mod = {"np": "numpy"})
        self.alias_to_mod: dict[str, str] = {}

    def push(self) -> None:
        """Push a new scope."""
        self.scope = Scope(self.scope)

    def pop(self) -> None:
        """Pop the current scope."""
        self.scope = self.scope.parent  # type: ignore

    # --- utilities ---
    def _define_target(self, target: ast.AST) -> None:
        """Define a variable or variables in the current scope."""
        # x = 1 -> defines "x"
        if isinstance(target, ast.Name):
            self.scope.defined.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._define_target(elt)

    def _load_name(self, name_node: ast.Name) -> None:
        """Check if a variable is defined; if not, add it to issues."""
        name = name_node.id
        if not (self.scope.is_defined(name) or name in self.provided):
            self.issues.add(name)

    # ---------- visitors ----------
    def visit_Name(self, node: ast.Name) -> None:
        """Visit a Name node."""
        # y + 1 -> visit_Name("y", Load) -> flagged if not defined
        if isinstance(node.ctx, ast.Load):
            self._load_name(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle assignment statements in the AST."""
        # x = 5 -> defines "x"
        self.visit(node.value)
        for t in node.targets:
            self._define_target(t)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignment statements in the AST."""
        # x: int = 10 -> defines "x"
        if node.value:
            self.visit(node.value)
        self._define_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle augmented assignment statements in the AST."""
        # x += 1 -> requires "x" defined first
        self.visit(node.target)
        self.visit(node.value)
        self._define_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Handle named expressions (walrus operator) in the AST."""
        # (x := 5) -> defines "x
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.scope.defined.add(node.target.id)

    def visit_For(self, node: ast.For) -> None:
        """Handle for-loops in the AST."""
        # for i in range(3): ... -> defines "i"
        self.visit(node.iter)
        self._define_target(node.target)
        self.push()
        for s in node.body:
            self.visit(s)
        self.pop()
        for s in node.orelse:
            self.visit(s)

    def visit_With(self, node: ast.With) -> None:
        """Handle with-statements in the AST."""
        # with open("x.txt") as f: ... -> defines "f"
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._define_target(item.optional_vars)
        self.push()
        for s in node.body:
            self.visit(s)
        self.pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Handle exception handlers in the AST."""
        # except Exception as e: ... -> defines "e"
        if node.name:
            # py>=3.11 can be ast.ExceptHandler.name as str or ast.Name older
            name = node.name if isinstance(node.name, str) else node.name.id
            self.scope.defined.add(name)
        self.push()
        for s in node.body:
            self.visit(s)
        self.pop()

    def visit_Match(self, node: ast.Match) -> None:
        """Handle match-statements (Python 3.10+) in the AST."""
        #   match x: ... case y: ... -> defines "y"
        self.visit(node.subject)
        for case in node.cases:
            self.push()
            self.visit(case.pattern)
            if case.guard:
                self.visit(case.guard)
            for s in case.body:
                self.visit(s)
            self.pop()

    def visit_comprehension(self, comp: ast.comprehension) -> None:
        """Handle comprehensions (e.g., list comprehensions) in the AST."""
        # [x for x in data if cond(x)] -> defines "x" in inner scope
        self.visit(comp.iter)
        self._define_target(comp.target)
        for if_ in comp.ifs:
            self.visit(if_)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Handle list comprehensions in the AST."""
        # squares = [x*x for x in range(5)]
        self.push()
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.elt)
        self.pop()

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Handle dictionary comprehensions in the AST."""
        # {x: x**2 for x in range(3)}
        self.push()
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.key)
        self.visit(node.value)
        self.pop()

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Handle generator expressions in the AST."""
        # (x for x in range(3))
        self.push()
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.elt)
        self.pop()

    def visit_Global(self, node: ast.Global) -> None:
        """Handle global variable declarations in the AST."""
        # global x -> declares x as global
        self.scope.globals |= set(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Handle nonlocal variable declarations in the AST."""
        # nonlocal x -> refers to x in outer, non-global, scope
        self.scope.nonlocals |= set(node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle function definitions in the AST."""
        # def f(a): b = a -> defines "f" and "a", "b" inside inner scope

        self.scope.defined.add(node.name)  # function name binds in outer scope
        # new inner scope for params + body
        self.push()
        args = node.args
        for a in args.args + args.kwonlyargs:
            self.scope.defined.add(a.arg)
        if args.vararg:
            self.scope.defined.add(args.vararg.arg)
        if args.kwarg:
            self.scope.defined.add(args.kwarg.arg)
        for s in node.body:
            self.visit(s)
        self.pop()

    # to handle both normal and async versions with the same logic
    visit_AsyncFor: Callable[[ast.AsyncFor], None] = visit_For  # type: ignore
    visit_AsyncWith: Callable[[ast.AsyncWith], None] = visit_With  # type: ignore
    visit_SetComp: Callable[[ast.SetComp], None] = visit_ListComp  # type: ignore
    visit_AsyncFunctionDef: Callable[
        [ast.AsyncFunctionDef], None
    ] = visit_FunctionDef  # type: ignore

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Handle lambda expressions in the AST."""
        # lambda x: x + 1 -> defines "x"
        self.push()
        args = node.args
        for a in args.args + args.kwonlyargs:
            self.scope.defined.add(a.arg)
        if args.vararg:
            self.scope.defined.add(args.vararg.arg)
        if args.kwarg:
            self.scope.defined.add(args.kwarg.arg)
        self.visit(node.body)
        self.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definitions in the AST."""
        # class A: pass -> defines "A"
        # class name binds in enclosing scope
        self.scope.defined.add(node.name)
        self.push()
        for s in node.body:
            self.visit(s)
        self.pop()

    def visit_Import(self, node: ast.Import) -> None:
        """Handle import statements in the AST."""

        for alias in node.names:
            mod = alias.name.split(".")[0]
            asname = alias.asname or mod
            self.scope.defined.add(asname)
            self.alias_to_mod[asname] = mod

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle from-import statements in the AST."""
        for alias in node.names:
            asname = alias.asname or alias.name
            self.scope.defined.add(asname)


def find_undefined_names(
    source: str, *, provided_globals: set[str] | None = None
) -> set[str]:
    """Identify undefined variable names in the given Python source code."""
    tree = ast.parse(source)
    v = UndefinedVisitor(provided_globals=provided_globals or set())
    v.visit(tree)
    return v.issues  # set of strings


class SemanticsPyStubRepromptCheck(RepromptCheck):
    """Check whether the semantics_py_stub field contains valid, executable
    Python code that meets the required syntax and execution constraints."""

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        logging.info("Checking semantics_py_stub for validity...")

        # Assume JSON structure is already validated
        llm_output = json.loads(response.text)
        stub = llm_output["proposal"]["semantics_py_stub"]
        stub = stub.replace("\\n", "\n")
        logging.info(stub)
        try:
            # Ensure the stub is valid Python code /Syntax
            tree = ast.parse(stub)  # pylint: disable=unused-variable
        except SyntaxError as e:
            logging.info(f"Syntax error in semantics_py_stub: {e}")  # Log the error
            error_msg = (
                f"The semantics_py_stub contains invalid Python syntax: {str(e)}"
            )
            return create_reprompt_from_error_message(query, response, error_msg)

        # Check executability of the stub
        try:
            local_namespace: dict[str, Any] = {}
            # Execute in a restricted environment
            exec(stub, {}, local_namespace)  # pylint: disable=exec-used
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.info(f"Execution error in semantics_py_stub: {e}")  # Log the error
            error_msg = (
                f"The semantics_py_stub raised an error during execution: {str(e)}"
            )
            return create_reprompt_from_error_message(query, response, error_msg)

        # Add a set of hand-written function names
        hand_written_functions = list(get_dsl_functions_dict().keys())

        # Merge the hand-written functions with the provided globals
        undefined = find_undefined_names(
            stub, provided_globals=set(local_namespace) | set(hand_written_functions)
        )
        if undefined:
            error_msg = "The semantics_py_stub contains undefined names: " + ", ".join(
                sorted(undefined)
            )
            logging.info(error_msg)
            return create_reprompt_from_error_message(query, response, error_msg)
        logging.info("semantics_py_stub passed all checks.")
        return None


def create_function_from_stub(stub: str, function_name: str) -> Callable[..., Any]:
    """Convert a Python stub string into a callable function."""
    stub = stub.replace("\\n", "\n")
    stub = stub.replace("...", "pass")
    local_namespace: dict[str, Any] = {}
    exec(stub, {}, local_namespace)  # pylint: disable=exec-used
    return local_namespace[function_name]
