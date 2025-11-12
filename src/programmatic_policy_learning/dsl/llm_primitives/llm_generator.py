"""This module defines utilities for transforming the structured output of a
LLM into a formal Grammar object.

It processes nonterminals, terminals, and production rules to
dynamically construct grammars for DSLs.
"""

import importlib.util
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, MutableMapping, Union, cast

import black
import numpy as np
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.dsl.generators.grammar_based_generator import Grammar
from programmatic_policy_learning.dsl.llm_primitives.utils import (
    JSONStructureRepromptCheck,
    SemanticJSONVerifierReprompt,
    SemanticsPyStubRepromptCheck,
)
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    GridInput,
    _eval,
    at_action_cell,
    at_cell_with_value,
    cell_is_value,
    get_dsl_functions_dict,
    scanning,
    shifted,
)

Cell = tuple[int, int] | None
LocalProgram = Callable[[Cell, np.ndarray], Any]  # LocalProgram - (cell, obs) -> *
Sym = Union[str, int]  # terminal = str, nonterminal = int


class LLMPrimitivesGenerator:
    """A class to interact with an LLM, process its response, and generate a
    Grammar object."""

    def __init__(
        self,
        llm_client: PretrainedLargeModel | None,
        removed_primitive: str | None,
        output_dir: str = "outputs/",
    ) -> None:
        """Initialize the generator with an LLM client.

        Args:
            llm_client (Any): An object or function to interact
            with the LLM (e.g., OpenAI API client).
        """
        self.llm_client = llm_client
        base_dir = Path(__file__).parent
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = base_dir / output_dir / self.run_id
        if llm_client is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)
        self.grammar: Grammar[str, int, int] | None = None
        self.removed_primitive = removed_primitive

    def query_llm(self, prompt: str) -> dict:
        """Send a query to the LLM and return the parsed JSON response.

        Args:
            prompt (str): The query or prompt to send to the LLM.

        Returns:
            dict: The parsed JSON response from the LLM.
        """
        if self.llm_client is None:
            raise ValueError("LLM client is not initialized.")

        query = Query(prompt)
        reprompt_checks = [
            JSONStructureRepromptCheck(),
            SemanticJSONVerifierReprompt(),
            SemanticsPyStubRepromptCheck(),
        ]

        response = query_with_reprompts(
            self.llm_client,
            query,
            reprompt_checks,  # type: ignore[arg-type]
            max_attempts=5,
        )
        logging.debug("Response from LLM:")
        logging.debug(response)
        return json.loads(response.text)

    def create_grammar_from_response(
        self,
        llm_output: dict[str, Any],
        object_types: list[Any],
    ) -> Grammar[str, int, int]:
        """Convert a JSON-like grammar spec (with strings for productions) into
        a Grammar where nonterminals are numbered 0..N-1 and RHS alternatives
        are lists of symbols,

        with terminals as strings and nonterminals as ints. Probabilities are uniform.
        - If productions['VALUE'] contains '{OBJECT_TYPES}',
        it's expanded using object_types.
        """
        rules = llm_output["updated_grammar"]
        nonterminals: list[str] = rules["nonterminals"]
        productions: dict[str, str] = rules["productions"]

        # Map nonterminal name -> index
        nt_to_id: dict[str, int] = {nt: i for i, nt in enumerate(nonterminals)}

        # Find any NT name in a string. Match longest first to avoid partial overlaps.
        # Literal match so NTs inside text are also caught.
        sorted_nts = sorted(nonterminals, key=len, reverse=True)
        nt_pattern = re.compile("|".join(re.escape(nt) for nt in sorted_nts))

        def split_replace_nts(s: str) -> list[Sym]:
            """Split s around any NT occurrences, returning
            [str/int/str/...]."""
            out: list[Sym] = []
            pos = 0
            for m in nt_pattern.finditer(s):
                if m.start() > pos:
                    out.append(s[pos : m.start()])
                out.append(nt_to_id[m.group(0)])
                pos = m.end()
            if pos < len(s):
                out.append(s[pos:])
            # If no NTs found, keep the whole string as a single terminal
            if not out:
                out.append(s)
            # Drop empty-string terminals (purely cosmetic)
            out = [tok for tok in out if not (isinstance(tok, str) and tok == "")]
            return out

        def parse_alternatives(prod_str: str, nt_name: str) -> list[list[Sym]]:
            """Split a production 'a | b | c' into RHS lists with NT
            replacement."""
            prod_str = prod_str.strip()
            # Special-case VALUE expansion
            if nt_name == "VALUE" and "{OBJECT_TYPES}" in prod_str:
                alts = [str(x) for x in object_types]
                return [[alt] for alt in alts]  # each is a single terminal
            # Normal split on '|'
            raw_alts = [alt.strip() for alt in prod_str.split("|")]
            return [split_replace_nts(alt) for alt in raw_alts]

        # Build the rules: { nt_id: (alternatives, probabilities) }
        grammar_rules: dict[int, tuple[list[list[Sym]], list[float]]] = {}
        for nt_name, nt_id in nt_to_id.items():
            if nt_name not in productions:
                raise KeyError(f"Missing production for nonterminal '{nt_name}'")
            alts_tokens = parse_alternatives(productions[nt_name], nt_name)
            if not alts_tokens:
                raise ValueError(f"No alternatives for nonterminal '{nt_name}'")
            p = 1.0 / len(alts_tokens)
            probs = [p] * len(alts_tokens)
            grammar_rules[nt_id] = (alts_tokens, probs)

        for nt_id, (alts, _) in grammar_rules.items():
            if not alts:
                raise ValueError(f"No RHS for nonterminal {nt_id}")
            for alt in alts:
                for sym in alt:
                    if not isinstance(sym, (int, str)):
                        raise TypeError(f"Bad symbol {sym!r} in {nt_id}")

        self.write_json("grammar.json", grammar_rules)
        return Grammar(rules=grammar_rules)

    def generate_and_process_grammar(
        self, prompt_text: str, object_types: list[Any]
    ) -> tuple[
        Grammar[str, int, int],
        dict[str, Any],
        DSL[
            Callable[[tuple[int, int] | None, np.ndarray[Any, Any]], Any],
            GridInput,
            Any,
        ],
    ]:
        """Generate a Grammar object by querying the LLM and processing its
        response."""
        llm_response = self.query_llm(prompt_text)
        self.write_json("metadata.json", llm_response)

        new_primitive_name = llm_response["proposal"]["name"]
        python_str = llm_response["proposal"]["semantics_py_stub"]
        # python_file = create_function_from_stub(python_str, new_primitive_name)
        self.write_python_file(new_primitive_name, python_str)
        implementation = self.load_function_from_file(
            str(self.output_path / f"{new_primitive_name}.py"), new_primitive_name
        )
        new_dsl_object = self.make_dsl(new_primitive_name, implementation)
        new_get_dsl_functions_fn = self.add_primitive_to_dsl(
            new_primitive_name, implementation
        )
        self.grammar = self.create_grammar_from_response(llm_response, object_types)
        logging.info(self.grammar)
        logging.info(python_str)
        return self.grammar, new_get_dsl_functions_fn(), new_dsl_object

    def create_grammar(
        self, env_spec: dict[str, Any] | None
    ) -> Grammar[
        Callable[[tuple[int, int] | None, np.ndarray[Any, Any]], Any], GridInput, Any
    ]:
        """Replace the object types of grammar with the given ones."""
        if env_spec is None:
            raise ValueError("env_spec cannot be None")
        if self.grammar is None:
            raise ValueError("Grammar is not initialized")
        object_types = env_spec["object_types"]
        self.grammar.rules[4] = (
            [[str(v)] for v in object_types],
            [1.0 / len(object_types) for _ in object_types],
        )
        return cast(
            Grammar[
                Callable[[tuple[int, int] | None, np.ndarray[Any, Any]], Any],
                GridInput,
                Any,
            ],
            self.grammar,
        )

    def add_primitive_to_dsl(
        self, name: str, implementation: Callable[..., Any]
    ) -> Callable[[], dict[str, Any]]:
        """Add a new primitive to the DSL functions dictionary.

        Args:
            name: The name of the new primitive.
            implementation: The Python implementation of the primitive.

        Returns:
            A modified `get_dsl_functions_dict` function.
        """
        # Get the base DSL functions
        base_dsl_functions = get_dsl_functions_dict()

        if name in base_dsl_functions:
            raise ValueError(f"Primitive '{name}' already exists in the DSL.")

        # Add the new primitive
        base_dsl_functions[name] = implementation

        # Return a new function that includes the updated DSL
        def updated_get_dsl_functions_dict() -> dict[str, Any]:
            return base_dsl_functions

        return updated_get_dsl_functions_dict

    def write_json(self, filename: str, data: dict) -> None:
        """Write JSON data to a file in the output directory."""
        json_path = self.output_path / filename
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)

    def write_python_file(self, primitive_name: str, code: str) -> None:
        """Write Python code (str) to a file in the output directory."""
        code = code.replace("\\n", "\n")
        python_file_path = self.output_path / f"{primitive_name}.py"
        try:
            # Format the code using black
            formatted_code = black.format_str(code, mode=black.FileMode())
        except black.InvalidInput:
            # If the code is invalid, write it as-is
            formatted_code = code

        with open(python_file_path, "w", encoding="utf-8") as python_file:
            python_file.write(formatted_code)

    def load_function_from_file(
        self, file_path: str, function_name: str
    ) -> Callable[..., Any]:
        """Dynamically load a function from a Python file."""
        spec = importlib.util.spec_from_file_location("module_name", file_path)
        if spec is None:
            raise ValueError(
                "Failed to create a module spec. Ensure the file path is correct."
            )
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ValueError("Module spec loader is None. Ensure the file is valid.")
        spec.loader.exec_module(module)
        return getattr(module, function_name)

    def make_dsl(
        self, added_name: str, added_python_fn: Callable[..., Any]
    ) -> DSL[LocalProgram, GridInput, Any]:
        """Construct the grid DSL object with the added primitive."""
        prims: MutableMapping[str, Callable[..., Any]] = {
            "cell_is_value": cell_is_value,
            "shifted": shifted,
            "at_cell_with_value": at_cell_with_value,
            "at_action_cell": at_action_cell,
            "scanning": scanning,
            added_name: added_python_fn,
        }
        if self.removed_primitive is not None:
            del prims[self.removed_primitive]
        return DSL(id="grid_v1_augmented", primitives=prims, evaluate_fn=_eval)

    def offline_loader(self, run_id: str) -> tuple[
        Grammar[str, int, int],
        dict[str, Any],
        DSL[LocalProgram, GridInput, Any],
    ]:
        """Load the grammar, DSL functions, and DSL object from a previous run.

        Args:
            run_id (str): The run ID of the previous generation.

        Returns:
            tuple: A tuple containing the Grammar, updated DSL functions, and DSL object.
        """
        base_dir = Path(__file__).parent
        output_path = base_dir / "outputs" / run_id

        # Load the grammar
        grammar_path = output_path / "grammar.json"
        with open(grammar_path, "r", encoding="utf-8") as grammar_file:
            grammar_data = json.load(
                grammar_file, object_hook=lambda d: {int(k): v for k, v in d.items()}
            )
        self.grammar = Grammar(rules=grammar_data)

        # Load the metadata
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        # Extract the new primitive details
        new_primitive_name = metadata["proposal"]["name"]

        # Dynamically construct the file path using run_id
        output_dir = Path(__file__).parent / "outputs" / run_id
        python_files = list(output_dir.glob("*.py"))
        if not python_files:
            raise FileNotFoundError(f"No Python files found in {output_dir}")
        file_path = python_files[0]  # Assuming there's only one Python file

        implementation = self.load_function_from_file(
            str(file_path), new_primitive_name
        )
        updated_dsl_fn = self.add_primitive_to_dsl(new_primitive_name, implementation)

        # Create the DSL object
        new_dsl_object = self.make_dsl(new_primitive_name, implementation)

        return self.grammar, updated_dsl_fn(), new_dsl_object
