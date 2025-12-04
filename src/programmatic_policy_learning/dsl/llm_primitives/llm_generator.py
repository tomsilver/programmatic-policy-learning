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
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.dsl.generators.grammar_based_generator import Grammar
from programmatic_policy_learning.dsl.llm_primitives.dsl_evaluator import (
    evaluate_primitive,
)

# SemanticJSONVerifierReprompt,; SemanticsPyStubRepromptCheck,
# from programmatic_policy_learning.dsl.llm_primitives.utils import (
#     JSONStructureRepromptCheck,
# )
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    GridInput,
    _eval,
    at_action_cell,
    at_cell_with_value,
    cell_is_value,
    get_core_boolean_primitives,
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
        reprompt_checks: list[RepromptCheck] = [  # TODOO: Add for full version
            # JSONStructureRepromptCheck(),
            # SemanticJSONVerifierReprompt(),
            # SemanticsPyStubRepromptCheck(),
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
        object_types: tuple[Any],
    ) -> Grammar[str, int, int]:
        """Convert a JSON-like grammar spec (with strings for productions) into
        a Grammar where nonterminals are numbered 0..N-1 and RHS alternatives
        are lists of symbols,

        with terminals as strings and nonterminals as ints. Probabilities are uniform.
        - If productions['VALUE'] contains '{OBJECT_TYPES}',
        it's expanded using object_types.
        """

        rules = llm_output["updated_grammar"]
        logging.info(rules)
        terminals: list[str] = rules["terminals"]
        nonterminals: list[str] = rules["nonterminals"]
        productions: dict[str, str] = rules["productions"]

        missing_terms = [t for t in terminals if t not in str(productions)]
        if missing_terms:
            logging.warning(
                f"Some terminals not referenced in productions: {missing_terms}"
            )

        # Map nonterminal name -> index
        nt_to_id: dict[str, int] = {nt: i for i, nt in enumerate(nonterminals)}

        # Avoids if the name contains NTs by accident
        token_pattern = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

        def split_replace_nts(s: str) -> list[Sym]:
            """Tokenize s and replace tokens that are EXACTLY equal to a
            nonterminal name with its int id.

            Prevents accidental NT matches inside longer identifiers like:
            AT_CELL_WITH_VALUE  → stays whole and untouched.
            """
            out: list[Sym] = []
            for tok in token_pattern.findall(s):
                if tok in nt_to_id:  # match FULL TOKEN ONLY
                    out.append(nt_to_id[tok])
                else:
                    out.append(tok)
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
        self,
        prompt_text: str,
        object_types: tuple[Any],
        env_factory: Callable[[int], Any],  # type: ignore[arg-type]
        outer_feedback: str | None = None,
        mode: str = "full",
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
        response, supporting two modes:

        mode = "single"
            - Query for ONE primitive.
            - Apply evaluation+retry only if the DSL already has ≥1 primitive.
            - If DSL is empty, accept the first primitive without evaluation.

        mode = "full"
            - Query for a LIST of primitives.
            - Accept the first primitive (bootstrap).
            - Evaluate each subsequent primitive and retry on failure.
        """

        # ----------------------------------------------------------------------
        # 1. Build prompt (optional outer feedback from Timeout error)
        # ----------------------------------------------------------------------
        base_prompt = prompt_text
        if outer_feedback:
            prompt_text = prompt_text + f"\n\n{outer_feedback}\n"

        # ----------------------------------------------------------------------
        # 2. Query the LLM
        # ----------------------------------------------------------------------
        llm_response = self.query_llm(prompt_text)
        self.write_json("metadata.json", llm_response)
        # ----------------------------------------------------------------------
        # 3. Extract proposals uniformly
        # ----------------------------------------------------------------------
        raw_proposal = llm_response.get("proposal", None)

        if mode == "single":
            if not isinstance(raw_proposal, dict):
                raise ValueError("In single mode, expected 'proposal' to be a dict.")
            proposals = [raw_proposal]  # force list for unified processing

        elif mode == "full":
            if not isinstance(raw_proposal, list):
                raise ValueError("In full mode, expected 'proposal' to be a list.")
            proposals = raw_proposal

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ----------------------------------------------------------------------
        # 4. Prepare DSL extension step
        # ----------------------------------------------------------------------
        accepted_primitives: dict[str, Callable[..., Any]] = {}

        # Function for writing+loading the implementation
        def load_impl(proposal_dict: dict[str, Any]) -> Callable:
            name = proposal_dict["name"]
            py_stub = proposal_dict["semantics_py_stub"]
            self.write_python_file(name, py_stub)
            return self.load_function_from_file(
                str(self.output_path / f"{name}.py"), name
            )

        # Evaluation helper
        def evaluate_and_retry(
            proposal_dict: dict[str, Any],
            existing_prims: dict[str, Callable[..., Any]],
            object_types: tuple[Any],
            env_factory: Callable[[int], Any],
            llm_response: dict[str, Any],
        ) -> tuple[Callable, str, dict[str, Any], bool]:
            """Evaluate a proposed primitive and retry if it fails."""
            MAX_ATTEMPTS = 5
            feedback = None
            rep = llm_response
            for attempt in range(MAX_ATTEMPTS):
                impl = load_impl(proposal_dict)
                signature = tuple(
                    sorted((arg["name"], arg["type"]) for arg in proposal_dict["args"])
                )
                logging.info(existing_primitives)
                # input("EXISTING ONES")
                eval_result = evaluate_primitive(
                    impl,
                    existing_primitives=existing_prims,
                    object_types=object_types,
                    env_factory=env_factory,
                    proposal_signature=signature,
                    seed=1,
                    max_steps=20,
                    num_samples=200,
                    degeneracy_threshold=0.1,
                    equivalence_threshold=0.95,
                )

                if eval_result["keep"]:
                    # 4th one: returning the boolean that:
                    # if it has been replaced (return True)
                    return impl, proposal_dict["name"], rep, attempt != 0

                # primitive rejected → reprompt with feedback
                feedback = eval_result["reason"]
                logging.info(
                    f"Rejected primitive (attempt {attempt+1}/{MAX_ATTEMPTS}):\
                        {feedback}"
                )
                new_prompt = ""
                # Reconstruct prompt with reject reason
                if mode == "single":
                    new_prompt = (
                        base_prompt
                        + "\n\nThe previous primitive was rejected:\n"
                        + feedback
                        + "\n\nPlease propose a NEW primitive that avoids this issue."
                    )
                if mode == "full":

                    addition_prompt = (
                        f"\n\nThe previous primitive called {proposal_dict['name']}"
                        " you suggested was rejected because of the following reason"
                        f": {feedback}. This is the full dict information of that"
                        f"primitive: {proposal_dict}. Try to repair this primitive based"
                        "on the feedback that you got, or propose a new one that"
                        "contains the same arg types and pcfg insertion location, but"
                        "can have a different logic. Here are the existing primitives"
                        f"added to the language so far: {existing_prims}; avoid similar"
                        "names or functionalities.\n"
                        "- Propose one replacement primitive only. "
                        "Same signature: (args…). Only return: "
                        "{{'proposal':"
                        "[{{'name': ..., 'semantics_py_stub': ..., 'args': ...}}]}}.\n"
                        "- DO NOT regenerate the full DSL.\n"
                    )
                    new_prompt = base_prompt + addition_prompt
                    logging.info(addition_prompt)
                    # input("PROPOSING A NEW ONE BASED ON FEEDBACK")

                rep = self.query_llm(new_prompt)

                # can later enforce to not be a list
                proposal_dict = rep["proposal"][0]
                logging.info(proposal_dict)
                # input("THIS IS THE NEW ONE")

            # If every attempt failed:
            raise RuntimeError("Failed to generate a valid primitive after retries.")

        def patch_llm_response(
            llm_response: dict,
            old_name: str,
            new_proposal: dict,
        ) -> dict[str, Any]:
            """
            Mutates llm_response in-place:
            - Replaces primitive 'old_name' with 'new_name'
                                            everywhere in updated_grammar.
            - Updates the 'proposal' list entry (name, semantics, args).
            """

            new_name = new_proposal["name"]

            # ------------------------------------------------------------------
            # 1. Update the "proposal" list entry
            # ------------------------------------------------------------------
            for i, p in enumerate(llm_response.get("proposal", [])):
                if p.get("name") == old_name:
                    llm_response["proposal"][i]["name"] = new_name
                    llm_response["proposal"][i]["semantics_py_stub"] = new_proposal[
                        "semantics_py_stub"
                    ]
                    llm_response["proposal"][i]["args"] = new_proposal["args"]
                    # Keep pcfg_insertion the same (same NT + same types)
                    break

            # ------------------------------------------------------------------
            # 2. Update "terminals" list in updated_grammar
            # ------------------------------------------------------------------
            ug = llm_response["updated_grammar"]
            terminals = ug.get("terminals", [])
            ug["terminals"] = [new_name if t == old_name else t for t in terminals]

            # ------------------------------------------------------------------
            # 3. Replace occurrences in production strings
            # ------------------------------------------------------------------
            for nt, production in ug["productions"].items():
                if old_name in production:
                    ug["productions"][nt] = production.replace(old_name, new_name)

            return llm_response

        # ----------------------------------------------------------------------
        # 5. Process each proposed primitive
        # ----------------------------------------------------------------------
        count = 0
        added_responses: list[Any] = []
        for _, proposal_dict in enumerate(proposals):
            count += 1
            name = proposal_dict["name"]
            logging.info(f"ORIGINAL: {name}")
            # Determine DSL primitives present BEFORE adding this one
            if mode == "single":
                # Core + LLM-accepted primitives
                existing_primitives = {
                    **get_core_boolean_primitives(self.removed_primitive),
                }
            else:  # mode == "full"
                # Only primitives generated and accepted so far
                existing_primitives = {**accepted_primitives}

            is_bootstrap = len(existing_primitives) == 0

            # FIRST PRIMITIVE: accept immediately with no evaluation
            if is_bootstrap:
                logging.info(f"Bootstrap-accept primitive: {name}")
                impl = load_impl(proposal_dict)

            else:
                # SECOND+ PRIMITIVE: evaluate + retry if needed
                logging.info(f"Evaluating primitive: {name}")
                impl, name, new_llm_proposal, retried = evaluate_and_retry(
                    proposal_dict,
                    existing_primitives,
                    object_types,
                    env_factory,
                    llm_response,
                )

                if retried:
                    # call the function that reconstructs the grammar and metadata.
                    llm_response = patch_llm_response(
                        llm_response=llm_response,
                        old_name=proposal_dict["name"],
                        new_proposal=new_llm_proposal["proposal"][0],
                    )
                    logging.info(llm_response)
                    added_responses.append(llm_response)
                    # input("CHECKKKK @!!!!!!!")

            accepted_primitives[name] = impl

        # ----------------------------------------------------------------------
        # 6. Build DSL with all accepted primitives
        # ----------------------------------------------------------------------

        all_fns = {**accepted_primitives, **existing_primitives}
        logging.info(all_fns)
        new_dsl_object = DSL(
            id=f"grid_v1_{mode}_generated",
            primitives=all_fns,
            evaluate_fn=_eval,
        )
        logging.info(new_dsl_object)
        # input(all_fns)

        # Add to DSL set
        new_get_dsl_functions_fn = self.add_primitive_to_dsl(  # fix this
            list(accepted_primitives.keys()),
            list(accepted_primitives.values()),
        )
        self.write_json("new_metadata.json", llm_response)

        # ----------------------------------------------------------------------
        # 7. Write metadata + Build Grammar
        # ----------------------------------------------------------------------
        # self.write_json("metadata.json", llm_response)
        self.grammar = self.create_grammar_from_response(llm_response, object_types)

        # ----------------------------------------------------------------------
        # 8. Return the results
        # ----------------------------------------------------------------------
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
        self,
        names: list[str],
        implementations: list[Callable[..., Any]],
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

        for each_name, each_fn in zip(names, implementations):
            if each_name in base_dsl_functions:
                raise ValueError(f"Primitive '{each_name}' already exists in the DSL.")
            base_dsl_functions[each_name] = each_fn

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
        """Load the grammar, DSL functions, and DSL object from a previous
        run."""
        base_dir = Path(__file__).parent
        output_path = base_dir / "outputs" / run_id
        self.output_path = output_path
        self.run_id = run_id
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
        updated_dsl_fn = self.add_primitive_to_dsl(
            [new_primitive_name], [implementation]
        )

        # Create the DSL object
        new_dsl_object = self.make_dsl(new_primitive_name, implementation)

        return self.grammar, updated_dsl_fn(), new_dsl_object

    def offline_loader_full_version(self, run_id: str) -> tuple[
        Grammar[str, int, int],
        dict[str, Any],
        DSL[LocalProgram, GridInput, Any],
    ]:
        """Load the grammar, DSL functions, and DSL object from a previous run
        with multiple proposals."""
        base_dir = Path(__file__).parent
        output_path = base_dir / "outputs" / run_id
        self.output_path = output_path
        self.run_id = run_id

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

        # Extract the proposals
        proposals = metadata["proposal"]
        if not isinstance(proposals, list):
            raise ValueError("Expected 'proposal' to be a list of primitives.")

        # Dynamically construct the file path using run_id
        output_dir = Path(__file__).parent / "outputs" / run_id
        python_files = list(output_dir.glob("*.py"))
        if not python_files:
            raise FileNotFoundError(f"No Python files found in {output_dir}")

        # Collect implementations
        new_primitives: dict[str, Callable[..., Any]] = {}
        for primitive in proposals:
            new_primitive_name = primitive["name"]
            python_str = primitive["semantics_py_stub"]

            # Write each primitive to its own .py file
            self.write_python_file(new_primitive_name, python_str)

            # Load the implementation dynamically
            implementation = self.load_function_from_file(
                str(output_dir / f"{new_primitive_name}.py"), new_primitive_name
            )

            # Accumulate for DSL update later
            new_primitives[new_primitive_name] = implementation

        updated_dsl_fn = self.add_primitive_to_dsl(
            list(new_primitives.keys()), list(new_primitives.values())
        )

        # Create the DSL object
        new_dsl_object = DSL(
            id="grid_v1_full_version",
            primitives=new_primitives,
            evaluate_fn=_eval,
        )

        return self.grammar, updated_dsl_fn(), new_dsl_object
