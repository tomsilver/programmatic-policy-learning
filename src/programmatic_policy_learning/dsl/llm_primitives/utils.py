"""Util for LLM-based Primitives Generation."""

import json
from typing import Any, Callable

from prpl_llm_utils.reprompting import RepromptCheck, create_reprompt_from_error_message
from prpl_llm_utils.structs import Query, Response


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


def create_function_from_stub(stub: str, function_name: str) -> Callable[..., Any]:
    """Convert a Python stub string into a callable function."""
    stub = stub.replace("\\n", "\n")
    stub = stub.replace("...", "pass")
    local_namespace: dict[str, Any] = {}
    exec(stub, {}, local_namespace)  # pylint: disable=exec-used
    return local_namespace[function_name]
