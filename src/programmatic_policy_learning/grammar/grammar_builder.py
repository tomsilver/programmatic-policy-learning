"""Grammar builder for constructing context-free grammars for logical program
synthesis."""

from typing import Any

from programmatic_policy_learning.grammar.constants import (
    CONDITION,
    DIRECTION,
    LOCAL_PROGRAM,
    NEGATIVE_NUM,
    POSITIVE_NUM,
    START,
    VALUE,
)


class GrammarBuilder:
    """Builds grammars for different environments."""

    ### Grammatical Prior
    START, CONDITION, LOCAL_PROGRAM, DIRECTION, POSITIVE_NUM, NEGATIVE_NUM, VALUE = (
        range(7)
    )

    @staticmethod
    def get_object_types(base_class_name: str) -> tuple[str, ...]:
        """Get object types for the given environment."""
        if base_class_name == "TwoPileNim":
            return ("tpn.EMPTY", "tpn.TOKEN", "None")
        if base_class_name == "CheckmateTactic":
            return (
                "ct.EMPTY",
                "ct.HIGHLIGHTED_WHITE_QUEEN",
                "ct.BLACK_KING",
                "ct.HIGHLIGHTED_WHITE_KING",
                "ct.WHITE_KING",
                "ct.WHITE_QUEEN",
                "None",
            )
        if base_class_name == "StopTheFall":
            return (
                "stf.EMPTY",
                "stf.FALLING",
                "stf.RED",
                "stf.STATIC",
                "stf.ADVANCE",
                "stf.DRAWN",
                "None",
            )
        if base_class_name == "Chase":
            return (
                "ec.EMPTY",
                "ec.TARGET",
                "ec.AGENT",
                "ec.WALL",
                "ec.DRAWN",
                "ec.LEFT_ARROW",
                "ec.RIGHT_ARROW",
                "ec.UP_ARROW",
                "ec.DOWN_ARROW",
                "None",
            )
        if base_class_name == "ReachForTheStar":
            return (
                "rfts.EMPTY",
                "rfts.AGENT",
                "rfts.STAR",
                "rfts.DRAWN",
                "rfts.LEFT_ARROW",
                "rfts.RIGHT_ARROW",
                "None",
            )

        raise ValueError(f"Unknown class name: {base_class_name}")

    @staticmethod
    def create_grammar(object_types: dict[str, Any]) -> Any:
        """Create grammar from object types."""
        grammar = {
            START: (
                [
                    [
                        "at_cell_with_value(",
                        VALUE,
                        ",",
                        LOCAL_PROGRAM,
                        ", s)",
                    ],
                    ["at_action_cell(", LOCAL_PROGRAM, ", a, s)"],
                ],
                [0.5, 0.5],
            ),
            LOCAL_PROGRAM: (
                [
                    [CONDITION],
                    [
                        "lambda cell,o : shifted(",
                        DIRECTION,
                        ",",
                        CONDITION,
                        ", cell, o)",
                    ],
                ],
                [0.5, 0.5],
            ),
            CONDITION: (
                [
                    [
                        "lambda cell,o : cell_is_value(",
                        VALUE,
                        ", cell, o)",
                    ],
                    [
                        "lambda cell,o : scanning(",
                        DIRECTION,
                        ",",
                        LOCAL_PROGRAM,
                        ",",
                        LOCAL_PROGRAM,
                        ", cell, o)",
                    ],
                ],
                [0.5, 0.5],
            ),
            DIRECTION: (
                [
                    ["(", POSITIVE_NUM, ", 0)"],
                    ["(0,", POSITIVE_NUM, ")"],
                    ["(", NEGATIVE_NUM, ", 0)"],
                    ["(0,", NEGATIVE_NUM, ")"],
                    [
                        "(",
                        POSITIVE_NUM,
                        ",",
                        POSITIVE_NUM,
                        ")",
                    ],
                    [
                        "(",
                        NEGATIVE_NUM,
                        ",",
                        POSITIVE_NUM,
                        ")",
                    ],
                    [
                        "(",
                        POSITIVE_NUM,
                        ",",
                        NEGATIVE_NUM,
                        ")",
                    ],
                    [
                        "(",
                        NEGATIVE_NUM,
                        ",",
                        NEGATIVE_NUM,
                        ")",
                    ],
                ],
                [1.0 / 8] * 8,
            ),
            POSITIVE_NUM: (
                [["1"], [POSITIVE_NUM, "+1"]],
                [0.99, 0.01],
            ),
            NEGATIVE_NUM: (
                [["-1"], [NEGATIVE_NUM, "-1"]],
                [0.99, 0.01],
            ),
            VALUE: (
                object_types,
                [1.0 / len(object_types) for _ in object_types],
            ),
        }
        return grammar
