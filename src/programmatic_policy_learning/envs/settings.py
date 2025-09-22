"""Environment settings and object type definitions."""


def get_object_types(base_class_name: str) -> tuple[str, ...]:
    """Get object types for a specific environment class."""
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

    raise ValueError(f"Unknown environment class: {base_class_name}")
