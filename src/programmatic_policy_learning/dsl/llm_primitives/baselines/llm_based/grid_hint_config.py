"""Domain-specific token whitelists for hint extraction."""

SALIENT_TOKENS = {
    "Chase": [
        "agent",
        "target",
        "wall",
        "drawn",
        "left_arrow",
        "right_arrow",
        "up_arrow",
        "down_arrow",
    ],
    "TwoPileNim": ["token"],
    "ReachForTheStar": ["agent", "star", "left_arrow", "right_arrow", "drawn"],
    "StopTheFall": ["falling", "red", "static", "advance"],
    "CheckmateTactic": [
        "black_king",
        "white_king",
        "white_queen",
        "highlighted_white_king",
        "highlighted_white_queen",
    ],
}

SYMBOL_MAPS = {
    "Chase": {
        "empty": ".",
        "agent": "A",
        "target": "T",
        "wall": "#",
        "drawn": "*",
        "left_arrow": "<",
        "right_arrow": ">",
        "up_arrow": "^",
        "down_arrow": "v",
    },
    # "Chase": {
    #     "empty": "0",
    #     "agent": "1",
    #     "target": "2",
    #     "wall": "3",
    #     "drawn": "4",
    #     "left_arrow": "5",
    #     "right_arrow": "6",
    #     "up_arrow": "7",
    #     "down_arrow": "8",
    # },
    "TwoPileNim": {
        "empty": ".",
        "token": "*",
    },
    "ReachForTheStar": {
        "empty": ".",
        "agent": "A",
        "star": "*",
        "left_arrow": "<",
        "right_arrow": ">",
        "drawn": "#",
    },
    "StopTheFall": {
        "empty": ".",
        "falling": "F",
        "red": "R",
        "static": "S",
        "advance": "A",
    },
    "CheckmateTactic": {
        "empty": ".",
        "black_king": "K",
        "white_king": "k",
        "white_queen": "Q",
        "highlighted_white_king": "k*",
        "highlighted_white_queen": "Q*",
    },
}


def get_symbol_map(env_name: str) -> dict[str, str]:
    """Return the ASCII symbol map for the given environment."""
    try:
        return SYMBOL_MAPS[env_name]
    except KeyError as exc:
        raise KeyError(f"No symbol map configured for {env_name}") from exc
