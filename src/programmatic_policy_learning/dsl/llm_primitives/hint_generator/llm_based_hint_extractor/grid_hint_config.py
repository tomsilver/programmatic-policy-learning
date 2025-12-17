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
    "Checkmate": [
        "black_king",
        "white_king",
        "white_queen",
        "highlighted_white_king",
        "highlighted_white_queen",
    ],
}
