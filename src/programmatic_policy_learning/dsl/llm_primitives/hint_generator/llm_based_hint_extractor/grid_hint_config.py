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




HINT_EXTRACTION_BIASED_PROMPT_TEMPLATE = """
    You are analyzing expert demonstrations from a grid-based environment.

    We ONLY control the agent, not other entities.

    Your job:
    1) Infer the agent’s high-level objective.
    2) Extract RECURRING spatial–relational patterns used for decision-making.
    3) Identify directional asymmetries, alignment behavior, and distance-based cues.
    4) Produce NON-CHEATY hints that could guide a symbolic policy or DSL design.

    Hard constraints:
    - Do NOT propose DSL primitives, function names, or code.
    - Use descriptive relational language only.
    - Focus on observable spatial relations, not abstract strategy names.
    - If uncertain, say so.

    You are given a sequence of expert state transitions below.

    Each step contains:
    - ASCII grid of the state
    - Coordinate-based object listing
    - Action taken
    - Observed transition effects

    ---

    {TRAJECTORY_TEXT}

    ---

    Output ONLY the following template:

    ## DEMONSTRATION-INFERRED FEATURES (HINTS)

    ### High-frequency relational patterns:
    - ...

    ### Useful directional / asymmetry relations:
    - ...

    ### Example state–action correlations:
    - ...

    ### Frequently observed local spatial configurations:
    - ...

    ### Observed distance thresholds or step ranges:
    - ...
    """
