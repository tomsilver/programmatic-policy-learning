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
    # "Chase": {
    #     "empty": ".",
    #     "agent": "A",
    #     "target": "T",
    #     "wall": "#",
    #     "drawn": "*",
    #     "left_arrow": "<",
    #     "right_arrow": ">",
    #     "up_arrow": "^",
    #     "down_arrow": "v",
    # },
    "Chase": {
        "empty": "0",
        "agent": "1",
        "target": "2",
        "wall": "3",
        "drawn": "4",
        "left_arrow": "5",
        "right_arrow": "6",
        "up_arrow": "7",
        "down_arrow": "8",
    },
    "TwoPileNim": {
        "empty": "0",
        "token": "1",
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


nim_game_description = """
        Environment description:

        - The environment is a grid-based implementation of the game 'Nim'.
        - The observation `obs` is a 2D Python NumPy array (rows × columns).
        - Do not use boolean checks such as `if obs`, `if not obs`, or `obs == []`. 
        - Each cell contains one of the following values:
        - 'empty'
        - 'token'


        ## Game Dynamics and Rules (TwoPileNim)

        - The grid encodes a two-pile Nim position using **two token columns** (two vertical stacks of `'token'`).
        - **Each pile corresponds to one column**: the number of `'token'` cells in that column is the pile size.
        - The game is **turn-based**. On each agent turn, the policy selects exactly one action `(row, col)`.

        ### Action meaning
        - The action `(row, col)` indicates choosing **pile `col`** and removing tokens according to the selected **row**.
        - A move is legal only if the selected `(row, col)` corresponds to a location in a token-stack column that results in removing **at least one** token from that pile.
        - Intuitively: selecting a cell in a pile column removes tokens **from that cell and all tokens “below it” in the same column** (i.e., it reduces the pile to the tokens strictly above the selected cell).  
        - Selecting higher rows removes fewer tokens; selecting lower rows removes more tokens.

        ### Terminal condition and winner
        - The game ends when **no `'token'` cells remain in the grid** (both piles are empty).
        - The player who makes the move that leaves the grid with **no remaining tokens** wins (standard normal-play Nim).

        ### Optimal play objective
        - The optimal strategy is the standard Nim strategy: on your turn, make a move that leaves the opponent a **losing position** (for two piles, this corresponds to leaving the piles with **equal sizes**, when possible).
"""
