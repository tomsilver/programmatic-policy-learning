# template_expand.py
from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PLACEHOLDER_RE = re.compile(r"\$\{(TOKEN|DRs|K)\}")
TOKEN_CODE_RE = re.compile(r"^(stf|rfts)\.[A-Z_][A-Z0-9_]*$")

# Replace the first function def name at the start of the source.
# Supports: "def f2(s, a):" or "def something_else(s,a):"
DEF_RE = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)

TokenCode = str
Dir = Tuple[int, int]


@dataclass(frozen=True)
class Domains:
    tokens: Sequence[TokenCode]
    dr_lists: Sequence[List[Dir]]
    ks: Sequence[int]


def all_nonempty_dir_lists(dirs8: Sequence[Dir]) -> List[List[Dir]]:
    out: List[List[Dir]] = []
    n = len(dirs8)
    for r in range(1, n + 1):
        for comb in itertools.combinations(dirs8, r):
            out.append(list(comb))
    return out


def extract_placeholders(src: str) -> List[str]:
    return sorted(set(m.group(1) for m in PLACEHOLDER_RE.finditer(src)))


def render_placeholder(name: str, value: Any) -> str:
    if name == "TOKEN":
        tok = str(value).strip()
        if not TOKEN_CODE_RE.match(tok):
            raise ValueError(
                f"Invalid TOKEN expansion '{tok}'. Must be like stf.FALLING or rfts.DRAWN (no quotes)."
            )
        return tok
    if name == "K":
        return str(int(value))
    if name == "DRs":
        items = ", ".join(f"({int(dr)},{int(dc)})" for dr, dc in value)
        return "[" + items + "]"
    raise ValueError(f"Unknown placeholder: {name}")


def substitute(src: str, assignment: Dict[str, Any]) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in assignment:
            raise KeyError(f"Missing binding for {key}")
        return render_placeholder(key, assignment[key])

    return PLACEHOLDER_RE.sub(repl, src)


def build_token_map(tokens: Sequence[str]) -> dict[str, str]:
    prefixes = {t.split(".", 1)[0] for t in tokens if "." in t}
    if "rfts" in prefixes:
        return {
            "#": "rfts.DRAWN",
            "*": "rfts.STAR",
            ".": "rfts.EMPTY",
            "<": "rfts.LEFT_ARROW",
            ">": "rfts.RIGHT_ARROW",
            "A": "rfts.AGENT",
        }
    if "stf" in prefixes:
        return {
            ".": "stf.EMPTY",
            "A": "stf.ADVANCE",
            "D": "stf.DRAWN",
            "F": "stf.FALLING",
            "R": "stf.RED",
            "S": "stf.STATIC",
        }
    return {}


def normalize_template(src: str, token_map: dict[str, str]) -> str:
    """Normalize template source to canonical stf.* tokens.

    - Replace quoted ${TOKEN} with ${TOKEN}
    - Replace quoted single-char tokens (., A, D, F, R, S) with stf.*
    """
    # Remove quotes around ${TOKEN}
    src = re.sub(r"(['\"])\\$\\{TOKEN\\}\\1", "${TOKEN}", src)

    if token_map:
        chars = "".join(re.escape(ch) for ch in token_map.keys())

        def repl_char(m: re.Match) -> str:
            ch = m.group(2)
            return token_map.get(ch, m.group(0))

        # Replace quoted single-char literals where we know the mapping
        src = re.sub(rf"(['\"])({chars})\\1", repl_char, src)
    return src


def find_bad_token_quotes(
    src: str, token_map: dict[str, str], token_prefixes: Sequence[str]
) -> Optional[str]:
    """Return a short description of a bad quoted token if found, else None.

    We only reject quoted char tokens or quoted ${TOKEN}/stf.*.
    """
    if token_map:
        chars = "".join(re.escape(ch) for ch in token_map.keys())
        m = re.search(rf"(['\"])({chars})\\1", src)
        if m:
            return f"quoted char token {m.group(0)}"
    m = re.search(r"(['\"])\\$\\{TOKEN\\}\\1", src)
    if m:
        return f"quoted placeholder {m.group(0)}"
    if token_prefixes:
        prefixes = "|".join(re.escape(p) for p in token_prefixes)
        m = re.search(rf"(['\"])({prefixes})\\.[A-Z_][A-Z0-9_]*\\1", src)
        if m:
            return f"quoted token {m.group(0)}"
    return None


def rename_def(source: str, new_fn_name: str) -> str:
    """Ensure the function defined inside `source` matches new_fn_name.

    Replaces the first 'def <name>(' occurrence.
    """
    m = DEF_RE.search(source)
    if not m:
        raise ValueError(
            "Could not find a function definition line starting with 'def ...('"
        )
    start, end = m.span(1)  # span of the captured function name
    return source[:start] + new_fn_name + source[end:]


def build_char_token_compare_re(token_map: dict[str, str]) -> Optional[re.Pattern]:
    if not token_map:
        return None
    chars = "|".join(re.escape(ch) for ch in token_map.keys())
    return re.compile(rf"(==|!=|in|not\s+in)\s*(['\"])({chars})\\2")


def assert_no_char_token_literals(src: str, token_map: dict[str, str]) -> None:
    char_re = build_char_token_compare_re(token_map)
    if char_re is None:
        return
    m = char_re.search(src)
    if m:
        print(src)
        raise ValueError(
            f"Found forbidden char-token literal comparison: {m.group(0)}\n"
            "Templates must use ${TOKEN} (or token-domain), not '.', 'A', etc."
        )


def expand_templates(
    payload: Union[str, Dict[str, Any]],
    domains: Domains,
    *,
    max_expansions_per_template: Optional[int] = None,
) -> Dict[str, Any]:
    if isinstance(payload, str):
        data = json.loads(payload)
    else:
        data = copy.deepcopy(payload)

    for t in domains.tokens:
        if not TOKEN_CODE_RE.match(t.strip()):
            raise ValueError(
                f"Bad token domain entry: {t!r}. Use stf.NAME or rfts.NAME (no quotes)."
            )

    token_map = build_token_map(domains.tokens)
    token_prefixes = sorted({t.split(".", 1)[0] for t in domains.tokens if "." in t})

    expanded: List[Dict[str, str]] = []
    next_id = 1

    for feat in data.get("features", []):
        src = normalize_template(feat["source"], token_map)
        try:
            assert_no_char_token_literals(src, token_map)
        except ValueError as e:
            print(e)
            continue  # skip this template feature

        phs = extract_placeholders(src)

        choices: List[Tuple[str, Sequence[Any]]] = []
        for ph in phs:
            if ph == "TOKEN":
                choices.append((ph, domains.tokens))
            elif ph == "DRs":
                choices.append((ph, domains.dr_lists))
            elif ph == "K":
                choices.append((ph, domains.ks))
            else:
                raise ValueError(f"Unexpected placeholder {ph}")

        def emit(inst_src: str) -> None:
            nonlocal next_id
            fid = f"f{next_id}"
            inst_src = rename_def(inst_src, fid)  # <-- FIX: align def-name with id
            bad = find_bad_token_quotes(inst_src, token_map, token_prefixes)
            if bad is not None:
                raise ValueError(
                    f"Expanded source contains quotes around tokens ({bad})."
                )
            expanded.append({"id": fid, "name": fid, "source": inst_src})
            next_id += 1

        if not choices:
            emit(src)
            continue

        keys = [k for k, _ in choices]
        vals = [v for _, v in choices]

        count = 0
        for combo in itertools.product(*vals):
            assignment = dict(zip(keys, combo))
            inst_src = substitute(src, assignment)
            emit(inst_src)

            count += 1
            if (
                max_expansions_per_template is not None
                and count >= max_expansions_per_template
            ):
                break

    return {"features": expanded}


def write_json(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def write_jsonl(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for feat in payload.get("features", []):
            f.write(json.dumps(feat) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", required=True, help="Input template features JSON")
    p.add_argument("--out", required=True, help="Output path (.json or .jsonl)")
    p.add_argument("--format", choices=["json", "jsonl"], default="json")
    p.add_argument(
        "--cap", type=int, default=None, help="Optional cap per template expansion"
    )
    args = p.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        template_payload = json.load(f)

    # TOKENS = ["stf.EMPTY", "stf.ADVANCE", "stf.DRAWN", "stf.FALLING", "stf.RED", "stf.STATIC"]
    TOKENS = [
        "rfts.DRAWN",
        "rfts.STAR",
        "rfts.EMPTY",
        "rfts.LEFT_ARROW",
        "rfts.RIGHT_ARROW",
        "rfts.AGENT",
    ]
    DIRS8: List[Dir] = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]
    KS = [1, 2, 3]
    DR_LISTS = all_nonempty_dir_lists(DIRS8)

    domains = Domains(tokens=TOKENS, dr_lists=DR_LISTS, ks=KS)
    expanded = expand_templates(
        template_payload, domains, max_expansions_per_template=args.cap
    )

    if args.format == "json":
        write_json(expanded, args.out)
    else:
        write_jsonl(expanded, args.out)

    print(f"Expanded features: {len(expanded['features'])}")


if __name__ == "__main__":
    main()
