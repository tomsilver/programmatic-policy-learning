def build_feature_library():
    def make_feature(fid, name, body_lines):
        src = f"def {fid}(s, a):\n"
        for line in body_lines:
            src += "    " + line + "\n"
        return {"id": fid, "name": name, "source": src}

    features = []
    fid = 1

    # --- Common geometry variables ---
    def target_center_vars():
        return [
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
        ]

    def passage0_gap_vars():
        return [
            "r = s[3]",
            "gap_lower = s[20] + s[28] + r",
            "gap_upper = s[30] - r",
            "gap_center = (gap_lower + gap_upper) / 2.0",
        ]

    def passage0_wall_vars():
        return [
            "wall_x = s[19]",
            "wall_right = s[19] + s[27] + s[3]",
        ]

    # --- Pre-passage alignment regime ---
    # Robot left of wall and not aligned with gap
    def pre_passage_vars():
        return (
            passage0_wall_vars()
            + passage0_gap_vars()
            + [
                "robot_x = s[0]",
                "robot_y = s[1]",
                "left_of_wall = (robot_x + s[3] < wall_x)",
                "not_aligned = (robot_y < gap_lower) or (robot_y > gap_upper)",
            ]
        )

    # Action dy moves y toward gap center
    features.append(
        make_feature(
            f"f{fid}",
            "prepassage_action_moves_y_toward_gap_center",
            pre_passage_vars()
            + [
                "dy = a[1]",
                "toward = ((robot_y < gap_center) and (dy > 0)) or ((robot_y > gap_center) and (dy < 0))",
                "return left_of_wall and not_aligned and toward",
            ],
        )
    )
    fid += 1

    # Action dy moves y away from gap center
    features.append(
        make_feature(
            f"f{fid}",
            "prepassage_action_moves_y_away_from_gap_center",
            pre_passage_vars()
            + [
                "dy = a[1]",
                "away = ((robot_y < gap_center) and (dy < 0)) or ((robot_y > gap_center) and (dy > 0))",
                "return left_of_wall and not_aligned and away",
            ],
        )
    )
    fid += 1

    # Align-first: |dx| small, dy moves y toward gap center
    features.append(
        make_feature(
            f"f{fid}",
            "prepassage_alignfirst_dx_small_dy_toward_gap_center",
            pre_passage_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "dx_small = abs(dx) < 0.02",
                "toward = ((robot_y < gap_center) and (dy > 0)) or ((robot_y > gap_center) and (dy < 0))",
                "return left_of_wall and not_aligned and dx_small and toward",
            ],
        )
    )
    fid += 1

    # Align-first: |dx| large, dy moves y toward gap center (should be discouraged)
    features.append(
        make_feature(
            f"f{fid}",
            "prepassage_alignfirst_dx_large_dy_toward_gap_center",
            pre_passage_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "dx_large = abs(dx) > 0.04",
                "toward = ((robot_y < gap_center) and (dy > 0)) or ((robot_y > gap_center) and (dy < 0))",
                "return left_of_wall and not_aligned and dx_large and toward",
            ],
        )
    )
    fid += 1

    # Align-first: |dx| small, dy moves y away from gap center (should be discouraged)
    features.append(
        make_feature(
            f"f{fid}",
            "prepassage_alignfirst_dx_small_dy_away_gap_center",
            pre_passage_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "dx_small = abs(dx) < 0.02",
                "away = ((robot_y < gap_center) and (dy < 0)) or ((robot_y > gap_center) and (dy > 0))",
                "return left_of_wall and not_aligned and dx_small and away",
            ],
        )
    )
    fid += 1

    # --- Crossing regime: aligned with gap, left of wall ---
    def crossing_vars():
        return (
            passage0_wall_vars()
            + passage0_gap_vars()
            + [
                "robot_x = s[0]",
                "robot_y = s[1]",
                "left_of_wall = (robot_x + s[3] < wall_x)",
                "aligned = (robot_y >= gap_lower) and (robot_y <= gap_upper)",
            ]
        )

    # Action is mostly positive dx, |dy| small
    features.append(
        make_feature(
            f"f{fid}",
            "crossing_aligned_dx_positive_dy_small",
            crossing_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "dx_pos = dx > 0.01",
                "dy_small = abs(dy) < 0.02",
                "return left_of_wall and aligned and dx_pos and dy_small",
            ],
        )
    )
    fid += 1

    # Action is mostly positive dx, |dy| large (should be discouraged)
    features.append(
        make_feature(
            f"f{fid}",
            "crossing_aligned_dx_positive_dy_large",
            crossing_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "dx_pos = dx > 0.01",
                "dy_large = abs(dy) > 0.04",
                "return left_of_wall and aligned and dx_pos and dy_large",
            ],
        )
    )
    fid += 1

    # Action is negative dx (should be discouraged)
    features.append(
        make_feature(
            f"f{fid}",
            "crossing_aligned_dx_negative",
            crossing_vars()
            + [
                "dx = a[0]",
                "dx_neg = dx < -0.01",
                "return left_of_wall and aligned and dx_neg",
            ],
        )
    )
    fid += 1

    # --- Inside-passage regime ---
    def inside_passage_vars():
        return (
            passage0_wall_vars()
            + passage0_gap_vars()
            + [
                "robot_x = s[0]",
                "robot_y = s[1]",
                "inside_passage = (robot_x >= wall_x) and (robot_x < wall_right)",
            ]
        )

    # Next y remains in gap, dx positive
    features.append(
        make_feature(
            f"f{fid}",
            "insidepassage_next_y_in_gap_dx_positive",
            inside_passage_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "next_y = robot_y + dy",
                "in_gap = (next_y >= gap_lower) and (next_y <= gap_upper)",
                "dx_pos = dx > 0.01",
                "return inside_passage and in_gap and dx_pos",
            ],
        )
    )
    fid += 1

    # Next y leaves gap, dx positive (should be discouraged)
    features.append(
        make_feature(
            f"f{fid}",
            "insidepassage_next_y_out_gap_dx_positive",
            inside_passage_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "next_y = robot_y + dy",
                "out_gap = (next_y < gap_lower) or (next_y > gap_upper)",
                "dx_pos = dx > 0.01",
                "return inside_passage and out_gap and dx_pos",
            ],
        )
    )
    fid += 1

    # Next y in gap, dx negative (should be discouraged)
    features.append(
        make_feature(
            f"f{fid}",
            "insidepassage_next_y_in_gap_dx_negative",
            inside_passage_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "next_y = robot_y + dy",
                "in_gap = (next_y >= gap_lower) and (next_y <= gap_upper)",
                "dx_neg = dx < -0.01",
                "return inside_passage and in_gap and dx_neg",
            ],
        )
    )
    fid += 1

    # --- Post-passage regime ---
    def post_passage_vars():
        return (
            passage0_wall_vars()
            + target_center_vars()
            + [
                "robot_x = s[0]",
                "robot_y = s[1]",
                "passed_wall = (robot_x >= wall_right)",
            ]
        )

    # Action reduces distance to target center (projected)
    features.append(
        make_feature(
            f"f{fid}",
            "postpassage_action_reduces_dist_to_target",
            post_passage_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "next_x = robot_x + dx",
                "next_y = robot_y + dy",
                "dist_now = abs(robot_x - target_cx) + abs(robot_y - target_cy)",
                "dist_next = abs(next_x - target_cx) + abs(next_y - target_cy)",
                "return passed_wall and (dist_next < dist_now)",
            ],
        )
    )
    fid += 1

    # Action increases distance to target center (should be discouraged)
    features.append(
        make_feature(
            f"f{fid}",
            "postpassage_action_increases_dist_to_target",
            post_passage_vars()
            + [
                "dx = a[0]",
                "dy = a[1]",
                "next_x = robot_x + dx",
                "next_y = robot_y + dy",
                "dist_now = abs(robot_x - target_cx) + abs(robot_y - target_cy)",
                "dist_next = abs(next_x - target_cx) + abs(next_y - target_cy)",
                "return passed_wall and (dist_next > dist_now)",
            ],
        )
    )
    fid += 1

    # --- Existing general features (subset for brevity, add more as needed) ---
    features.append(
        make_feature(
            f"f{fid}",
            "robot_left_of_target_center",
            target_center_vars()
            + [
                "robot_x = s[0]",
                "return robot_x < target_cx",
            ],
        )
    )
    fid += 1

    features.append(
        make_feature(
            f"f{fid}",
            "robot_inside_target_region",
            [
                "robot_x = s[0]",
                "robot_y = s[1]",
                "target_x = s[9]",
                "target_y = s[10]",
                "target_w = s[17]",
                "target_h = s[18]",
                "inside_x = (robot_x >= target_x) and (robot_x <= target_x + target_w)",
                "inside_y = (robot_y >= target_y) and (robot_y <= target_y + target_h)",
                "return inside_x and inside_y",
            ],
        )
    )
    fid += 1

    features.append(
        make_feature(
            f"f{fid}",
            "action_dx_toward_target_center",
            target_center_vars()
            + [
                "robot_x = s[0]",
                "dx = a[0]",
                "return ((robot_x < target_cx) and (dx > 0)) or ((robot_x > target_cx) and (dx < 0))",
            ],
        )
    )
    fid += 1

    features.append(
        make_feature(
            f"f{fid}",
            "action_moves_toward_gap_center_y",
            passage0_gap_vars()
            + [
                "robot_y = s[1]",
                "dy = a[1]",
                "toward = ((robot_y < gap_center) and (dy > 0)) or ((robot_y > gap_center) and (dy < 0))",
                "return toward",
            ],
        )
    )
    fid += 1

    features.append(
        make_feature(
            f"f{fid}",
            "action_moves_away_from_gap_center_y",
            passage0_gap_vars()
            + [
                "robot_y = s[1]",
                "dy = a[1]",
                "away = ((robot_y < gap_center) and (dy < 0)) or ((robot_y > gap_center) and (dy > 0))",
                "return away",
            ],
        )
    )
    fid += 1

    features.append(
        make_feature(
            f"f{fid}",
            "action_mostly_forward",
            [
                "dx = a[0]",
                "dy = a[1]",
                "return abs(dx) > abs(dy)",
            ],
        )
    )
    fid += 1

    features.append(
        make_feature(
            f"f{fid}",
            "action_mostly_upward",
            [
                "dx = a[0]",
                "dy = a[1]",
                "return abs(dy) > abs(dx)",
            ],
        )
    )
    fid += 1

    features.append(
        make_feature(
            f"f{fid}",
            "action_small_translation",
            [
                "dx = a[0]",
                "dy = a[1]",
                "mag = (dx * dx + dy * dy) ** 0.5",
                "return mag < 0.05",
            ],
        )
    )
    fid += 1

    features.append(
        make_feature(
            f"f{fid}",
            "action_large_translation",
            [
                "dx = a[0]",
                "dy = a[1]",
                "mag = (dx * dx + dy * dy) ** 0.5",
                "return mag > 0.1",
            ],
        )
    )
    fid += 1

    return {"features": features}


import json

if __name__ == "__main__":
    feature_set = build_feature_library()
    # write to json file
    with open("new.json", "w") as f:
        json.dump(feature_set, f, indent=4)
