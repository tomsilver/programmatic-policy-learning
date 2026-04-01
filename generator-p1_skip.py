def build_feature_library():
    # Helper to generate function source code as string
    def make_feature(fid, name, body_lines):
        src = f"def {fid}(s, a):\n"
        for line in body_lines:
            src += "    " + line + "\n"
        return {"id": fid, "name": name, "source": src}

    features = []
    fid = 1

    # Indexes for readability
    robot_x = 0
    robot_y = 1
    robot_theta = 2
    robot_base_radius = 3
    robot_arm_joint = 4
    robot_arm_length = 5
    robot_vacuum = 6
    robot_gripper_height = 7
    robot_gripper_width = 8

    target_x = 9
    target_y = 10
    target_theta = 11
    target_static = 12
    target_color_r = 13
    target_color_g = 14
    target_color_b = 15
    target_z_order = 16
    target_width = 17
    target_height = 18

    obstacle0_x = 19
    obstacle0_y = 20
    obstacle0_theta = 21
    obstacle0_static = 22
    obstacle0_color_r = 23
    obstacle0_color_g = 24
    obstacle0_color_b = 25
    obstacle0_z_order = 26
    obstacle0_width = 27
    obstacle0_height = 28

    obstacle1_x = 29
    obstacle1_y = 30
    obstacle1_theta = 31
    obstacle1_static = 32
    obstacle1_color_r = 33
    obstacle1_color_g = 34
    obstacle1_color_b = 35
    obstacle1_z_order = 36
    obstacle1_width = 37
    obstacle1_height = 38

    dx = 0
    dy = 1
    dtheta = 2
    darm = 3
    vac = 4

    # Feature templates

    # 1. Relative position to target center
    features.append(make_feature(
        f"f{fid}",
        "robot_left_of_target_center",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "return s[0] < target_cx"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_right_of_target_center",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "return s[0] > target_cx"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_below_target_center",
        [
            "target_cy = s[10] + s[18] / 2.0",
            "return s[1] < target_cy"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_above_target_center",
        [
            "target_cy = s[10] + s[18] / 2.0",
            "return s[1] > target_cy"
        ]
    ))
    fid += 1

    # 2. Distance to target center (L2 and Manhattan)
    features.append(make_feature(
        f"f{fid}",
        "robot_close_to_target_center_l2_0.3",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "dx = s[0] - target_cx",
            "dy = s[1] - target_cy",
            "dist = (dx * dx + dy * dy) ** 0.5",
            "return dist < 0.3"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_far_from_target_center_l2_1.0",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "dx = s[0] - target_cx",
            "dy = s[1] - target_cy",
            "dist = (dx * dx + dy * dy) ** 0.5",
            "return dist > 1.0"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_manhattan_dist_to_target_lt_0.5",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "dx = abs(s[0] - target_cx)",
            "dy = abs(s[1] - target_cy)",
            "return dx + dy < 0.5"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_manhattan_dist_to_target_gt_2.0",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "dx = abs(s[0] - target_cx)",
            "dy = abs(s[1] - target_cy)",
            "return dx + dy > 2.0"
        ]
    ))
    fid += 1

    # 3. Is robot inside target region
    features.append(make_feature(
        f"f{fid}",
        "robot_inside_target_region",
        [
            "target_x = s[9]",
            "target_y = s[10]",
            "target_w = s[17]",
            "target_h = s[18]",
            "return (s[0] >= target_x) and (s[0] <= target_x + target_w) and (s[1] >= target_y) and (s[1] <= target_y + target_h)"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_outside_target_region",
        [
            "target_x = s[9]",
            "target_y = s[10]",
            "target_w = s[17]",
            "target_h = s[18]",
            "return not ((s[0] >= target_x) and (s[0] <= target_x + target_w) and (s[1] >= target_y) and (s[1] <= target_y + target_h))"
        ]
    ))
    fid += 1

    # 4. Passage/gap alignment for obstacle0 (vertical wall)
    for obs_idx, obs_x, obs_y, obs_w, obs_h in [
        (0, 19, 20, 27, 28),  # obstacle0
        (1, 29, 30, 37, 38)   # obstacle1
    ]:
        features.append(make_feature(
            f"f{fid}",
            f"robot_left_of_obstacle{obs_idx}_wall",
            [
                f"wall_x = s[{obs_x}]",
                f"r = s[3]",
                f"return s[0] + r < wall_x"
            ]
        ))
        fid += 1

        features.append(make_feature(
            f"f{fid}",
            f"robot_right_of_obstacle{obs_idx}_wall",
            [
                f"wall_x = s[{obs_x}]",
                f"wall_w = s[{obs_w}]",
                f"r = s[3]",
                f"return s[0] - r > wall_x + wall_w"
            ]
        ))
        fid += 1

        features.append(make_feature(
            f"f{fid}",
            f"robot_in_obstacle{obs_idx}_zone",
            [
                f"wall_x = s[{obs_x}]",
                f"wall_w = s[{obs_w}]",
                f"r = s[3]",
                f"return (s[0] + r >= wall_x) and (s[0] - r <= wall_x + wall_w)"
            ]
        ))
        fid += 1

        # y-alignment with gap (for vertical wall)
        features.append(make_feature(
            f"f{fid}",
            f"robot_y_aligned_with_obstacle{obs_idx}_gap",
            [
                f"gap_lower = s[{obs_y}] + s[{obs_h}] + s[3]",
                f"gap_upper = s[{obs_y + 10}] - s[{obs_h + 10}] - s[3]",
                f"return (s[1] >= gap_lower) and (s[1] <= gap_upper)"
            ]
        ))
        fid += 1

    # 5. Clearance to obstacles (robot center to obstacle rectangle edge)
    for obs_idx, obs_x, obs_y, obs_w, obs_h in [
        (0, 19, 20, 27, 28),
        (1, 29, 30, 37, 38)
    ]:
        features.append(make_feature(
            f"f{fid}",
            f"robot_clear_of_obstacle{obs_idx}_by_0.1",
            [
                f"ox = s[{obs_x}]",
                f"oy = s[{obs_y}]",
                f"ow = s[{obs_w}]",
                f"oh = s[{obs_h}]",
                f"rx = s[0]",
                f"ry = s[1]",
                f"r = s[3]",
                f"closest_x = min(max(rx, ox), ox + ow)",
                f"closest_y = min(max(ry, oy), oy + oh)",
                f"dist = ((rx - closest_x) ** 2 + (ry - closest_y) ** 2) ** 0.5",
                f"return dist > r + 0.1"
            ]
        ))
        fid += 1

    # 6. Progress direction: is action moving toward target center?
    features.append(make_feature(
        f"f{fid}",
        "action_moves_toward_target_center",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "to_target_x = target_cx - s[0]",
            "to_target_y = target_cy - s[1]",
            "dot = to_target_x * a[0] + to_target_y * a[1]",
            "return dot > 0"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "action_moves_away_from_target_center",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "to_target_x = target_cx - s[0]",
            "to_target_y = target_cy - s[1]",
            "dot = to_target_x * a[0] + to_target_y * a[1]",
            "return dot < 0"
        ]
    ))
    fid += 1

    # 7. Action directionality (signs)
    features.append(make_feature(
        f"f{fid}",
        "action_dx_positive",
        [
            "return a[0] > 0"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "action_dx_negative",
        [
            "return a[0] < 0"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "action_dy_positive",
        [
            "return a[1] > 0"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "action_dy_negative",
        [
            "return a[1] < 0"
        ]
    ))
    fid += 1

    # 8. Action magnitude
    features.append(make_feature(
        f"f{fid}",
        "action_translation_magnitude_gt_0.04",
        [
            "mag = (a[0] * a[0] + a[1] * a[1]) ** 0.5",
            "return mag > 0.04"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "action_translation_magnitude_lt_0.02",
        [
            "mag = (a[0] * a[0] + a[1] * a[1]) ** 0.5",
            "return mag < 0.02"
        ]
    ))
    fid += 1

    # 9. Is action mostly along x or y
    features.append(make_feature(
        f"f{fid}",
        "action_is_mostly_x",
        [
            "return abs(a[0]) > abs(a[1])"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "action_is_mostly_y",
        [
            "return abs(a[1]) > abs(a[0])"
        ]
    ))
    fid += 1

    # 10. Is robot before, inside, or past passage_0 wall (vertical wall at obstacle0_x)
    features.append(make_feature(
        f"f{fid}",
        "robot_before_passage0_wall",
        [
            "wall_x = s[19]",
            "r = s[3]",
            "return s[0] + r < wall_x"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_inside_passage0_zone",
        [
            "wall_x = s[19]",
            "wall_w = s[27]",
            "r = s[3]",
            "return (s[0] + r >= wall_x) and (s[0] - r <= wall_x + wall_w)"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_past_passage0_wall",
        [
            "wall_x = s[19]",
            "wall_w = s[27]",
            "r = s[3]",
            "return s[0] - r > wall_x + wall_w"
        ]
    ))
    fid += 1

    # 11. Is robot y aligned with passage_0 gap (between obstacle0 and obstacle1)
    features.append(make_feature(
        f"f{fid}",
        "robot_y_aligned_with_passage0_gap",
        [
            "gap_lower = s[20] + s[28] + s[3]",
            "gap_upper = s[30] - s[38] - s[3]",
            "return (s[1] >= gap_lower) and (s[1] <= gap_upper)"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_y_below_passage0_gap",
        [
            "gap_lower = s[20] + s[28] + s[3]",
            "return s[1] < gap_lower"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "robot_y_above_passage0_gap",
        [
            "gap_upper = s[30] - s[38] - s[3]",
            "return s[1] > gap_upper"
        ]
    ))
    fid += 1

    # 12. Is action moving robot toward passage0 gap center (y)
    features.append(make_feature(
        f"f{fid}",
        "action_moves_toward_passage0_gap_center_y",
        [
            "gap_lower = s[20] + s[28] + s[3]",
            "gap_upper = s[30] - s[38] - s[3]",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "to_gap = gap_center - s[1]",
            "return to_gap * a[1] > 0"
        ]
    ))
    fid += 1

    # 13. Is action moving robot toward passage0 wall (x)
    features.append(make_feature(
        f"f{fid}",
        "action_moves_toward_passage0_wall_x",
        [
            "wall_x = s[19]",
            "to_wall = wall_x - s[0]",
            "return to_wall * a[0] > 0"
        ]
    ))
    fid += 1

    # 14. Is action moving robot away from passage0 wall (x)
    features.append(make_feature(
        f"f{fid}",
        "action_moves_away_from_passage0_wall_x",
        [
            "wall_x = s[19]",
            "to_wall = wall_x - s[0]",
            "return to_wall * a[0] < 0"
        ]
    ))
    fid += 1

    # 15. Is robot close to passage0 wall (within 0.05)
    features.append(make_feature(
        f"f{fid}",
        "robot_close_to_passage0_wall_0.05",
        [
            "wall_x = s[19]",
            "r = s[3]",
            "dist = abs((s[0] + r) - wall_x)",
            "return dist < 0.05"
        ]
    ))
    fid += 1

    # 16. Is robot y near passage0 gap center (within 0.1)
    features.append(make_feature(
        f"f{fid}",
        "robot_y_near_passage0_gap_center_0.1",
        [
            "gap_lower = s[20] + s[28] + s[3]",
            "gap_upper = s[30] - s[38] - s[3]",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "return abs(s[1] - gap_center) < 0.1"
        ]
    ))
    fid += 1

    # 17. Is robot y far from passage0 gap center (greater than 0.3)
    features.append(make_feature(
        f"f{fid}",
        "robot_y_far_from_passage0_gap_center_0.3",
        [
            "gap_lower = s[20] + s[28] + s[3]",
            "gap_upper = s[30] - s[38] - s[3]",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "return abs(s[1] - gap_center) > 0.3"
        ]
    ))
    fid += 1

    # 18. Is action rotation significant
    features.append(make_feature(
        f"f{fid}",
        "action_rotation_gt_0.1",
        [
            "return abs(a[2]) > 0.1"
        ]
    ))
    fid += 1

    features.append(make_feature(
        f"f{fid}",
        "action_rotation_lt_0.05",
        [
            "return abs(a[2]) < 0.05"
        ]
    ))
    fid += 1

    # 19. Is action arm movement significant
    features.append(make_feature(
        f"f{fid}",
        "action_arm_movement_gt_0.05",
        [
            "return abs(a[3]) > 0.05"
        ]
    ))
    fid += 1

    # 20. Is action vacuum/gripper active
    features.append(make_feature(
        f"f{fid}",
        "action_vacuum_active",
        [
            "return abs(a[4]) > 0.01"
        ]
    ))
    fid += 1

    # 21. Conjunction: robot before passage0 wall AND action moves toward wall
    features.append(make_feature(
        f"f{fid}",
        "robot_before_passage0_wall_and_action_toward_wall",
        [
            "wall_x = s[19]",
            "r = s[3]",
            "before = s[0] + r < wall_x",
            "to_wall = wall_x - s[0]",
            "return before and (to_wall * a[0] > 0)"
        ]
    ))
    fid += 1

    # 22. Conjunction: robot y not aligned with passage0 gap AND action moves toward gap center
    features.append(make_feature(
        f"f{fid}",
        "robot_y_not_aligned_and_action_toward_gap_center",
        [
            "gap_lower = s[20] + s[28] + s[3]",
            "gap_upper = s[30] - s[38] - s[3]",
            "aligned = (s[1] >= gap_lower) and (s[1] <= gap_upper)",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "to_gap = gap_center - s[1]",
            "return (not aligned) and (to_gap * a[1] > 0)"
        ]
    ))
    fid += 1

    # 23. Conjunction: robot past passage0 wall AND action moves toward target
    features.append(make_feature(
        f"f{fid}",
        "robot_past_passage0_wall_and_action_toward_target",
        [
            "wall_x = s[19]",
            "wall_w = s[27]",
            "r = s[3]",
            "past = s[0] - r > wall_x + wall_w",
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "to_target_x = target_cx - s[0]",
            "to_target_y = target_cy - s[1]",
            "dot = to_target_x * a[0] + to_target_y * a[1]",
            "return past and (dot > 0)"
        ]
    ))
    fid += 1

    # 24. Conjunction: robot inside passage0 zone AND action is mostly y
    features.append(make_feature(
        f"f{fid}",
        "robot_inside_passage0_zone_and_action_mostly_y",
        [
            "wall_x = s[19]",
            "wall_w = s[27]",
            "r = s[3]",
            "inside = (s[0] + r >= wall_x) and (s[0] - r <= wall_x + wall_w)",
            "return inside and (abs(a[1]) > abs(a[0]))"
        ]
    ))
    fid += 1

    # 25. Conjunction: robot close to target AND action translation small
    features.append(make_feature(
        f"f{fid}",
        "robot_close_to_target_and_action_small",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "dx = s[0] - target_cx",
            "dy = s[1] - target_cy",
            "dist = (dx * dx + dy * dy) ** 0.5",
            "mag = (a[0] * a[0] + a[1] * a[1]) ** 0.5",
            "return (dist < 0.3) and (mag < 0.02)"
        ]
    ))
    fid += 1

    # 26. Conjunction: robot far from target AND action translation large
    features.append(make_feature(
        f"f{fid}",
        "robot_far_from_target_and_action_large",
        [
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "dx = s[0] - target_cx",
            "dy = s[1] - target_cy",
            "dist = (dx * dx + dy * dy) ** 0.5",
            "mag = (a[0] * a[0] + a[1] * a[1]) ** 0.5",
            "return (dist > 1.0) and (mag > 0.04)"
        ]
    ))
    fid += 1

    # 27. Conjunction: robot y far from passage0 gap center AND action moves toward gap center
    features.append(make_feature(
        f"f{fid}",
        "robot_y_far_from_gap_center_and_action_toward_gap",
        [
            "gap_lower = s[20] + s[28] + s[3]",
            "gap_upper = s[30] - s[38] - s[3]",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "offset = s[1] - gap_center",
            "to_gap = gap_center - s[1]",
            "return (abs(offset) > 0.3) and (to_gap * a[1] > 0)"
        ]
    ))
    fid += 1

    # 28. Conjunction: robot before passage0 wall AND action is mostly x
    features.append(make_feature(
        f"f{fid}",
        "robot_before_passage0_wall_and_action_mostly_x",
        [
            "wall_x = s[19]",
            "r = s[3]",
            "before = s[0] + r < wall_x",
            "return before and (abs(a[0]) > abs(a[1]))"
        ]
    ))
    fid += 1

    # 29. Conjunction: robot inside target region AND action translation small
    features.append(make_feature(
        f"f{fid}",
        "robot_inside_target_and_action_small",
        [
            "target_x = s[9]",
            "target_y = s[10]",
            "target_w = s[17]",
            "target_h = s[18]",
            "inside = (s[0] >= target_x) and (s[0] <= target_x + target_w) and (s[1] >= target_y) and (s[1] <= target_y + target_h)",
            "mag = (a[0] * a[0] + a[1] * a[1]) ** 0.5",
            "return inside and (mag < 0.02)"
        ]
    ))
    fid += 1

    # 30. Conjunction: robot outside target region AND action moves toward target
    features.append(make_feature(
        f"f{fid}",
        "robot_outside_target_and_action_toward_target",
        [
            "target_x = s[9]",
            "target_y = s[10]",
            "target_w = s[17]",
            "target_h = s[18]",
            "inside = (s[0] >= target_x) and (s[0] <= target_x + target_w) and (s[1] >= target_y) and (s[1] <= target_y + target_h)",
            "target_cx = s[9] + s[17] / 2.0",
            "target_cy = s[10] + s[18] / 2.0",
            "to_target_x = target_cx - s[0]",
            "to_target_y = target_cy - s[1]",
            "dot = to_target_x * a[0] + to_target_y * a[1]",
            "return (not inside) and (dot > 0)"
        ]
    ))
    fid += 1

    return {"features": features}

import json
if __name__ == "__main__":
    feature_set = build_feature_library()
    #write to json file
    with open("feature_library_skip.json", "w") as f:
        json.dump(feature_set, f, indent=4)
