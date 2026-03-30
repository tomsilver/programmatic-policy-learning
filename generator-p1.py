def build_feature_library():
    # Helper to generate feature source code as a string
    def feature_source(fid, name, body_lines):
        src = f"def {fid}(s, a):\n"
        for line in body_lines:
            src += "    " + line + "\n"
        return {"id": fid, "name": name, "source": src}

    features = []
    fid = 1

    # --- Helper variable definitions (for use in all features) ---
    # robot_x = s[0], robot_y = s[1], robot_theta = s[2], robot_base_radius = s[3]
    # target_x = s[9], target_y = s[10], target_width = s[17], target_height = s[18]
    # obstacle0_x = s[19], obstacle0_y = s[20], obstacle0_width = s[27], obstacle0_height = s[28]
    # obstacle1_x = s[29], obstacle1_y = s[30], obstacle1_width = s[37], obstacle1_height = s[38]
    # a[0]=dx, a[1]=dy, a[2]=dtheta, a[3]=darm, a[4]=vac

    # --- Feature families ---

    # 1. Relative position to target (robot vs. target center)
    for axis, idx_r, idx_t, idx_w, idx_h, axis_name in [
        ("x", 0, 9, 17, 18, "horizontal"),
        ("y", 1, 10, 18, 17, "vertical"),
    ]:
        # robot left/below/above/right of target center
        features.append(feature_source(
            f"f{fid}",
            f"robot_{'left' if axis=='x' else 'below'}_of_target_center",
            [
                "robot_x = s[0]",
                "robot_y = s[1]",
                "target_x = s[9]",
                "target_y = s[10]",
                "target_width = s[17]",
                "target_height = s[18]",
                "target_cx = target_x + target_width / 2.0",
                "target_cy = target_y + target_height / 2.0",
                f"return robot_{axis} < target_c{'x' if axis=='x' else 'y'}"
            ]
        ))
        fid += 1
        features.append(feature_source(
            f"f{fid}",
            f"robot_{'right' if axis=='x' else 'above'}_of_target_center",
            [
                "robot_x = s[0]",
                "robot_y = s[1]",
                "target_x = s[9]",
                "target_y = s[10]",
                "target_width = s[17]",
                "target_height = s[18]",
                "target_cx = target_x + target_width / 2.0",
                "target_cy = target_y + target_height / 2.0",
                f"return robot_{axis} > target_c{'x' if axis=='x' else 'y'}"
            ]
        ))
        fid += 1

    # 2. Manhattan and Euclidean distance to target center (thresholded)
    for dist_type, name, expr in [
        ("manhattan", "manhattan", "abs(robot_x - target_cx) + abs(robot_y - target_cy)"),
        ("euclidean", "euclidean", "((robot_x - target_cx)**2 + (robot_y - target_cy)**2)**0.5"),
    ]:
        for thresh, tname in [(0.25, "very_close"), (0.5, "close"), (1.0, "medium"), (2.0, "far")]:
            features.append(feature_source(
                f"f{fid}",
                f"{dist_type}_distance_to_target_{tname}",
                [
                    "robot_x = s[0]",
                    "robot_y = s[1]",
                    "target_x = s[9]",
                    "target_y = s[10]",
                    "target_width = s[17]",
                    "target_height = s[18]",
                    "target_cx = target_x + target_width / 2.0",
                    "target_cy = target_y + target_height / 2.0",
                    f"dist = {expr}",
                    f"return dist < {thresh}"
                ]
            ))
            fid += 1

    # 3. Is robot inside target region
    features.append(feature_source(
        f"f{fid}",
        "robot_inside_target_region",
        [
            "robot_x = s[0]",
            "robot_y = s[1]",
            "target_x = s[9]",
            "target_y = s[10]",
            "target_width = s[17]",
            "target_height = s[18]",
            "return (robot_x >= target_x) and (robot_x <= target_x + target_width) and (robot_y >= target_y) and (robot_y <= target_y + target_height)"
        ]
    ))
    fid += 1

    # 4. Is robot about to enter target region (one step ahead)
    features.append(feature_source(
        f"f{fid}",
        "robot_next_step_inside_target_region",
        [
            "robot_x = s[0]",
            "robot_y = s[1]",
            "dx = a[0]",
            "dy = a[1]",
            "target_x = s[9]",
            "target_y = s[10]",
            "target_width = s[17]",
            "target_height = s[18]",
            "next_x = robot_x + dx",
            "next_y = robot_y + dy",
            "return (next_x >= target_x) and (next_x <= target_x + target_width) and (next_y >= target_y) and (next_y <= target_y + target_height)"
        ]
    ))
    fid += 1

    # 5. Is robot moving toward target center (dot product)
    features.append(feature_source(
        f"f{fid}",
        "robot_moving_toward_target_center",
        [
            "robot_x = s[0]",
            "robot_y = s[1]",
            "dx = a[0]",
            "dy = a[1]",
            "target_x = s[9]",
            "target_y = s[10]",
            "target_width = s[17]",
            "target_height = s[18]",
            "target_cx = target_x + target_width / 2.0",
            "target_cy = target_y + target_height / 2.0",
            "to_target_x = target_cx - robot_x",
            "to_target_y = target_cy - robot_y",
            "dot = dx * to_target_x + dy * to_target_y",
            "return dot > 0"
        ]
    ))
    fid += 1

    # 6. Is robot moving away from target center
    features.append(feature_source(
        f"f{fid}",
        "robot_moving_away_from_target_center",
        [
            "robot_x = s[0]",
            "robot_y = s[1]",
            "dx = a[0]",
            "dy = a[1]",
            "target_x = s[9]",
            "target_y = s[10]",
            "target_width = s[17]",
            "target_height = s[18]",
            "target_cx = target_x + target_width / 2.0",
            "target_cy = target_y + target_height / 2.0",
            "to_target_x = target_cx - robot_x",
            "to_target_y = target_cy - robot_y",
            "dot = dx * to_target_x + dy * to_target_y",
            "return dot < 0"
        ]
    ))
    fid += 1

    # 7. Is robot aligned with passage_0 gap (obstacle0)
    features.append(feature_source(
        f"f{fid}",
        "robot_y_aligned_with_passage0_gap",
        [
            "robot_y = s[1]",
            "robot_base_radius = s[3]",
            "obstacle0_y = s[20]",
            "obstacle0_height = s[28]",
            "obstacle1_y = s[30]",
            "obstacle1_height = s[38]",
            "gap_lower = obstacle0_y + obstacle0_height + robot_base_radius",
            "gap_upper = obstacle1_y + obstacle1_height - robot_base_radius",
            "return (robot_y >= gap_lower) and (robot_y <= gap_upper)"
        ]
    ))
    fid += 1

    # 8. Is robot about to be aligned with passage_0 gap (after action)
    features.append(feature_source(
        f"f{fid}",
        "robot_next_y_aligned_with_passage0_gap",
        [
            "robot_y = s[1]",
            "dy = a[1]",
            "robot_base_radius = s[3]",
            "obstacle0_y = s[20]",
            "obstacle0_height = s[28]",
            "obstacle1_y = s[30]",
            "obstacle1_height = s[38]",
            "gap_lower = obstacle0_y + obstacle0_height + robot_base_radius",
            "gap_upper = obstacle1_y + obstacle1_height - robot_base_radius",
            "next_y = robot_y + dy",
            "return (next_y >= gap_lower) and (next_y <= gap_upper)"
        ]
    ))
    fid += 1

    # 9. Is robot left of passage_0 wall (obstacle0)
    features.append(feature_source(
        f"f{fid}",
        "robot_left_of_passage0_wall",
        [
            "robot_x = s[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "return robot_x + robot_base_radius < obstacle0_x"
        ]
    ))
    fid += 1

    # 10. Is robot inside passage_0 zone (between walls)
    features.append(feature_source(
        f"f{fid}",
        "robot_inside_passage0_zone",
        [
            "robot_x = s[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "obstacle0_width = s[27]",
            "wall_left = obstacle0_x",
            "wall_right = obstacle0_x + obstacle0_width + robot_base_radius",
            "return (robot_x >= wall_left) and (robot_x < wall_right)"
        ]
    ))
    fid += 1

    # 11. Is robot past passage_0 wall (right of wall)
    features.append(feature_source(
        f"f{fid}",
        "robot_past_passage0_wall",
        [
            "robot_x = s[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "obstacle0_width = s[27]",
            "wall_right = obstacle0_x + obstacle0_width + robot_base_radius",
            "return robot_x >= wall_right"
        ]
    ))
    fid += 1

    # 12. Is robot about to cross passage_0 wall (after action)
    features.append(feature_source(
        f"f{fid}",
        "robot_next_past_passage0_wall",
        [
            "robot_x = s[0]",
            "dx = a[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "obstacle0_width = s[27]",
            "wall_right = obstacle0_x + obstacle0_width + robot_base_radius",
            "next_x = robot_x + dx",
            "return next_x >= wall_right"
        ]
    ))
    fid += 1

    # 13. Is robot moving horizontally (dx > |dy|)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_more_horizontal_than_vertical",
        [
            "dx = a[0]",
            "dy = a[1]",
            "return abs(dx) > abs(dy)"
        ]
    ))
    fid += 1

    # 14. Is robot moving vertically (|dy| > |dx|)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_more_vertical_than_horizontal",
        [
            "dx = a[0]",
            "dy = a[1]",
            "return abs(dy) > abs(dx)"
        ]
    ))
    fid += 1

    # 15. Is robot rotating significantly (|dtheta| > 0.1)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_significant_rotation",
        [
            "dtheta = a[2]",
            "return abs(dtheta) > 0.1"
        ]
    ))
    fid += 1

    # 16. Is robot not rotating (|dtheta| < 0.01)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_no_rotation",
        [
            "dtheta = a[2]",
            "return abs(dtheta) < 0.01"
        ]
    ))
    fid += 1

    # 17. Is robot moving forward in x (dx > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_dx_positive",
        [
            "dx = a[0]",
            "return dx > 0"
        ]
    ))
    fid += 1

    # 18. Is robot moving forward in y (dy > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_dy_positive",
        [
            "dy = a[1]",
            "return dy > 0"
        ]
    ))
    fid += 1

    # 19. Is robot moving backward in x (dx < 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_dx_negative",
        [
            "dx = a[0]",
            "return dx < 0"
        ]
    ))
    fid += 1

    # 20. Is robot moving backward in y (dy < 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_dy_negative",
        [
            "dy = a[1]",
            "return dy < 0"
        ]
    ))
    fid += 1

    # 21. Is robot moving toward passage_0 wall (if left of wall and dx > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_left_of_passage0_wall_and_dx_positive",
        [
            "robot_x = s[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "dx = a[0]",
            "return (robot_x + robot_base_radius < obstacle0_x) and (dx > 0)"
        ]
    ))
    fid += 1

    # 22. Is robot inside passage_0 zone and moving horizontally (|dx| > |dy|)
    features.append(feature_source(
        f"f{fid}",
        "robot_inside_passage0_zone_and_horizontal_action",
        [
            "robot_x = s[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "obstacle0_width = s[27]",
            "wall_left = obstacle0_x",
            "wall_right = obstacle0_x + obstacle0_width + robot_base_radius",
            "dx = a[0]",
            "dy = a[1]",
            "return (robot_x >= wall_left) and (robot_x < wall_right) and (abs(dx) > abs(dy))"
        ]
    ))
    fid += 1

    # 23. Is robot y aligned with passage_0 gap and moving horizontally
    features.append(feature_source(
        f"f{fid}",
        "robot_y_aligned_with_passage0_gap_and_horizontal_action",
        [
            "robot_y = s[1]",
            "robot_base_radius = s[3]",
            "obstacle0_y = s[20]",
            "obstacle0_height = s[28]",
            "obstacle1_y = s[30]",
            "obstacle1_height = s[38]",
            "gap_lower = obstacle0_y + obstacle0_height + robot_base_radius",
            "gap_upper = obstacle1_y + obstacle1_height - robot_base_radius",
            "dx = a[0]",
            "dy = a[1]",
            "return (robot_y >= gap_lower) and (robot_y <= gap_upper) and (abs(dx) > abs(dy))"
        ]
    ))
    fid += 1

    # 24. Is robot about to collide with passage_0 wall (next_x overlaps wall)
    features.append(feature_source(
        f"f{fid}",
        "robot_next_x_collides_with_passage0_wall",
        [
            "robot_x = s[0]",
            "dx = a[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "obstacle0_width = s[27]",
            "wall_left = obstacle0_x",
            "wall_right = obstacle0_x + obstacle0_width",
            "next_left = robot_x + dx - robot_base_radius",
            "next_right = robot_x + dx + robot_base_radius",
            "return (next_right > wall_left) and (next_left < wall_right)"
        ]
    ))
    fid += 1

    # 25. Is robot about to collide with obstacle1 (vertical wall)
    features.append(feature_source(
        f"f{fid}",
        "robot_next_x_collides_with_obstacle1",
        [
            "robot_x = s[0]",
            "dx = a[0]",
            "robot_base_radius = s[3]",
            "obstacle1_x = s[29]",
            "obstacle1_width = s[37]",
            "wall_left = obstacle1_x",
            "wall_right = obstacle1_x + obstacle1_width",
            "next_left = robot_x + dx - robot_base_radius",
            "next_right = robot_x + dx + robot_base_radius",
            "return (next_right > wall_left) and (next_left < wall_right)"
        ]
    ))
    fid += 1

    # 26. Is robot about to collide with obstacle0 (vertical wall)
    features.append(feature_source(
        f"f{fid}",
        "robot_next_x_collides_with_obstacle0",
        [
            "robot_x = s[0]",
            "dx = a[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "obstacle0_width = s[27]",
            "wall_left = obstacle0_x",
            "wall_right = obstacle0_x + obstacle0_width",
            "next_left = robot_x + dx - robot_base_radius",
            "next_right = robot_x + dx + robot_base_radius",
            "return (next_right > wall_left) and (next_left < wall_right)"
        ]
    ))
    fid += 1

    # 27. Is robot about to collide with obstacle0 (horizontal wall, y overlap)
    features.append(feature_source(
        f"f{fid}",
        "robot_next_y_collides_with_obstacle0",
        [
            "robot_y = s[1]",
            "dy = a[1]",
            "robot_base_radius = s[3]",
            "obstacle0_y = s[20]",
            "obstacle0_height = s[28]",
            "wall_bottom = obstacle0_y",
            "wall_top = obstacle0_y + obstacle0_height",
            "next_bottom = robot_y + dy - robot_base_radius",
            "next_top = robot_y + dy + robot_base_radius",
            "return (next_top > wall_bottom) and (next_bottom < wall_top)"
        ]
    ))
    fid += 1

    # 28. Is robot about to collide with obstacle1 (horizontal wall, y overlap)
    features.append(feature_source(
        f"f{fid}",
        "robot_next_y_collides_with_obstacle1",
        [
            "robot_y = s[1]",
            "dy = a[1]",
            "robot_base_radius = s[3]",
            "obstacle1_y = s[30]",
            "obstacle1_height = s[38]",
            "wall_bottom = obstacle1_y",
            "wall_top = obstacle1_y + obstacle1_height",
            "next_bottom = robot_y + dy - robot_base_radius",
            "next_top = robot_y + dy + robot_base_radius",
            "return (next_top > wall_bottom) and (next_bottom < wall_top)"
        ]
    ))
    fid += 1

    # 29. Is robot moving toward passage_0 gap (if not aligned, dy points toward gap center)
    features.append(feature_source(
        f"f{fid}",
        "robot_moving_toward_passage0_gap_center",
        [
            "robot_y = s[1]",
            "dy = a[1]",
            "robot_base_radius = s[3]",
            "obstacle0_y = s[20]",
            "obstacle0_height = s[28]",
            "obstacle1_y = s[30]",
            "obstacle1_height = s[38]",
            "gap_lower = obstacle0_y + obstacle0_height + robot_base_radius",
            "gap_upper = obstacle1_y + obstacle1_height - robot_base_radius",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "to_gap = gap_center - robot_y",
            "return (abs(to_gap) > 0.01) and (dy * to_gap > 0)"
        ]
    ))
    fid += 1

    # 30. Is robot moving away from passage_0 gap center
    features.append(feature_source(
        f"f{fid}",
        "robot_moving_away_from_passage0_gap_center",
        [
            "robot_y = s[1]",
            "dy = a[1]",
            "robot_base_radius = s[3]",
            "obstacle0_y = s[20]",
            "obstacle0_height = s[28]",
            "obstacle1_y = s[30]",
            "obstacle1_height = s[38]",
            "gap_lower = obstacle0_y + obstacle0_height + robot_base_radius",
            "gap_upper = obstacle1_y + obstacle1_height - robot_base_radius",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "to_gap = gap_center - robot_y",
            "return (abs(to_gap) > 0.01) and (dy * to_gap < 0)"
        ]
    ))
    fid += 1

    # 31. Is robot y within 0.1 of passage_0 gap center
    features.append(feature_source(
        f"f{fid}",
        "robot_y_near_passage0_gap_center",
        [
            "robot_y = s[1]",
            "robot_base_radius = s[3]",
            "obstacle0_y = s[20]",
            "obstacle0_height = s[28]",
            "obstacle1_y = s[30]",
            "obstacle1_height = s[38]",
            "gap_lower = obstacle0_y + obstacle0_height + robot_base_radius",
            "gap_upper = obstacle1_y + obstacle1_height - robot_base_radius",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "return abs(robot_y - gap_center) < 0.1"
        ]
    ))
    fid += 1

    # 32. Is robot y far from passage_0 gap center (> 0.2)
    features.append(feature_source(
        f"f{fid}",
        "robot_y_far_from_passage0_gap_center",
        [
            "robot_y = s[1]",
            "robot_base_radius = s[3]",
            "obstacle0_y = s[20]",
            "obstacle0_height = s[28]",
            "obstacle1_y = s[30]",
            "obstacle1_height = s[38]",
            "gap_lower = obstacle0_y + obstacle0_height + robot_base_radius",
            "gap_upper = obstacle1_y + obstacle1_height - robot_base_radius",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "return abs(robot_y - gap_center) > 0.2"
        ]
    ))
    fid += 1

    # 33. Is robot action small (|dx| < 0.01 and |dy| < 0.01)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_small_translation",
        [
            "dx = a[0]",
            "dy = a[1]",
            "return (abs(dx) < 0.01) and (abs(dy) < 0.01)"
        ]
    ))
    fid += 1

    # 34. Is robot action large (|dx| > 0.04 or |dy| > 0.04)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_large_translation",
        [
            "dx = a[0]",
            "dy = a[1]",
            "return (abs(dx) > 0.04) or (abs(dy) > 0.04)"
        ]
    ))
    fid += 1

    # 35. Is robot action diagonal (|dx| > 0.01 and |dy| > 0.01)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_diagonal",
        [
            "dx = a[0]",
            "dy = a[1]",
            "return (abs(dx) > 0.01) and (abs(dy) > 0.01)"
        ]
    ))
    fid += 1

    # 36. Is robot action purely horizontal (|dx| > 0.01 and |dy| < 0.01)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_purely_horizontal",
        [
            "dx = a[0]",
            "dy = a[1]",
            "return (abs(dx) > 0.01) and (abs(dy) < 0.01)"
        ]
    ))
    fid += 1

    # 37. Is robot action purely vertical (|dy| > 0.01 and |dx| < 0.01)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_purely_vertical",
        [
            "dx = a[0]",
            "dy = a[1]",
            "return (abs(dy) > 0.01) and (abs(dx) < 0.01)"
        ]
    ))
    fid += 1

    # 38. Is robot arm moving (|darm| > 0.01)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_arm_moving",
        [
            "darm = a[3]",
            "return abs(darm) > 0.01"
        ]
    ))
    fid += 1

    # 39. Is robot vacuum/gripper active (|vac| > 0.01)
    features.append(feature_source(
        f"f{fid}",
        "robot_action_vacuum_active",
        [
            "vac = a[4]",
            "return abs(vac) > 0.01"
        ]
    ))
    fid += 1

    # 40. Is robot moving toward obstacle0 (if left of wall and dx > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_left_of_obstacle0_and_dx_positive",
        [
            "robot_x = s[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "dx = a[0]",
            "return (robot_x + robot_base_radius < obstacle0_x) and (dx > 0)"
        ]
    ))
    fid += 1

    # 41. Is robot moving toward obstacle1 (if left of wall and dx > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_left_of_obstacle1_and_dx_positive",
        [
            "robot_x = s[0]",
            "robot_base_radius = s[3]",
            "obstacle1_x = s[29]",
            "dx = a[0]",
            "return (robot_x + robot_base_radius < obstacle1_x) and (dx > 0)"
        ]
    ))
    fid += 1

    # 42. Is robot moving away from obstacle0 (if right of wall and dx > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_right_of_obstacle0_and_dx_positive",
        [
            "robot_x = s[0]",
            "robot_base_radius = s[3]",
            "obstacle0_x = s[19]",
            "obstacle0_width = s[27]",
            "wall_right = obstacle0_x + obstacle0_width",
            "dx = a[0]",
            "return (robot_x - robot_base_radius > wall_right) and (dx > 0)"
        ]
    ))
    fid += 1

    # 43. Is robot moving away from obstacle1 (if right of wall and dx > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_right_of_obstacle1_and_dx_positive",
        [
            "robot_x = s[0]",
            "robot_base_radius = s[3]",
            "obstacle1_x = s[29]",
            "obstacle1_width = s[37]",
            "wall_right = obstacle1_x + obstacle1_width",
            "dx = a[0]",
            "return (robot_x - robot_base_radius > wall_right) and (dx > 0)"
        ]
    ))
    fid += 1

    # 44. Is robot moving toward target region in x (robot_x < target_x and dx > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_left_of_target_and_dx_positive",
        [
            "robot_x = s[0]",
            "target_x = s[9]",
            "dx = a[0]",
            "return (robot_x < target_x) and (dx > 0)"
        ]
    ))
    fid += 1

    # 45. Is robot moving toward target region in y (robot_y < target_y and dy > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_below_target_and_dy_positive",
        [
            "robot_y = s[1]",
            "target_y = s[10]",
            "dy = a[1]",
            "return (robot_y < target_y) and (dy > 0)"
        ]
    ))
    fid += 1

    # 46. Is robot moving away from target region in x (robot_x > target_x + target_width and dx > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_right_of_target_and_dx_positive",
        [
            "robot_x = s[0]",
            "target_x = s[9]",
            "target_width = s[17]",
            "dx = a[0]",
            "return (robot_x > target_x + target_width) and (dx > 0)"
        ]
    ))
    fid += 1

    # 47. Is robot moving away from target region in y (robot_y > target_y + target_height and dy > 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_above_target_and_dy_positive",
        [
            "robot_y = s[1]",
            "target_y = s[10]",
            "target_height = s[18]",
            "dy = a[1]",
            "return (robot_y > target_y + target_height) and (dy > 0)"
        ]
    ))
    fid += 1

    # 48. Is robot moving toward target region in x (robot_x > target_x + target_width and dx < 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_right_of_target_and_dx_negative",
        [
            "robot_x = s[0]",
            "target_x = s[9]",
            "target_width = s[17]",
            "dx = a[0]",
            "return (robot_x > target_x + target_width) and (dx < 0)"
        ]
    ))
    fid += 1

    # 49. Is robot moving toward target region in y (robot_y > target_y + target_height and dy < 0)
    features.append(feature_source(
        f"f{fid}",
        "robot_above_target_and_dy_negative",
        [
            "robot_y = s[1]",
            "target_y = s[10]",
            "target_height = s[18]",
            "dy = a[1]",
            "return (robot_y > target_y + target_height) and (dy < 0)"
        ]
    ))
    fid += 1

    # 50. Is robot moving toward passage_0 gap and not about to collide with wall
    features.append(feature_source(
        f"f{fid}",
        "robot_moving_toward_passage0_gap_and_not_colliding",
        [
            "robot_y = s[1]",
            "dy = a[1]",
            "robot_base_radius = s[3]",
            "obstacle0_y = s[20]",
            "obstacle0_height = s[28]",
            "obstacle1_y = s[30]",
            "obstacle1_height = s[38]",
            "gap_lower = obstacle0_y + obstacle0_height + robot_base_radius",
            "gap_upper = obstacle1_y + obstacle1_height - robot_base_radius",
            "gap_center = (gap_lower + gap_upper) / 2.0",
            "to_gap = gap_center - robot_y",
            "robot_x = s[0]",
            "dx = a[0]",
            "obstacle0_x = s[19]",
            "obstacle0_width = s[27]",
            "wall_left = obstacle0_x",
            "wall_right = obstacle0_x + obstacle0_width",
            "next_left = robot_x + dx - robot_base_radius",
            "next_right = robot_x + dx + robot_base_radius",
            "collides = (next_right > wall_left) and (next_left < wall_right)",
            "return (abs(to_gap) > 0.01) and (dy * to_gap > 0) and (not collides)"
        ]
    ))
    fid += 1

    return {"features": features}
import json
if __name__ == "__main__":
    feature_set = build_feature_library()
    #write to json file
    with open("feature_library.json", "w") as f:
        json.dump(feature_set, f, indent=4)
