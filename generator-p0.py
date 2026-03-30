import math

def build_feature_library():
    # Helper functions for feature code generation

    def feature_source_template(fname, args, body):
        # Returns a function definition as a string
        lines = [f"def {fname}({', '.join(args)}):"]
        for b in body:
            for subline in str(b).splitlines():
                lines.append("    " + subline)
        return "\n".join(lines)

    # Index mapping for readability
    idx = {
        "robot_x": 0,
        "robot_y": 1,
        "robot_theta": 2,
        "robot_base_radius": 3,
        "robot_arm_joint": 4,
        "robot_arm_length": 5,
        "robot_vacuum": 6,
        "robot_gripper_height": 7,
        "robot_gripper_width": 8,
        "target_x": 9,
        "target_y": 10,
        "target_theta": 11,
        "target_static": 12,
        "target_color_r": 13,
        "target_color_g": 14,
        "target_color_b": 15,
        "target_z_order": 16,
        "target_width": 17,
        "target_height": 18,
        # Action
        "dx": 0,
        "dy": 1,
        "dtheta": 2,
        "darm": 3,
        "vac": 4,
    }

    # Feature families

    features = []
    fid = 1

    # 1. Directional movement toward target center (normalized)
    for axis, sign, name in [
        ("x", "+", "move_toward_target_x_pos"),
        ("x", "-", "move_toward_target_x_neg"),
        ("y", "+", "move_toward_target_y_pos"),
        ("y", "-", "move_toward_target_y_neg"),
    ]:
        if axis == "x":
            robot = "s[%d]" % idx["robot_x"]
            target = "s[%d] + 0.5 * s[%d]" % (idx["target_x"], idx["target_width"])
            action = "a[%d]" % idx["dx"]
        else:
            robot = "s[%d]" % idx["robot_y"]
            target = "s[%d] + 0.5 * s[%d]" % (idx["target_y"], idx["target_height"])
            action = "a[%d]" % idx["dy"]
        if sign == "+":
            cond = f"({target} - {robot}) > 0 and {action} > 0"
        else:
            cond = f"({target} - {robot}) < 0 and {action} < 0"
        fname = f"f{fid}"
        features.append({
            "id": fname,
            "name": name,
            "source": feature_source_template(fname, ["s", "a"], [f"return {cond}"])
        })
        fid += 1

    # 2. Is robot left/right/above/below target center
    for rel, axis, op, name in [
        ("left_of", "x", "<", "robot_left_of_target"),
        ("right_of", "x", ">", "robot_right_of_target"),
        ("below", "y", "<", "robot_below_target"),
        ("above", "y", ">", "robot_above_target"),
    ]:
        if axis == "x":
            robot = "s[%d]" % idx["robot_x"]
            target = "s[%d] + 0.5 * s[%d]" % (idx["target_x"], idx["target_width"])
        else:
            robot = "s[%d]" % idx["robot_y"]
            target = "s[%d] + 0.5 * s[%d]" % (idx["target_y"], idx["target_height"])
        cond = f"{robot} {op} {target}"
        fname = f"f{fid}"
        features.append({
            "id": fname,
            "name": name,
            "source": feature_source_template(fname, ["s", "a"], [f"return {cond}"])
        })
        fid += 1

    # 3. Is action moving robot closer to target center (Euclidean)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "dx = a[0]",
        "dy = a[1]",
        "dist_before = (rx - tx)**2 + (ry - ty)**2",
        "dist_after = ((rx + dx) - tx)**2 + ((ry + dy) - ty)**2",
        "return dist_after < dist_before"
    ]
    features.append({
        "id": fname,
        "name": "action_moves_closer_to_target_center",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 4. Is action moving robot away from target center (Euclidean)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "dx = a[0]",
        "dy = a[1]",
        "dist_before = (rx - tx)**2 + (ry - ty)**2",
        "dist_after = ((rx + dx) - tx)**2 + ((ry + dy) - ty)**2",
        "return dist_after > dist_before"
    ]
    features.append({
        "id": fname,
        "name": "action_moves_away_from_target_center",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 5. Is action direction aligned with vector to target (cosine similarity > 0)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "dx = a[0]",
        "dy = a[1]",
        "vec_to_target_x = tx - rx",
        "vec_to_target_y = ty - ry",
        "dot = dx * vec_to_target_x + dy * vec_to_target_y",
        "return dot > 0"
    ]
    features.append({
        "id": fname,
        "name": "action_aligned_with_target_vector",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 6. Is action direction anti-aligned with vector to target (cosine similarity < 0)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "dx = a[0]",
        "dy = a[1]",
        "vec_to_target_x = tx - rx",
        "vec_to_target_y = ty - ry",
        "dot = dx * vec_to_target_x + dy * vec_to_target_y",
        "return dot < 0"
    ]
    features.append({
        "id": fname,
        "name": "action_anti_aligned_with_target_vector",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 7. Is robot inside target region (axis-aligned bounding box)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9]",
        "ty = s[10]",
        "tw = s[17]",
        "th = s[18]",
        "inside_x = (rx >= tx) and (rx <= tx + tw)",
        "inside_y = (ry >= ty) and (ry <= ty + th)",
        "return inside_x and inside_y"
    ]
    features.append({
        "id": fname,
        "name": "robot_inside_target_region",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 8. Will action put robot inside target region (after move)
    fname = f"f{fid}"
    body = [
        "rx = s[0] + a[0]",
        "ry = s[1] + a[1]",
        "tx = s[9]",
        "ty = s[10]",
        "tw = s[17]",
        "th = s[18]",
        "inside_x = (rx >= tx) and (rx <= tx + tw)",
        "inside_y = (ry >= ty) and (ry <= ty + th)",
        "return inside_x and inside_y"
    ]
    features.append({
        "id": fname,
        "name": "action_puts_robot_inside_target_region",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 9. Is robot within 1.5x base radius of target center (proximity)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "r = s[3]",
        "dist = ((rx - tx)**2 + (ry - ty)**2)**0.5",
        "return dist < 1.5 * r"
    ]
    features.append({
        "id": fname,
        "name": "robot_near_target_center",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 10. Is action a small move (norm of dx,dy < 0.1)
    fname = f"f{fid}"
    body = [
        "norm = (a[0]**2 + a[1]**2)**0.5",
        "return norm < 0.1"
    ]
    features.append({
        "id": fname,
        "name": "action_is_small_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 11. Is action a large move (norm of dx,dy > 0.2)
    fname = f"f{fid}"
    body = [
        "norm = (a[0]**2 + a[1]**2)**0.5",
        "return norm > 0.2"
    ]
    features.append({
        "id": fname,
        "name": "action_is_large_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 12. Is action mostly along x axis (|dx| > 2*|dy|)
    fname = f"f{fid}"
    body = [
        "return abs(a[0]) > 2 * abs(a[1])"
    ]
    features.append({
        "id": fname,
        "name": "action_mostly_x_axis",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 13. Is action mostly along y axis (|dy| > 2*|dx|)
    fname = f"f{fid}"
    body = [
        "return abs(a[1]) > 2 * abs(a[0])"
    ]
    features.append({
        "id": fname,
        "name": "action_mostly_y_axis",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 14. Is action nearly diagonal (|dx| ~= |dy|, both > 0.01)
    fname = f"f{fid}"
    body = [
        "return abs(abs(a[0]) - abs(a[1])) < 0.01 and abs(a[0]) > 0.01 and abs(a[1]) > 0.01"
    ]
    features.append({
        "id": fname,
        "name": "action_nearly_diagonal",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 15. Is robot facing toward target (angle between heading and target vector < 45 deg)
    fname = f"f{fid}"
    body = [
        "import math",
        "rx = s[0]",
        "ry = s[1]",
        "theta = s[2]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "vec_to_target_x = tx - rx",
        "vec_to_target_y = ty - ry",
        "heading_x = math.cos(theta)",
        "heading_y = math.sin(theta)",
        "dot = heading_x * vec_to_target_x + heading_y * vec_to_target_y",
        "norm1 = (heading_x**2 + heading_y**2)**0.5",
        "norm2 = (vec_to_target_x**2 + vec_to_target_y**2)**0.5",
        "if norm2 == 0: return True",
        "cos_angle = dot / (norm1 * norm2)",
        "return cos_angle > 0.7071"
    ]
    features.append({
        "id": fname,
        "name": "robot_facing_toward_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 16. Is robot facing away from target (angle > 135 deg)
    fname = f"f{fid}"
    body = [
        "import math",
        "rx = s[0]",
        "ry = s[1]",
        "theta = s[2]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "vec_to_target_x = tx - rx",
        "vec_to_target_y = ty - ry",
        "heading_x = math.cos(theta)",
        "heading_y = math.sin(theta)",
        "dot = heading_x * vec_to_target_x + heading_y * vec_to_target_y",
        "norm1 = (heading_x**2 + heading_y**2)**0.5",
        "norm2 = (vec_to_target_x**2 + vec_to_target_y**2)**0.5",
        "if norm2 == 0: return False",
        "cos_angle = dot / (norm1 * norm2)",
        "return cos_angle < -0.7071"
    ]
    features.append({
        "id": fname,
        "name": "robot_facing_away_from_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 17. Is action rotating robot toward target (sign of dtheta matches angle diff)
    fname = f"f{fid}"
    body = [
        "import math",
        "rx = s[0]",
        "ry = s[1]",
        "theta = s[2]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "vec_to_target_x = tx - rx",
        "vec_to_target_y = ty - ry",
        "target_angle = math.atan2(vec_to_target_y, vec_to_target_x)",
        "angle_diff = ((target_angle - theta + math.pi) % (2*math.pi)) - math.pi",
        "return (angle_diff > 0 and a[2] > 0) or (angle_diff < 0 and a[2] < 0)"
    ]
    features.append({
        "id": fname,
        "name": "action_rotates_toward_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 18. Is action rotating robot away from target
    fname = f"f{fid}"
    body = [
        "import math",
        "rx = s[0]",
        "ry = s[1]",
        "theta = s[2]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "vec_to_target_x = tx - rx",
        "vec_to_target_y = ty - ry",
        "target_angle = math.atan2(vec_to_target_y, vec_to_target_x)",
        "angle_diff = ((target_angle - theta + math.pi) % (2*math.pi)) - math.pi",
        "return (angle_diff > 0 and a[2] < 0) or (angle_diff < 0 and a[2] > 0)"
    ]
    features.append({
        "id": fname,
        "name": "action_rotates_away_from_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 19. Is robot already aligned (angle diff < 10 deg)
    fname = f"f{fid}"
    body = [
        "import math",
        "rx = s[0]",
        "ry = s[1]",
        "theta = s[2]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "vec_to_target_x = tx - rx",
        "vec_to_target_y = ty - ry",
        "target_angle = math.atan2(vec_to_target_y, vec_to_target_x)",
        "angle_diff = abs(((target_angle - theta + math.pi) % (2*math.pi)) - math.pi)",
        "return angle_diff < 0.1745"
    ]
    features.append({
        "id": fname,
        "name": "robot_aligned_with_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 20. Is action a pure translation (no rotation, no arm, no vac)
    fname = f"f{fid}"
    body = [
        "return a[2] == 0 and a[3] == 0 and a[4] == 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_pure_translation",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 21. Is action a pure rotation (dtheta != 0, dx=dy=0)
    fname = f"f{fid}"
    body = [
        "return a[2] != 0 and a[0] == 0 and a[1] == 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_pure_rotation",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 22. Is action a pure arm move (darm != 0, others zero)
    fname = f"f{fid}"
    body = [
        "return a[3] != 0 and a[0] == 0 and a[1] == 0 and a[2] == 0 and a[4] == 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_pure_arm_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 23. Is action a pure vacuum (vac != 0, others zero)
    fname = f"f{fid}"
    body = [
        "return a[4] != 0 and a[0] == 0 and a[1] == 0 and a[2] == 0 and a[3] == 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_pure_vacuum",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 24. Is action a translation and rotation (dx/dy and dtheta nonzero)
    fname = f"f{fid}"
    body = [
        "return (a[0] != 0 or a[1] != 0) and a[2] != 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_translation_and_rotation",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 25. Is action a translation and arm move (dx/dy and darm nonzero)
    fname = f"f{fid}"
    body = [
        "return (a[0] != 0 or a[1] != 0) and a[3] != 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_translation_and_arm_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 26. Is action a translation and vacuum (dx/dy and vac nonzero)
    fname = f"f{fid}"
    body = [
        "return (a[0] != 0 or a[1] != 0) and a[4] != 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_translation_and_vacuum",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 27. Is action a rotation and arm move (dtheta and darm nonzero)
    fname = f"f{fid}"
    body = [
        "return a[2] != 0 and a[3] != 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_rotation_and_arm_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 28. Is action a rotation and vacuum (dtheta and vac nonzero)
    fname = f"f{fid}"
    body = [
        "return a[2] != 0 and a[4] != 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_rotation_and_vacuum",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 29. Is action an arm move and vacuum (darm and vac nonzero)
    fname = f"f{fid}"
    body = [
        "return a[3] != 0 and a[4] != 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_arm_move_and_vacuum",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 30. Is robot at edge of target region (within 0.05 of any edge)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9]",
        "ty = s[10]",
        "tw = s[17]",
        "th = s[18]",
        "on_left = abs(rx - tx) < 0.05",
        "on_right = abs(rx - (tx + tw)) < 0.05",
        "on_bottom = abs(ry - ty) < 0.05",
        "on_top = abs(ry - (ty + th)) < 0.05",
        "return on_left or on_right or on_bottom or on_top"
    ]
    features.append({
        "id": fname,
        "name": "robot_at_edge_of_target_region",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 31. Is robot far from target (> 2 units)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "dist = ((rx - tx)**2 + (ry - ty)**2)**0.5",
        "return dist > 2.0"
    ]
    features.append({
        "id": fname,
        "name": "robot_far_from_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 32. Is robot close to target (< 0.5 units)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "dist = ((rx - tx)**2 + (ry - ty)**2)**0.5",
        "return dist < 0.5"
    ]
    features.append({
        "id": fname,
        "name": "robot_close_to_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 33. Is action moving robot toward target along x axis (sign of dx matches x diff)
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "tx = s[9] + 0.5 * s[17]",
        "dx = a[0]",
        "return (tx - rx > 0 and dx > 0) or (tx - rx < 0 and dx < 0)"
    ]
    features.append({
        "id": fname,
        "name": "action_moves_toward_target_x",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 34. Is action moving robot toward target along y axis (sign of dy matches y diff)
    fname = f"f{fid}"
    body = [
        "ry = s[1]",
        "ty = s[10] + 0.5 * s[18]",
        "dy = a[1]",
        "return (ty - ry > 0 and dy > 0) or (ty - ry < 0 and dy < 0)"
    ]
    features.append({
        "id": fname,
        "name": "action_moves_toward_target_y",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 35. Is action moving robot away from target along x axis
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "tx = s[9] + 0.5 * s[17]",
        "dx = a[0]",
        "return (tx - rx > 0 and dx < 0) or (tx - rx < 0 and dx > 0)"
    ]
    features.append({
        "id": fname,
        "name": "action_moves_away_from_target_x",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 36. Is action moving robot away from target along y axis
    fname = f"f{fid}"
    body = [
        "ry = s[1]",
        "ty = s[10] + 0.5 * s[18]",
        "dy = a[1]",
        "return (ty - ry > 0 and dy < 0) or (ty - ry < 0 and dy > 0)"
    ]
    features.append({
        "id": fname,
        "name": "action_moves_away_from_target_y",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 37. Is action a no-op (all zeros)
    fname = f"f{fid}"
    body = [
        "return a[0] == 0 and a[1] == 0 and a[2] == 0 and a[3] == 0 and a[4] == 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_noop",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 38. Is action a forward move in robot's heading (dot(dx,dy) with heading > 0)
    fname = f"f{fid}"
    body = [
        "import math",
        "theta = s[2]",
        "heading_x = math.cos(theta)",
        "heading_y = math.sin(theta)",
        "dot = a[0] * heading_x + a[1] * heading_y",
        "return dot > 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_forward_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 39. Is action a backward move in robot's heading (dot(dx,dy) with heading < 0)
    fname = f"f{fid}"
    body = [
        "import math",
        "theta = s[2]",
        "heading_x = math.cos(theta)",
        "heading_y = math.sin(theta)",
        "dot = a[0] * heading_x + a[1] * heading_y",
        "return dot < 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_backward_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 40. Is action a leftward move in robot's frame (perpendicular dot < 0)
    fname = f"f{fid}"
    body = [
        "import math",
        "theta = s[2]",
        "left_x = -math.sin(theta)",
        "left_y = math.cos(theta)",
        "dot = a[0] * left_x + a[1] * left_y",
        "return dot > 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_leftward_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 41. Is action a rightward move in robot's frame (perpendicular dot > 0)
    fname = f"f{fid}"
    body = [
        "import math",
        "theta = s[2]",
        "right_x = math.sin(theta)",
        "right_y = -math.cos(theta)",
        "dot = a[0] * right_x + a[1] * right_y",
        "return dot > 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_rightward_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 42. Is robot and target color similar (Euclidean RGB < 0.2)
    fname = f"f{fid}"
    body = [
        "dr = abs(s[13] - s[13])",
        "dg = abs(s[14] - s[14])",
        "db = abs(s[15] - s[15])",
        "dist = (dr**2 + dg**2 + db**2)**0.5",
        "return dist < 0.2"
    ]
    features.append({
        "id": fname,
        "name": "robot_target_color_similar",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 43. Is target static
    fname = f"f{fid}"
    body = [
        "return s[12] == 1"
    ]
    features.append({
        "id": fname,
        "name": "target_is_static",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 44. Is target dynamic
    fname = f"f{fid}"
    body = [
        "return s[12] == 0"
    ]
    features.append({
        "id": fname,
        "name": "target_is_dynamic",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 45. Is robot base radius larger than target width
    fname = f"f{fid}"
    body = [
        "return s[3] > s[17]"
    ]
    features.append({
        "id": fname,
        "name": "robot_base_larger_than_target_width",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 46. Is robot base radius smaller than target width
    fname = f"f{fid}"
    body = [
        "return s[3] < s[17]"
    ]
    features.append({
        "id": fname,
        "name": "robot_base_smaller_than_target_width",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 47. Is robot arm extended (joint > 0.5)
    fname = f"f{fid}"
    body = [
        "return s[4] > 0.5"
    ]
    features.append({
        "id": fname,
        "name": "robot_arm_extended",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 48. Is robot arm retracted (joint < 0.5)
    fname = f"f{fid}"
    body = [
        "return s[4] < 0.5"
    ]
    features.append({
        "id": fname,
        "name": "robot_arm_retracted",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 49. Is robot gripper open (width > 0.05)
    fname = f"f{fid}"
    body = [
        "return s[8] > 0.05"
    ]
    features.append({
        "id": fname,
        "name": "robot_gripper_open",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 50. Is robot gripper closed (width < 0.05)
    fname = f"f{fid}"
    body = [
        "return s[8] < 0.05"
    ]
    features.append({
        "id": fname,
        "name": "robot_gripper_closed",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 51. Is action a positive vacuum (vac > 0)
    fname = f"f{fid}"
    body = [
        "return a[4] > 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_positive_vacuum",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 52. Is action a negative vacuum (vac < 0)
    fname = f"f{fid}"
    body = [
        "return a[4] < 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_negative_vacuum",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 53. Is action a positive arm move (darm > 0)
    fname = f"f{fid}"
    body = [
        "return a[3] > 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_positive_arm_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 54. Is action a negative arm move (darm < 0)
    fname = f"f{fid}"
    body = [
        "return a[3] < 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_negative_arm_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 55. Is action a positive rotation (dtheta > 0)
    fname = f"f{fid}"
    body = [
        "return a[2] > 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_positive_rotation",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 56. Is action a negative rotation (dtheta < 0)
    fname = f"f{fid}"
    body = [
        "return a[2] < 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_negative_rotation",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 57. Is action a positive x move (dx > 0)
    fname = f"f{fid}"
    body = [
        "return a[0] > 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_positive_x_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 58. Is action a negative x move (dx < 0)
    fname = f"f{fid}"
    body = [
        "return a[0] < 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_negative_x_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 59. Is action a positive y move (dy > 0)
    fname = f"f{fid}"
    body = [
        "return a[1] > 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_positive_y_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 60. Is action a negative y move (dy < 0)
    fname = f"f{fid}"
    body = [
        "return a[1] < 0"
    ]
    features.append({
        "id": fname,
        "name": "action_is_negative_y_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 61. Is robot gripper height above target center
    fname = f"f{fid}"
    body = [
        "return s[7] > s[10] + 0.5 * s[18]"
    ]
    features.append({
        "id": fname,
        "name": "gripper_above_target_center",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 62. Is robot gripper height below target center
    fname = f"f{fid}"
    body = [
        "return s[7] < s[10] + 0.5 * s[18]"
    ]
    features.append({
        "id": fname,
        "name": "gripper_below_target_center",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 63. Is robot gripper width wider than target width
    fname = f"f{fid}"
    body = [
        "return s[8] > s[17]"
    ]
    features.append({
        "id": fname,
        "name": "gripper_wider_than_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 64. Is robot gripper width narrower than target width
    fname = f"f{fid}"
    body = [
        "return s[8] < s[17]"
    ]
    features.append({
        "id": fname,
        "name": "gripper_narrower_than_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 65. Is robot z-order above target
    fname = f"f{fid}"
    body = [
        "return s[16] > s[16]"
    ]
    features.append({
        "id": fname,
        "name": "robot_z_above_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 66. Is robot z-order below target
    fname = f"f{fid}"
    body = [
        "return s[16] < s[16]"
    ]
    features.append({
        "id": fname,
        "name": "robot_z_below_target",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 67. Is action a conjunction: moving toward target and small move
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "dx = a[0]",
        "dy = a[1]",
        "dist_before = (rx - tx)**2 + (ry - ty)**2",
        "dist_after = ((rx + dx) - tx)**2 + ((ry + dy) - ty)**2",
        "norm = (dx**2 + dy**2)**0.5",
        "return dist_after < dist_before and norm < 0.1"
    ]
    features.append({
        "id": fname,
        "name": "move_toward_target_and_small_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    # 68. Is action a conjunction: moving away from target and large move
    fname = f"f{fid}"
    body = [
        "rx = s[0]",
        "ry = s[1]",
        "tx = s[9] + 0.5 * s[17]",
        "ty = s[10] + 0.5 * s[18]",
        "dx = a[0]",
        "dy = a[1]",
        "dist_before = (rx - tx)**2 + (ry - ty)**2",
        "dist_after = ((rx + dx) - tx)**2 + ((ry + dy) - ty)**2",
        "norm = (dx**2 + dy**2)**0.5",
        "return dist_after > dist_before and norm > 0.2"
    ]
    features.append({
        "id": fname,
        "name": "move_away_from_target_and_large_move",
        "source": feature_source_template(fname, ["s", "a"], body)
    })
    fid += 1

    return {"features": features}

import json
if __name__ == "__main__":
    feature_set = build_feature_library()
    #write to json file
    with open("feature_library.json", "w") as f:
        json.dump(feature_set, f, indent=4)
