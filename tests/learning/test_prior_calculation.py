    f_complex = (
    "def f2(s, a):\n"
    "    r, c = a\n"
    "    h = len(s)\n"
    "    w = len(s[0]) if h else 0\n"
    "    if r < 0 or r >= h or c < 0 or c >= w:\n"
    "        return False\n"
    "    if s[r][c] != 'empty':\n"
    "        return False\n"
    "    # count walls in 4-neighborhood\n"
    "    cnt = 0\n"
    "    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:\n"
    "        rr, cc = r+dr, c+dc\n"
    "        if 0 <= rr < h and 0 <= cc < w and s[rr][cc] == 'wall':\n"
    "            cnt += 1\n"
    "    return cnt == 0\n"
    )
    features = {
    "f1": "def f1(s, a):\n    r,c=a\n    return s[r][c] == 'empty'\n",
    "f2": "def f2(s, a):\n    r,c=a\n    return s[r][c] == 'wall'\n",
    "f3": "def f3(s, a):\n    r,c=a\n    return s[r][c] == 'right_arrow'\n",
    "f4": f_complex,
    }
    scores = score_features_log_prior(features)
    probs = normalize_log_scores_to_probs(scores)
    print(scores, probs)
    log_probs = probs_to_logprobs(probs)
    
    
    if __name__=="__main__":
    f1 = "def f1(s, a):\n    r,c=a\n    return s[r][c] == 'empty'\n"

    features = [f1, f1, f1] 
    out = priors_from_features(features)
    log_scores = out["log_scores"]
    probs = out["probs"]
    logprobs = out["logprobs"]
    print(log_scores, probs, logprobs)