import os
import glob
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd

# Import the module you already have
from lim_simulator_module import (
    fit_match_model,
    construct_state_from_row,
    simulate_rollouts,
    SimConfig,
)

# ----------------------------
# Helpers: picking seeds
# ----------------------------
import pandas as pd

def resolve_xy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a column appears as both <name>_x and <name>_y (post-merge),
    create a single <name> that prefers _x when not null, else _y.
    If only one of them exists, copy it to <name>.
    """
    df = df.copy()
    def coalesce(name: str):
        cx, cy = f"{name}_x", f"{name}_y"
        if cx in df.columns or cy in df.columns:
            if name not in df.columns:
                if cx in df.columns and cy in df.columns:
                    df[name] = df[cx].where(df[cx].notna(), df[cy])
                elif cx in df.columns:
                    df[name] = df[cx]
                else:
                    df[name] = df[cy]

    # Coalesce the columns the simulator expects
    for base in [
        "team_name",
        "from_player_id", "from_player_name",
        "x_location_start", "y_location_start",
        "x_location_end",   "y_location_end",
        "event_type", "event",
        "pressure",
        "half_time",
        "match_run_time_in_ms", "event_end_time_in_ms",
        # if your synthetic sidecar also got _x/_y, include these too:
        "syn_f_i", "v_i_eff", "a_i_eff", "d_i_eff",
        "tau_i_react_eff", "tau_i_turn_eff",
    ]:
        coalesce(base)

    return df
def coerce_for_fit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coalesce *_x/_y, ensure required columns exist, and keep time/geometry as float
    to avoid any int-cast on NaN.
    """
    df = resolve_xy_columns(df).copy()

    cols = set(df.columns)

    # team_name
    if "team_name" not in cols:
        for alt in ["from_team_name", "team", "Team", "Team Name", "teamname"]:
            if alt in cols:
                df["team_name"] = df[alt]
                break
    if "team_name" not in df.columns:
        df["team_name"] = ""

    # from_player_id / name
    if "from_player_id" not in df.columns:
        for alt in ["player_id", "actor_id", "source_player_id"]:
            if alt in cols:
                df["from_player_id"] = df[alt]
                break
    df["from_player_id"] = pd.to_numeric(df.get("from_player_id"), errors="coerce")

    if "from_player_name" not in df.columns:
        for alt in ["player_name", "actor_name", "source_player_name"]:
            if alt in cols:
                df["from_player_name"] = df[alt]
                break
    if "from_player_name" not in df.columns:
        df["from_player_name"] = ""

    # event_type / event
    if "event_type" not in df.columns:
        for alt in ["type", "action_type", "Event Type"]:
            if alt in cols:
                df["event_type"] = df[alt]
                break
    if "event_type" not in df.columns:
        df["event_type"] = ""

    if "event" not in df.columns:
        for alt in ["Event", "action", "description"]:
            if alt in cols:
                df["event"] = df[alt]
                break
    if "event" not in df.columns:
        df["event"] = ""

    # pressure / half_time
    if "pressure" not in df.columns:
        for alt in ["under_pressure", "is_pressure", "pressure_flag"]:
            if alt in cols:
                df["pressure"] = df[alt]
                break
    if "pressure" not in df.columns:
        df["pressure"] = False

    if "half_time" not in df.columns:
        for alt in ["period", "half", "Half"]:
            if alt in cols:
                df["half_time"] = df[alt]
                break
    if "half_time" not in df.columns:
        df["half_time"] = 1

    # time columns -> numeric float (allow NaN)
    for c in ["match_run_time_in_ms", "event_end_time_in_ms",
              "match_time_in_ms", "start_time_ms", "event_time_ms", "end_time_ms", "next_time_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # geometry -> numeric float (allow NaN)
    for c in ["x_location_start","y_location_start","x_location_end","y_location_end",
              "x_start","y_start","x_end","y_end","start_x","start_y","end_x","end_y",
              "location_x_start","location_y_start","location_x_end","location_y_end"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # synthetic with safe defaults
    synth_defaults = {
        "syn_f_i": 1.0, "v_i_eff": 6.0, "a_i_eff": 3.0, "d_i_eff": 4.0,
        "tau_i_react_eff": 0.25, "tau_i_turn_eff": 0.34,
    }
    for k, v in synth_defaults.items():
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").fillna(v)
        else:
            df[k] = v
    # times -> numeric float, then fill NaN with 0 (so even if module casts to int, it's safe)
    time_cols = [
        "match_run_time_in_ms", "event_end_time_in_ms",
        "match_time_in_ms", "start_time_ms", "event_time_ms",
        "end_time_ms", "next_time_ms"
    ]
    for c in time_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            
    if "half_time" in df.columns:
        df["half_time"] = pd.to_numeric(df["half_time"], errors="coerce").fillna(1)

    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(str)
        # Optional: replace "nan" strings with a safe placeholder
        df.loc[df["event_id"].str.lower() == "nan", "event_id"] = ""

    if "match_id" in df.columns:
        df["match_id"] = df["match_id"].astype(str)
        df.loc[df["match_id"].str.lower() == "nan", "match_id"] = ""
    return df

def is_true_flag(val) -> bool:
    s = str(val).strip().lower()
    return s in {"1","true","t","yes","y"}

def final_third_mask(df: pd.DataFrame) -> pd.Series:
    # Assume attacking direction left→right for simplicity here.
    # We’ll call “final third” as x >= 70m in 105×68 coordinates.
    xs = df["x_location_start"].astype(float)
    # If normalized [0,1], scale
    if xs.abs().max() <= 1.2:
        xs = xs * 105.0
    return xs >= 70.0

def pick_seed_indices(merged: pd.DataFrame, max_seeds=5) -> list[int]:
    """
    Choose up to max_seeds interesting events:
    priority 1: shots
    priority 2: pressured events in final third
    priority 3: general pressured events
    fallback: random sample of valid rows
    """
    valid = merged.dropna(subset=["x_location_start","y_location_start","event_type"]).copy()
    et = valid["event_type"].astype(str).str.lower()

    shots = valid[et == "shot"].index.tolist()
    pressured = valid[valid["pressure"].apply(is_true_flag)].index.tolist()
    final3 = valid[final_third_mask(valid)].index.tolist()

    # scored lists with priority tiers
    chosen = []
    # 1) shots
    for idx in shots:
        chosen.append(idx)
        if len(chosen) >= max_seeds: return chosen
    # 2) pressured + final third
    tier2 = list(set(pressured).intersection(set(final3)))
    for idx in tier2:
        if idx not in chosen:
            chosen.append(idx)
            if len(chosen) >= max_seeds: return chosen
    # 3) pressured anywhere
    for idx in pressured:
        if idx not in chosen:
            chosen.append(idx)
            if len(chosen) >= max_seeds: return chosen
    # 4) fallback random from valid
    if len(chosen) < max_seeds:
        pool = [i for i in valid.index.tolist() if i not in chosen]
        if pool:
            rng = np.random.default_rng(7)
            rng.shuffle(pool)
            chosen.extend(pool[: max_seeds - len(chosen)])
    return chosen[:max_seeds]

# ----------------------------
# Helpers: roster imputation
# ----------------------------

def _scale_if_normalized(vals: pd.Series, axis_len: float) -> pd.Series:
    m = pd.to_numeric(vals, errors="coerce")
    if m.abs().max() <= 1.2:
        return m * axis_len
    return m

def rough_rosters_at_row(row: pd.Series, merged: pd.DataFrame) -> dict:
    """
    Build ~11v11 positions around the seed moment using only events.
    Strategy:
    - Take players who acted recently (from_player) per team, grab their closest x/y to seed time.
    - If fewer than 11, pad with dummy players around team centroid lines.
    """
    # Pitch in meters
    LEN, WID = 105.0, 68.0

    # Make a time column to sort by proximity
    def _choose_time_ms(r: pd.Series):
        for c in ("match_run_time_in_ms","event_end_time_in_ms","match_time_in_ms"):
            if c in r and pd.notna(r[c]):
                v = pd.to_numeric(r[c], errors="coerce")
                if pd.notna(v):
                    return int(v)
        return None

    merged = merged.copy()
    merged["__time_ms"] = merged.apply(_choose_time_ms, axis=1)

    seed_t = _choose_time_ms(row)
    if seed_t is None:
        # fallback 0
        seed_t = 0

    # ensure coords in meters
    for c in ["x_location_start","x_location_end"]:
        merged[c] = _scale_if_normalized(merged[c], LEN)
    for c in ["y_location_start","y_location_end"]:
        merged[c] = _scale_if_normalized(merged[c], WID)

    # Get recent actors (from_player) within ±3 minutes around seed_t
    WINDOW_MS = 180_000
    near = merged[(merged["__time_ms"].notna()) &
                  (merged["__time_ms"].between(seed_t - WINDOW_MS, seed_t + WINDOW_MS))]

    team_me = str(row.get("team_name",""))
    # Guess an opponent label if not encoded
    opp_guess = None
    # Grab a different team name if present nearby:
    teams_present = near["team_name"].dropna().astype(str).unique().tolist()
    for t in teams_present:
        if t != team_me:
            opp_guess = t
            break
    if opp_guess is None:
        opp_guess = f"Opponent_{row.get('match_id','X')}"

    # Build dicts: for each team, map (player_id, name) → closest (x,y) before seed
    def _closest_positions(team_name: str):
        sub = near[near["team_name"].astype(str) == team_name]
        # Focus on from_player since we know they acted there
        cols = ["from_player_id","from_player_name","x_location_start","y_location_start","__time_ms"]
        sub = sub[cols].dropna(subset=["x_location_start","y_location_start","from_player_name"])
        if sub.empty:
            return {}

        # pick the row with minimum |time - seed_t| per player
        sub["abs_dt"] = (sub["__time_ms"] - seed_t).abs()
        # Use player key
        sub["pid"] = sub["from_player_id"]
        sub = sub.sort_values(["from_player_name","abs_dt"])
        pos = {}
        for pname, grp in sub.groupby("from_player_name", sort=False):
            r0 = grp.iloc[0]
            pid = int(r0["pid"]) if pd.notna(r0["pid"]) else None
            pos[(pid, str(pname))] = (float(r0["x_location_start"]), float(r0["y_location_start"]))
        return pos

    pos_me = _closest_positions(team_me)
    pos_opp = _closest_positions(opp_guess)

    # Ensure we have ~11 per side by padding dummies near centroids
    def _pad_team(pos_dict: dict, team_label: str):
        if len(pos_dict) >= 11:
            return pos_dict
        # centroid & spread
        if pos_dict:
            xs = [xy[0] for xy in pos_dict.values()]
            ys = [xy[1] for xy in pos_dict.values()]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
        else:
            cx, cy = 52.5, 34.0
        # fill with synthetic players
        need = 11 - len(pos_dict)
        rng = np.random.default_rng(11)
        for k in range(need):
            dx = rng.normal(0, 7.0)
            dy = rng.normal(0, 6.0)
            x = float(max(0.0, min(LEN, cx + dx)))
            y = float(max(0.0, min(WID, cy + dy)))
            key = (None, f"{team_label}_SYN{k+1}")
            pos_dict[key] = (x, y)
        return pos_dict

    pos_me = _pad_team(pos_me, team_me or "TeamA")
    pos_opp = _pad_team(pos_opp, opp_guess or "TeamB")

    # Build roster lists: (player_id, player_name, x, y)
    team_rosters = {
        team_me: [(pid, pname, x, y) for (pid, pname), (x, y) in pos_me.items()],
        opp_guess: [(pid, pname, x, y) for (pid, pname), (x, y) in pos_opp.items()],
    }
    return team_rosters

# ----------------------------
# Runner
# ----------------------------

def run_all(
    base_dir="/Users/jessicafan/LIM",
    merged_subdir="Merged",
    out_subdir="Simulations",
    max_seeds=5,
    horizon_K=5,
    rollouts_R=200,
    half_life_events=3.0,
    seed=7
):
    merged_dir = os.path.join(base_dir, merged_subdir)
    out_root = os.path.join(base_dir, out_subdir)
    os.makedirs(out_root, exist_ok=True)

    merged_paths = sorted(glob.glob(os.path.join(merged_dir, "*_merged.csv")))
    if not merged_paths:
        print(f"[INFO] No merged files found in {merged_dir}")
        return

    for mp in merged_paths:
        base = os.path.splitext(os.path.basename(mp))[0].replace("_merged","")
        print(f"\n=== Processing match: {base} ===")
        try:
            merged = pd.read_csv(mp)
            merged = coerce_for_fit(merged)   # <- prevents any int-cast-on-NaN problems
        except Exception as e:
            print(f"[WARN] Skip {mp}: {e}")
            continue

        # Fit per-match model
        try:
            fit = fit_match_model(merged)
        except Exception as e:
            print(f"[WARN] Could not fit model for {base}: {e}")
            continue

        # Pick seeds
        seed_indices = pick_seed_indices(merged, max_seeds=max_seeds)
        if not seed_indices:
            print(f"[INFO] No seeds chosen for {base}")
            continue

        # Per-match output dir
        match_out_dir = os.path.join(out_root, base)
        os.makedirs(match_out_dir, exist_ok=True)

        # Summary rows to collect
        summary_rows = []

        # Sim config
        cfg = SimConfig(
            horizon_K=horizon_K,
            rollouts_R=rollouts_R,
            half_life_events=half_life_events,
            seed=seed
        )

        for idx in seed_indices:
            row = merged.iloc[idx]
            # Build rosters from events near this row
            rosters = rough_rosters_at_row(row, merged)

            # Build GameState
            state0 = construct_state_from_row(row, rosters)

            # Simulate
            result = simulate_rollouts(state0, fit, cfg)

            # Write per-seed JSON
            eid = row.get("event_id", f"row{idx}")
            out_json = os.path.join(match_out_dir, f"event_{eid}_sim.json")
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[OK] {base} :: wrote {out_json}")

            # Add to summary
            shot_prob = result.get("shot_prob_by_k", {})
            top_patterns = result.get("top_patterns", [])
            summary_rows.append({
                "match": base,
                "event_id": eid,
                "seed_index": idx,
                "mean_sequence_confidence": result.get("mean_sequence_confidence", None),
                "top_pattern": "|".join(top_patterns[0][0]) if top_patterns else None,
                "top_pattern_count": top_patterns[0][1] if top_patterns else None,
                "shot_prob_k1": shot_prob.get(1, None),
                "shot_prob_k2": shot_prob.get(2, None),
                "shot_prob_k3": shot_prob.get(3, None),
                "shot_prob_k4": shot_prob.get(4, None),
                "shot_prob_k5": shot_prob.get(5, None),
            })

        # Write per-match CSV summary
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            sum_csv = os.path.join(match_out_dir, f"{base}_summary.csv")
            df_sum.to_csv(sum_csv, index=False)
            print(f"[OK] {base} :: wrote {sum_csv}")

if __name__ == "__main__":
    run_all(
        base_dir="/Users/jessicafan/LIM",
        merged_subdir="Merged",
        out_subdir="Simulations",
        max_seeds=5,
        horizon_K=5,
        rollouts_R=200,
        half_life_events=3.0,
        seed=7
    )
