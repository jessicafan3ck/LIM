import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
from lim_simulator_module import (
    fit_match_model, construct_state_from_row,
    simulate_rollouts, SimConfig
)
from lim_batch_simulate import coerce_for_fit, rough_rosters_at_row

BASE_DIR = "/Users/jessicafan/LIM"
MERGED_DIR = os.path.join(BASE_DIR, "Merged")
SIM_DIR = os.path.join(BASE_DIR, "Simulations")  # written by your batch runner

app = Flask(__name__)
CORS(app)

def _load_merged_df(match_base: str) -> pd.DataFrame:
    path = os.path.join(MERGED_DIR, f"{match_base}_merged.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"merged not found: {path}")
    df = pd.read_csv(path)
    df = coerce_for_fit(df)
    return df

def _load_saved_sim(match_base: str, event_id: str):
    path = os.path.join(SIM_DIR, match_base, f"event_{event_id}_sim.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def _build_state_and_fit(match_base: str, event_id: str):
    df = _load_merged_df(match_base)
    # pick the event row; event_id is str in your batch
    row = df[df["event_id"].astype(str) == str(event_id)]
    if row.empty:
        # fallback to iloc if user passed an integer row index
        try:
            idx = int(event_id)
            row = df.iloc[[idx]]
        except Exception:
            raise ValueError(f"event_id {event_id} not found")
    row = row.iloc[0]
    rosters = rough_rosters_at_row(row, df)
    state0 = construct_state_from_row(row, rosters)
    fit = fit_match_model(df)
    return state0, fit

def _to_frontend_steps(sim_result: dict):
    """
    Convert LIM result -> [{k, t_ms, actor:{}, action:{}, target:{}, meta:{}}]
    """
    # choose a representative rollout (first), plus global score
    rep = sim_result.get("results", [{}])[0]
    seq = rep.get("seq", [])
    steps = []
    for e in seq:
        steps.append({
            "k": e["k"],
            "t_ms": e["t_ms"],
            "actor": {"id": e["actor_id"], "name": e["actor_name"], "team": e["team"]},
            "action": {"type": e["action"], "success": e["success"], "p_succ": e["p_succ"]},
            "target": {"x": e["tx"], "y": e["ty"]},
            "meta": {"ctrl_adv": e["ctrl_adv"], "hazard": e["hazard_here"], "c_step": e["c_step"]},
        })
    return {
        "steps": steps,
        "totalScore": sim_result.get("mean_sequence_confidence", 0.0),
        "shotProbByK": sim_result.get("shot_prob_by_k", {})
    }

@app.get("/sim/<match_base>/<event_id>")
def get_baseline(match_base, event_id):
    """
    Return saved baseline sim if present; else build on-the-fly once (optional).
    """
    saved = _load_saved_sim(match_base, event_id)
    if saved is None:
        # Optional: run one quick sim on the fly so UI can still show something
        state0, fit = _build_state_and_fit(match_base, event_id)
        sim = simulate_rollouts(state0, fit, SimConfig(horizon_K=5, rollouts_R=200, half_life_events=3.0, seed=7))
    else:
        sim = saved
    return jsonify(_to_frontend_steps(sim))

@app.post("/sim/<match_base>/<event_id>")
def post_counterfactual(match_base, event_id):
    """
    Body JSON:
    {
      "force_action": {"2":"shot"},
      "force_target": {"2":[100,34]},
      "temp_override": {"1":0.5},
      "horizon_K": 5,
      "rollouts_R": 200
    }
    """
    body = request.get_json(force=True, silent=True) or {}
    force_action = body.get("force_action") or {}
    force_target = body.get("force_target") or {}
    temp_override = body.get("temp_override") or {}
    horizon_K = int(body.get("horizon_K", 5))
    rollouts_R = int(body.get("rollouts_R", 200))
    half_life_events = float(body.get("half_life_events", 3.0))
    seed = body.get("seed", 7)

    # coerce keys to int
    force_action = {int(k): str(v) for k, v in force_action.items()}
    force_target = {int(k): (float(v[0]), float(v[1])) for k, v in force_target.items()}
    temp_override = {int(k): float(v) for k, v in temp_override.items()}

    state0, fit = _build_state_and_fit(match_base, event_id)
    sim = simulate_rollouts(
        state0, fit,
        SimConfig(horizon_K=horizon_K, rollouts_R=rollouts_R, half_life_events=half_life_events, seed=seed),
        force_action=force_action,
        force_target=force_target,
        temp_override=temp_override
    )
    return jsonify(_to_frontend_steps(sim))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
