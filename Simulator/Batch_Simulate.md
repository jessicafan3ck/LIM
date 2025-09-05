# 0) Imports & what they’re used for

* `os`, `glob`, `json`: filesystem ops, file discovery, writing JSON.
* `math`, `defaultdict`: small math & dict utils (only lightly used here).
* `numpy as np`, `pandas as pd`: data wrangling & numerics.
* From **`lim_simulator_module`**:

  * `fit_match_model(df)` → builds the per-match `MatchFit` (field, ball, policy, success).
  * `construct_state_from_row(row, rosters)` → turns a seed event + rough rosters into `GameState`.
  * `simulate_rollouts(state0, fit, cfg)` → runs the Monte Carlo simulation and returns results dict.
  * `SimConfig` → simulation hyperparameters container.

---

# 1) Column coalescing & coercion

## `resolve_xy_columns(df) -> DataFrame`

**Why:** After joining multiple sources (events + synthetic), columns can appear as `name_x`/`name_y`. The simulator expects base names (e.g., `team_name`, `x_location_start`).
**What it does:**

* Makes a copy of `df`.
* For a fixed list of expected bases (team & player identity, geometry, time columns, plus synthetic fields), it:

  * Checks whether `base_x` or `base_y` exists.
  * If yes and the base doesn’t exist yet:

    * If *both* exist → set `base = base_x.where(base_x.notna(), base_y)` (prefer `_x` unless it’s NaN).
    * Else copy whichever exists.
* Returns the updated frame (it does **not** drop the `_x/_y` columns; it just guarantees the base exists).

**Assumptions:** Any post-merge duplicates follow the `_x/_y` naming convention; `_x` is generally “events” and preferred over `_y` (typically “synthetic”).

---

## `coerce_for_fit(df) -> DataFrame`

**Why:** `fit_match_model` requires strict column names and numeric types; lots of feeds have different names and NaNs. Also, casting a NaN to `int` throws, so we keep times/coords as floats and fill safely.

**What it does (in order):**

1. **Coalesce** using `resolve_xy_columns`.
2. **team\_name**:

   * If missing, tries alternates (`from_team_name`, `team`, …).
   * If still missing, creates an empty string column.
3. **from\_player\_id** / **from\_player\_name**:

   * Try common alternates.
   * Force `from_player_id` to numeric with `errors="coerce"` (so invalid → NaN).
   * Ensure `from_player_name` exists (empty string fallback).
4. **event\_type** / **event**:

   * Try alternates; else set empty strings.
5. **pressure**:

   * Try alternates; else create `False` boolean (no pressure).
6. **half\_time**:

   * Try alternates; else create `1`.
7. **Time columns**: cast to **float** (not int) `NaN` allowed:
   `["match_run_time_in_ms", "event_end_time_in_ms", "match_time_in_ms", "start_time_ms", "event_time_ms", "end_time_ms", "next_time_ms"]`
8. **Geometry columns**: cast to **float** `NaN` allowed (start/end x/y under various naming spellings).
9. **Synthetic fields**: ensure they exist & numeric; fill NaN with safe defaults:

   * `syn_f_i=1.0`, `v_i_eff=6.0`, `a_i_eff=3.0`, `d_i_eff=4.0`, `tau_i_react_eff=0.25`, `tau_i_turn_eff=0.34`
10. **Time columns fill**: for the main time columns, fill remaining NaN with `0.0` (floats).
    Rationale: if something later casts to `int`, `0.0` is safe.
11. **half\_time**: numeric with fill `1`.
12. **event\_id**, **match\_id**: cast to string; replace `"nan"` strings with `""` so filenames/keys don’t become the literal `"nan"`.

**Output:** A frame with all required columns present, numeric types friendly for later steps, and no accidental int-casts on NaN.

---

# 2) Seed selection utilities

## `is_true_flag(val) -> bool`

* Normalizes truthy variants: `"1"`, `"true"`, `"t"`, `"yes"`, `"y"` → `True`; everything else `False`.
* Used to read `pressure`-like boolean-ish fields that might be strings/numbers.

## `final_third_mask(df) -> Series[bool]`

* Reads `x_location_start` and scales to meters if it looks normalized (max ≤ 1.2 → multiply by 105).
* Returns a boolean mask for **final third** defined as `x ≥ 70m` (left→right assumption for MVP).
* Used to prioritize “dangerous” seed events.

## `pick_seed_indices(merged, max_seeds=5) -> list[int]`

**Why:** We don’t want to simulate from every row at first. Pick a handful of “interesting” events per match.

**Steps:**

* Build `valid = merged.dropna(subset=["x_location_start","y_location_start","event_type"])`.
* `et = valid["event_type"].str.lower()` for comparisons.
* Create candidate lists:

  1. `shots`: indices where `event_type == "shot"`.
  2. `pressured`: indices where `pressure` is truthy (via `is_true_flag`).
  3. `final3`: indices in final third.
* Fill `chosen` in priority tiers:

  1. Add all `shots` until reaching `max_seeds`.
  2. Add events that are both `pressured` and in `final3`.
  3. Add any remaining `pressured`.
  4. If still short, add a random sample from the remaining valid rows (deterministic shuffle with seed 7).
* Returns up to `max_seeds` indices.

**Assumptions:** Left→right attack; in practice you might want to infer direction per half.

---

# 3) Roster imputation (events-only approximation)

## `_scale_if_normalized(vals, axis_len) -> Series`

* If abs(max) ≤ 1.2, treat as normalized and multiply by `axis_len` (105 for x, 68 for y). Otherwise pass through.

## `rough_rosters_at_row(row, merged) -> dict[str, list[(pid, name, x, y)]]`

**Goal:** Build **\~11v11** approximate positions around the seed event, *using events only* (no tracking). This gives the simulator enough players to compute control/hazard fields.

**Detailed flow:**

1. **Make a time column `__time_ms`**:

   * For each row, pick the first available & numeric among `match_run_time_in_ms`, `event_end_time_in_ms`, `match_time_in_ms`, turned into `int`. If none, leave as `NaN`.
2. **Pick seed time**: same logic on the seed `row`. If still `None`, fallback to `0`.
3. **Ensure coords in meters**: scale `x_location_*` by 105, `y_location_*` by 68 if they look normalized.
4. **Window rows near seed**: ±180,000 ms (±3 minutes) around `seed_t` → `near`.
5. **Determine teams**:

   * `team_me` = `row["team_name"]` (stringified).
   * `opp_guess`: first different `team_name` found in `near`. If none, synthesize a label like `Opponent_<match_id>`.
6. **Collect closest positions per player per team**:

   * For a team name, filter `near` by that team.
   * Restrict columns to: `from_player_id`, `from_player_name`, `x_location_start`, `y_location_start`, `__time_ms`.
   * Drop rows missing start x/y or player name.
   * Compute `abs_dt = |__time_ms − seed_t|`.
   * Group by `from_player_name` (string key), take the row with smallest `abs_dt`.
   * Extract `(pid, name) → (x,y)` into a dict for that team.
7. **Pad to \~11 players** if fewer found:

   * Compute centroid of known players (or center field if none).
   * Sample Gaussian offsets (`dx ~ N(0,7m)`, `dy ~ N(0,6m)`), clamp to pitch.
   * Insert synthetic players `(None, f"{team_label}_SYN{k}") → (x,y)` until 11.
8. **Produce rosters**:

   * Dict: `{ team_me: [(pid,name,x,y), ...], opp_guess: [(pid,name,x,y), ...] }`.

**Notes & limitations:**

* Works only off actors who recently initiated events; defenders/off-ball players are synthetic.
* Player identities for synthetic are `pid=None` with synthetic names; that’s fine for control/hazard.

---

# 4) Batch runner

## `run_all(...)`

**Purpose:** Iterate over all merged CSVs, fit a per-match model, pick seed events, simulate from each seed, save per-seed JSON & a per-match summary CSV.

**Parameters (defaults shown):**

* `base_dir`: root folder (e.g., `"/Users/jessicafan/LIM"`).
* `merged_subdir`: where merged CSVs live (e.g., `"Merged"`).
* `out_subdir`: where to write outputs (e.g., `"Simulations"`).
* `max_seeds`: number of seeds per match (5).
* `horizon_K`: steps per rollout (5).
* `rollouts_R`: Monte Carlo rollouts per seed (200).
* `half_life_events`: confidence half-life in events (3.0).
* `seed`: RNG seed (7).

**I/O paths:**

* Reads: `glob(f"{base_dir}/{merged_subdir}/*_merged.csv")`.
* Writes per match: JSONs into `f"{base_dir}/{out_subdir}/{match_base}/event_{event_id}_sim.json"`.
* Writes per match: summary CSV into `f"{base_dir}/{out_subdir}/{match_base}/{match_base}_summary.csv"`.

**Detailed flow:**

1. Build `merged_dir`, `out_root`; `os.makedirs(out_root, exist_ok=True)`.
2. Collect `merged_paths` (sorted). If none: print info & return.
3. For each CSV path `mp`:

   * Derive `base` by stripping `_merged` and extension (becomes folder & filenames).
   * Print banner `=== Processing match: {base} ===`.
   * **Read & normalize dataframe**:

     * `pd.read_csv(mp)`.
     * `coerce_for_fit(merged)` → ensures required columns exist, numeric types are safe, times filled with 0.0, IDs as strings.
     * On read/coercion error: warn & `continue`.
   * **Fit per-match model**:

     * `fit = fit_match_model(merged)`.
     * On error (e.g., missing required columns): warn & `continue`.
   * **Pick seeds**:

     * `seed_indices = pick_seed_indices(merged, max_seeds)`.
     * If empty: print info & `continue`.
   * **Per-match output dir**:

     * `match_out_dir = f"{out_root}/{base}"`, `os.makedirs(..., exist_ok=True)`.
   * **Prepare summary**: `summary_rows = []`.
   * **Build SimConfig** with the input hyperparameters.
   * **Loop seeds**:

     * `row = merged.iloc[idx]` for each chosen index.
     * `rosters = rough_rosters_at_row(row, merged)` (≈11v11 per side, events-only).
     * `state0 = construct_state_from_row(row, rosters)` → `GameState` seed.
     * `result = simulate_rollouts(state0, fit, cfg)`:

       * Runs R rollouts × K steps, returns:

         * `"mean_sequence_confidence"`
         * `"top_patterns"`: `[[["pass","carry",...], count], ...]`
         * `"shot_prob_by_k"`: `{k: prob}` per step
         * `"results"`: up to 50 detailed sequences (actor, team, action, target, p\_succ, timing, confidences…)
     * **Write per-seed JSON**:

       * `eid = row.get("event_id", f"row{idx}")` (string).
       * `out_json = f"{match_out_dir}/event_{eid}_sim.json"`.
       * `json.dump(result, f, indent=2)`.
       * Print `[OK] {base} :: wrote {out_json}`.
     * **Accumulate summary**:

       * Pull `shot_prob_by_k` and `top_patterns`.
       * Append one row with:

         * `"match": base`
         * `"event_id": eid`
         * `"seed_index": idx`
         * `"mean_sequence_confidence"`
         * `"top_pattern"`: the most common pattern joined by `|` (or `None`)
         * `"top_pattern_count"`: its count
         * `"shot_prob_k1..k5"`: stepwise shooting probabilities (or `None` if not present)
   * **Write per-match summary CSV** if any rows collected:

     * DataFrame → `{base}_summary.csv` (no index).
     * Print `[OK] {base} :: wrote {sum_csv}`.

**Error handling & logs:**

* Read failures: `[WARN] Skip <path>: <err>`.
* Fit failures: `[WARN] Could not fit model for <base>: <err>`.
* No seeds: `[INFO] No seeds chosen for <base>`.
* Successful JSON/CSV writes print `[OK]`.

**Assumptions & guards:**

* **Required columns** for the simulator are enforced inside `fit_match_model` (it throws with the exact missing column name).
* **NaN times** don’t break int casts later because `coerce_for_fit` fills them with `0.0`.
* `half_time`, `event_id`, `match_id` are cleaned to numeric/strings to avoid type errors and ugly file names.
* Roster imputation pads to 11 so `control_field` has sufficient players to compute fields.
* Coordinate normalization is handled both in `coerce_for_fit` (numeric conversion) and inside the simulator (it will scale if values look normalized).

---

# 5) What the outputs contain (quick refresher)

For each seed JSON (`event_<event_id>_sim.json`):

* `mean_sequence_confidence`: average roll-level confidence after horizon decay.
* `top_patterns`: the most common K-action sequences across rollouts (just the **action types**, not exact targets).
* `shot_prob_by_k`: fraction of rollouts where step `k`’s action was `"shot"`.
* `results`: up to 50 rollouts, each:

  * `seq`: list of K steps; each step has `actor_id`, `actor_name`, `team`, `action`, `success`, `p_succ`, `(tx,ty)`, `dist_m`, `t_ms`, `ctrl_adv`, `hazard_here`, `c_step`.
  * `C_seq`: that rollout’s overall confidence.

Per-match summary CSV (`{match}_summary.csv`):

* One row per seed summarizing the key metrics, so you can sort/filter quickly.

---

# 6) Common pitfalls + how this file defends against them

* **`cannot convert float NaN to integer`**:
  `coerce_for_fit` converts times/IDs to **floats** and fills with `0.0`, preventing bad casts later in the simulator.
* **Missing `team_name`**:
  It synthesizes an empty column if it can’t find alternates; later steps (like control) expect two teams; the roster imputation will try to infer an opponent name from nearby events and otherwise synthesize one.
* **Normalized coordinates** vs meters:
  The runner only ensures numeric types; the **simulator** checks ranges and scales to 105×68 if needed.
* **Seed selection finds nothing**:
  You’ll get an info message, the match is skipped (still safe).
* **Output IDs become `"nan"` strings**:
  The coercion step replaces `"nan"` with `""` so filenames don’t look broken.

---

# 7) Places safely tweakable

* **`max_seeds`**: how many moments per match to simulate.
* **`horizon_K` / `rollouts_R`**: quality vs compute; start small, increase if you need smoother distributions.
* **`half_life_events`**: if you want longer/shorter confidence horizons.
* **`pick_seed_indices`** logic: add priorities (e.g., turnovers, line-break flags).
* **`rough_rosters_at_row`**: change ±time window, dummy spread, or how opponents are guessed.
* **Paths**: `base_dir`, `merged_subdir`, `out_subdir` to wherever your files live.

---

# 8) Minimal mental model

1. **Read CSV** → **normalize columns**
2. **Fit** policy/success/ball for that match
3. **Pick seeds** (shots/pressure/final third)
4. For each seed:

   * **Impute rosters** (\~11v11 per side, events-only)
   * **Construct game state** at that moment
   * **Simulate rollouts** (K steps, R paths)
   * **Write JSON** (full details)
5. **Write summary CSV** per match
