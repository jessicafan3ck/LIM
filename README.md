# LIM MVP (Living Influence Model) - README
Please view 'LIM_Theory_Paper.pdf' and 'LIM_Simulation_Analytics.pdf' for insights on the theory and framework behind this proposed model.

A minimal, events-only implementation of the Living Influence Model (LIM) suitable for a coach-facing demo using FIFA Unified Event Data and a per‑player Physical Summary for the U17 Women's World Cup. Any quantities not present in the dataset are synthetically instantiated via stable, interpretable defaults so the pipeline is fully runnable end‑to‑end without tracking data. The logic and generation process for all synthetic data will be provided within this repository. The model in this repository is ran on 31 matches from the U17 Women's World Cup. 

## LIM Data
### Base Data For FIFA U17 Women World Cup

Starting data from the U17 Women's World Cup are sorted by 2 data types: events from matches and physical summaries of players in each match. These are separated in folders 'Events' and 'Phys_Summary' in this repository. This data is the private property of FIFA and may only be redistributed by their sole delegation. Do not redistribute this data nor use it for any purpose other than that which has been permitted. 
Summaries for the data contents and interpretations for each data type will be contained in a txt file in each data folder. Look for 'Events_DataCard.md' and 'Phys_Summary_DataCard.md'.

### Synthetic Data

Synthetic data was generated for each of these variables:

The generated data as well as the generating pipeline for each of these variables is contained in the 'Synthetic' folder. Read 'Synthetic_DataCard.md' for a more granular description of each variable type. Read 'Synthetic_Pipeline.md' or read the comments in the generating code file ('lim_generate_synthetic.py') to understand the generation process.

The base and synthetic data were merged for the simulator pipeline.

## LIM Simulator Pipeline
### 1) `lim_simulator_module.py` — per-match model + simulator

#### What it is

A self-contained library that:

* fits a **per-match model** from events (+ synthetic sidecar)
  (ball flight defaults, action policy by field zone/phase, and simple success probabilities),
* builds a **GameState** around a seed event (with lightweight roster imputation supplied by caller),
* runs **Monte Carlo rollouts** of the next *K* actions and returns rich JSON diagnostics.

#### Inputs (columns expected after your “Merged” join)

Minimum required columns (names must exist; types are coerced upstream):

* Identity & time: `match_id`, `event_id`, `half_time`,
  `match_run_time_in_ms` or `event_end_time_in_ms` (either can be 0.0 if unknown)
* Actors/teams: `team_name`, `from_player_id`, `from_player_name`
* Geometry: `x_location_start`, `y_location_start`, `x_location_end`, `y_location_end`
* Semantics: `event_type`, `event`, `pressure`
* Synthetic physiology snapshot (per row):
  `syn_f_i`, `v_i_eff`, `a_i_eff`, `d_i_eff`, `tau_i_react_eff`, `tau_i_turn_eff`

Optional columns (auto-used if present): `line_break_direction`, `line_break_outcome`, `body_type`, `style`, `style_additional`, `event_end_time_in_ms`, `match_run_time_in_ms`.

> Note: If coordinates look normalized (≤ \~1.2), the module auto-scales to a 105×68m pitch.

#### What it fits

* **Ball model**: ground-ball decay `kappa_ground` (falls back to 1.3 s⁻¹ if timing sparse).
* **Policy model**: action distribution π(action | grid cell, phase) from in-match frequencies.
* **Success models**: simple logistic shapes per action (pass/carry/takeon/shot) using distance, pressure, and aerial flags.

#### Simulation

* Given a seed `GameState`, samples K steps per rollout:

  * chooses action from the zone policy,
  * samples simple target geometry (e.g., pass radius/angle),
  * scores success probability via the success model,
  * updates ball/possession and a crude pressure/control proxy,
  * advances clock by travel + control delay.
* Repeats R rollouts; aggregates:

  * top action-type patterns,
  * step-wise shot probabilities,
  * a decayed sequence confidence metric.

#### Key outputs (Python dict you typically JSON-dump)

* `mean_sequence_confidence` (0–1),
* `top_patterns` (e.g., `[["pass","carry","shot"], count]`),
* `shot_prob_by_k` (per step),
* `results` (up to 50 rollouts with step-level details).

#### Assumptions/limits

* No tracking; player positions are approximations provided by the caller.
* Lightweight control/hazard fields; good enough for MVP intuition.
* Models are **match-local** (no external training).

---

### 2) `lim_batch_simulate.py` — batch runner over merged files

#### What it is

A CLI-style script that:

1. reads every `*_merged.csv` in your `Merged/` folder,
2. normalizes/repairs columns,
3. fits the per-match model (via the module above),
4. picks a handful of **seed events**,
5. imputes **\~11v11** rosters from nearby events (with synthetic padding),
6. runs simulations,
7. writes **per-seed JSON** and a **per-match summary CSV**.

#### Directory expectations

* **Input**: `base_dir/Merged/*_merged.csv`
  (Your own merge step should have joined Events + Synthetic rows into these files.)
* **Output**: `base_dir/Simulations/<match_base>/`

  * `event_<event_id>_sim.json` (rich rollout diagnostics)
  * `<match_base>_summary.csv` (one-row-per-seed summary)

> We’re keeping Events and Synthetic on disk **separate**; only the merged CSVs are consumed here.

#### Data hygiene & guards (important!)

* **Column coalescing**: if merges left `_x/_y` duplicates, we coalesce into the base name (prefers `_x`).
* **Flexible synonyms**: `team_name`, `event_type`, `pressure`, etc., are backfilled from common alternates.
* **Types**: times/coords coerced to **floats**; missing times filled with `0.0` to avoid “NaN to int” errors downstream.
* **Synthetic defaults**: if a synthetic column is missing/NaN, we inject safe MVP defaults.
* **IDs**: `event_id`/`match_id` cast to strings; the literal `"nan"` is replaced with `""` to avoid goofy filenames.

#### Seed selection (defaults)

Prefers:

1. shots, then
2. pressured events in the final third, then
3. pressured events anywhere,
   else random valid rows; up to `max_seeds` per match.

#### Roster imputation (events-only)

* Looks ±3 minutes around the seed; for each team, takes each actor’s closest event position.
* Pads up to 11 with synthetic players around the team centroid (Gaussian scatter, clamped to pitch).
* Returns `{team_name: [(pid, name, x, y), ...], opponent: [...]}` to the simulator.

#### How to run (example)

```bash
python lim_batch_simulate.py
```

Default args:

* `base_dir="/Users/jessicafan/LIM"`
* `merged_subdir="Merged"`
* `out_subdir="Simulations"`
* `max_seeds=5`, `horizon_K=5`, `rollouts_R=200`, `half_life_events=3.0`, `seed=7`

Tweak them in the `run_all(...)` call at the bottom or wrap this in your own CLI.

#### Outputs to expect

* Per seed: `event_<event_id>_sim.json` with:

  * stepwise actions, success draws, targets, timings, local control/hazard, and per-step confidence,
  * aggregate `C_seq` per rollout.
* Per match: `<match>_summary.csv` with:

  * `event_id`, `seed_index`, `mean_sequence_confidence`,
  * top pattern (as `action1|action2|...`) and its count,
  * `shot_prob_k1..k5`.

#### Common issues & fixes (already handled)

* **“Missing required column: team\_name”** → `coerce_for_fit` fills from synonyms or sets blank.
* **“cannot convert float NaN to integer”** → times are floats with `0.0` fallback before any int cast.
* **Grid shape mismatch (36×24 vs 24×36)** → fixed by consistent `indexing="ij"` in grid construction.
* **Normalized coords** → auto-scaled to meters inside the module.

#### Knobs to tune

* Scale `rollouts_R`/`horizon_K` for quality vs speed.
* Adjust seed selection rules to match demo goals (e.g., include turnovers/line-breaks).
* Change roster padding spread or window (`±180s`) for different density.

