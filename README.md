# Running the code

For Chinese instructions, see [README.zh-CN.md](README.zh-CN.md).

## 1. Install

Use Python 3.12 or newer on macOS/Linux, with a C compiler available as `cc`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Prepare an input CSV

Create `inputs.csv` with one treatment per row and the following required columns:

```csv
study_id,treatment_id,w0_g,days,temperature_c,ge_kj_g,protein_pct,lipid_pct,feeding
```

| Column | Value |
|---|---|
| `study_id` | Source/cohort identifier |
| `treatment_id` | Treatment identifier within the source |
| `w0_g` | Initial wet mass, g/fish |
| `days` | Duration, positive integer days |
| `temperature_c` | Constant temperature, °C |
| `ge_kj_g` | Feed gross energy, kJ/g |
| `protein_pct`, `lipid_pct` | Dietary composition, % |
| `feeding` | One of the feeding modes below |

| Feeding mode | Additional inputs |
|---|---|
| `satiation` | None |
| `mass_updated` | `offer_fraction`: daily offered feed/body mass, as a fraction; `update_days` defaults to 14 |
| `design_schedule` | `offer_fraction`, `design_fce`: prescribed gain/feed coefficient; `update_days` defaults to 14 |
| `relative_control` | `relative_ration`: fraction of simulated control capacity |
| `block_relative_control` | `relative_ration`; `update_days` defaults to 14 |

For relative feeding, include exactly one control row with `relative_ration=1` in the same `study_id`.

Optional dietary/environmental columns include `met_pct` (default 0.8), `do_mg_l` (default 6.5), `plant_replacement` and `insect_replacement` (fractions, default 0).

## 3. Run a simulation

```bash
python scripts/simulate.py --inputs inputs.csv --output outputs/predictions.csv
```

The default parameter file is `models/yellow_catfish.json`. To select another file:

```bash
python scripts/simulate.py --model models/yellow_catfish.json --inputs inputs.csv --output outputs/predictions.csv
```

To use stored shifts for source identifiers present in the parameter file:

```bash
python scripts/simulate.py --inputs inputs.csv --mode learned_context --output outputs/predictions.csv
```

The output CSV includes final mass (`predicted_weight_g`) and cumulative intake (`predicted_feed_g`), both per fish.

## 4. Calibrate with a completed control

In the completed control row, add `wf_g` (final wet mass, g/fish), or `observed_growth_ratio` (final/initial mass). Optionally add one of:

- `observed_feed_g`: cumulative intake, g/fish;
- `feeding_rate_pct_bw_day`: daily intake, % mean body mass/day;
- `fcr_g_g`: cumulative intake/biomass gain, g/g.

If several are supplied, intake takes precedence over feeding rate, then FCR. Leave these outcome columns blank for the other treatments.

```bash
python scripts/calibrate_source.py --inputs inputs.csv --control-id CONTROL_ID --output outputs/source_shifts.json
python scripts/simulate.py --inputs inputs.csv --calibration outputs/source_shifts.json --output outputs/calibrated_predictions.csv
```

The calibration file applies to the control's `study_id`; provide inputs from that source when running the calibrated simulation.

## 5. Run the v0.46 experiments

Install the additional experiment dependencies:

```bash
python -m pip install -r requirements-experiments.txt
```

Save the curated data archive as `yellow_catfish_data_package.zip`. Import its three input tables, saved fits and predictions, and figure source data:

```bash
python scripts/prepare_experiment_data.py --archive yellow_catfish_data_package.zip
```

Use the imported results to regenerate the study summaries, 20,000 study bootstrap samples, development predictions, and Figures 2–4 and S1–S2:

```bash
python scripts/review_targets_v046.py stats
python scripts/finalize_review_records_v046.py
python scripts/plot_review_v046.py
```

Results are written to `artifacts/reviewer-revision-v0.46/`; plots and trajectory CSVs are written to `outputs/figures_v0.46/`.

To fit all experiments from the input tables, use a fresh checkout and import with `--data-only`, then run these commands in order:

```bash
python scripts/prepare_experiment_data.py --archive yellow_catfish_data_package.zip --data-only
python scripts/review_experiments_v046.py nested
python scripts/review_targets_v046.py baselines
python scripts/review_targets_v046.py diagnostics
python scripts/review_tabular_baselines_v046.py
python scripts/review_tabular_baselines_v046.py --tune
python scripts/review_targets_v046.py stats
python scripts/finalize_review_records_v046.py
python scripts/plot_review_v046.py
```

The nested command runs 193 global fits across four configurations using four worker processes and reuses any saved fits. The baseline commands run SGR, TGC, ridge regression, XGBoost, and TabICLv2, followed by selection of ridge and XGBoost settings in the inner study blocks. TabICLv2 downloads its pretrained regression checkpoint on first use and runs on CPU.


## 6. Run the v0.47 global sensitivity experiment

After importing the literature data as above, run:

```bash
python scripts/global_sensitivity_v047.py
python scripts/summarize_global_sensitivity_v047.py
```

Outputs are written to `artifacts/global-sensitivity-v0.47/`. The experiment uses the frozen model, two independent Sobol scrambles and two additional coefficient ranges. Raw response arrays are saved as NPZ files; the second command produces CSV summaries.
