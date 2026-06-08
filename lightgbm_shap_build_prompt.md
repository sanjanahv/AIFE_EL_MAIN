# Build Prompt: LightGBM + Classical SHAP Analysis Dashboard
## For Antigravity / Cursor — Full Implementation Specification

---

## Project Summary

Build a localhost web application that:
1. Trains a LightGBM model on the UCI Adult Income dataset
2. Generates realistic applicant profiles (real data + optional synthetic augmentation)
3. Runs Classical SHAP analysis on those profiles
4. Presents an interactive analysis dashboard on localhost

This is **Mode 1** of a larger XAI pipeline. Only Classical SHAP is implemented here. The architecture must be modular so future SHAP variants (Causal, Layerwise, Conditional, Unified) can be added as new modes without changing the core structure.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask |
| Model | LightGBM (`lightgbm>=4.0`) |
| SHAP | `shap>=0.44` — TreeExplainer, interventional |
| Data | `pandas`, `scikit-learn`, `numpy` |
| Synthetic augmentation | `ctgan` (optional, flag-controlled) |
| Visualisation | `matplotlib`, `seaborn` (server-side plots → base64 PNG) |
| Frontend | Single HTML file with vanilla JS + CSS (no framework needed) |
| Config | `pyyaml` |

---

## Folder Structure

```
project/
├── config/
│   ├── hyperparameters.yaml
│   └── feature_metadata.yaml
├── data/
│   ├── raw/                        # auto-downloaded UCI Adult
│   └── processed/                  # saved parquets after preprocessing
├── models/
│   ├── train_lgbm.py
│   ├── model.pkl
│   └── model_info.json
├── explainers/
│   ├── base_explainer.py           # shared utilities
│   └── method2_classical_shap.py   # Mode 1 — Classical SHAP
├── profile_generator/
│   └── generator.py                # real + synthetic profile generation
├── analysis/
│   └── visualise.py                # plot generation → base64
├── app.py                          # Flask server
├── templates/
│   └── index.html                  # full dashboard UI
├── static/
│   └── style.css                   # optional external CSS
├── requirements.txt
└── README.md
```

---

## Step 1 — Config Files

### `config/hyperparameters.yaml`

```yaml
random_seed: 42
test_size: 0.2
stratify: true

lightgbm:
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 6
  num_leaves: 31
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  class_weight: balanced
  early_stopping_rounds: 50

profile_generator:
  n_real_profiles: 200          # sample from real test set
  use_synthetic: false          # set true to augment with CTGAN
  n_synthetic_profiles: 100     # only used if use_synthetic: true
  synthetic_noise_factor: 0.05  # small gaussian noise on synthetic

shap:
  background_samples: 100       # background dataset size for SHAP
  max_display_features: 15      # top N features shown in plots
```

### `config/feature_metadata.yaml`

```yaml
dataset: adult_income

target_column: income
positive_class: ">50K"

sensitive_features:
  - sex
  - race

mutable_features:
  - education_num
  - hours_per_week
  - workclass
  - occupation
  - capital_gain
  - capital_loss
  - marital_status

immutable_features:
  - age
  - sex
  - race
  - native_country
  - relationship

categorical_features:
  - workclass
  - education
  - marital_status
  - occupation
  - relationship
  - race
  - sex
  - native_country

continuous_features:
  - age
  - fnlwgt
  - education_num
  - capital_gain
  - capital_loss
  - hours_per_week
```

---

## Step 2 — Data Preprocessing

### `data/preprocess.py`

**What it must do:**

1. Load UCI Adult Income using scikit-learn:
```python
from sklearn.datasets import fetch_openml
data = fetch_openml('adult', version=2, as_frame=True)
```

2. Clean the data:
   - Drop rows with missing values (marked as '?' in UCI Adult)
   - Strip whitespace from categorical string values
   - Encode categoricals with `OrdinalEncoder` — fit on train, transform both
   - Scale continuous features with `StandardScaler` — fit on train, transform both
   - Encode target: '>50K' → 1, '<=50K' → 0

3. Split:
   - 80/20 stratified train/test split using `random_seed` from config
   - Save `X_train.parquet`, `X_test.parquet`, `y_train.parquet`, `y_test.parquet`
   - Save `encoders.pkl` (dict of fitted OrdinalEncoder + StandardScaler) for inverse transform later
   - Save `metadata.json`:

```json
{
  "dataset": "adult_income",
  "n_train": ...,
  "n_test": ...,
  "n_features": ...,
  "feature_names": [...],
  "feature_names_original": [...],
  "mutable_indices": [...],
  "immutable_indices": [...],
  "sensitive_indices": [...],
  "class_balance": {"0": ..., "1": ...},
  "categorical_features": [...],
  "continuous_features": [...]
}
```

**CRITICAL:** The number of features going into LightGBM must exactly match the number of features used in profile generation and SHAP. Store `n_features` and `feature_names` in metadata.json and use these everywhere — never hardcode feature counts.

---

## Step 3 — LightGBM Training

### `models/train_lgbm.py`

**What it must do:**

1. Load processed parquets
2. Load hyperparameters from yaml
3. Train `lightgbm.LGBMClassifier` with:
   - 10% of training data held out as validation set for early stopping
   - `callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]`
4. Evaluate on test set — compute and print:
   - AUC (ROC)
   - Accuracy
   - F1 score
   - Confusion matrix
5. Save:
   - `models/model.pkl` — pickle of fitted LGBMClassifier
   - `models/model.txt` — LightGBM text dump via `booster_.save_model()`
   - `models/model_info.json`:

```json
{
  "model_type": "LGBMClassifier",
  "n_estimators_trained": ...,
  "auc_test": ...,
  "accuracy_test": ...,
  "f1_test": ...,
  "feature_names": [...],
  "n_features": ...,
  "feature_importances_gain": {...},
  "training_time_seconds": ...,
  "random_seed": 42
}
```

6. Assert AUC > 0.80 — raise error if not met
7. Print a brief training summary table to console

**CRITICAL:** The `feature_names` list saved in `model_info.json` must be identical to the one in `metadata.json`. These are the ground truth feature names for the entire pipeline.

---

## Step 4 — Profile Generator

### `profile_generator/generator.py`

This module generates profiles that will be passed to the LightGBM model and SHAP. It has two modes controlled by `use_synthetic` in config.

**Class: `ProfileGenerator`**

```python
class ProfileGenerator:
    def __init__(self, config_path, metadata_path, data_path):
        # loads config, metadata, encoders
        # knows n_features and feature_names from metadata
        pass

    def generate(self, n_profiles=None, use_synthetic=None) -> pd.DataFrame:
        # returns DataFrame with exactly n_features columns
        # column names match feature_names from metadata exactly
        pass

    def _sample_real(self, n) -> pd.DataFrame:
        # sample n rows from X_test
        pass

    def _generate_synthetic(self, n, reference_df) -> pd.DataFrame:
        # Mode A (use_synthetic=false): perturb real profiles with gaussian noise
        #   - continuous features: add noise scaled by feature std * noise_factor
        #   - categorical features: occasionally swap to adjacent category
        #   - clip to valid ranges from training data
        # Mode B (use_synthetic=true): use CTGAN if installed
        #   - train CTGAN on X_train
        #   - generate n samples
        #   - clip to valid ranges
        # In both cases: validate output shape matches n_features exactly
        pass

    def inverse_transform(self, df) -> pd.DataFrame:
        # convert encoded values back to human-readable
        # for display in UI only — analysis uses encoded values
        pass
```

**CRITICAL dimension safety check:** After generation, always assert:
```python
assert profiles.shape[1] == self.n_features, \
    f"Profile shape mismatch: got {profiles.shape[1]}, expected {self.n_features}"
assert list(profiles.columns) == self.feature_names, \
    "Feature name mismatch between profiles and model"
```

If this assertion fails, raise a clear error before any model query.

---

## Step 5 — Classical SHAP Explainer

### `explainers/method2_classical_shap.py`

**Class: `ClassicalSHAPExplainer`**

```python
class ClassicalSHAPExplainer:
    def __init__(self, model, background_data, feature_names, config):
        # background_data: sample of X_train (n=100 by default)
        # feature_perturbation: "interventional" (marginal sampling)
        self.explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional"
        )
        self.feature_names = feature_names

    def explain(self, profiles: pd.DataFrame) -> dict:
        # profiles must have same columns as training data
        shap_values = self.explainer.shap_values(profiles)

        # For binary classification LightGBM, shap_values is a list
        # Use shap_values[1] (positive class) or handle both
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        return {
            "shap_values": sv,                          # shape: (n_profiles, n_features)
            "base_value": self.explainer.expected_value, # scalar or list
            "feature_names": self.feature_names,
            "mean_abs_shap": np.abs(sv).mean(axis=0),  # shape: (n_features,)
            "profiles": profiles.values,
            "method": "classical_shap"
        }

    def get_summary(self) -> dict:
        # Returns ranked feature importance summary
        # Used for the attribution summary table in UI
        pass
```

**Validation check:** After computing SHAP values, verify:
```python
# SHAP values + base_value should approximately equal model prediction
predictions = model.predict_proba(profiles)[:, 1]
shap_sum = sv.sum(axis=1) + base_value
max_error = np.abs(predictions - shap_sum).max()
assert max_error < 0.01, f"SHAP additivity check failed: max error {max_error}"
```

---

## Step 6 — Analysis Visualisation

### `analysis/visualise.py`

Generate all plots as base64-encoded PNG strings for embedding in HTML.

**Global matplotlib style:**
```python
import matplotlib as mpl
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F9FAFB",
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

METHOD_COLORS = {
    1: "#6B7280",   # plain model
    2: "#3B82F6",   # classical SHAP
    3: "#10B981",   # causal
    4: "#F59E0B",   # layerwise
    5: "#8B5CF6",   # conditional
    6: "#EF4444",   # unified
}
```

**Functions to implement:**

```python
def plot_shap_summary(shap_result: dict, max_features=15) -> str:
    # Horizontal bar chart — mean |SHAP| per feature, sorted descending
    # Bars coloured by METHOD_COLORS[2]
    # X-axis: "Mean |SHAP Value|"
    # Y-axis: feature names
    # Title: "Classical SHAP — Feature Attribution Summary"
    # Returns: base64 PNG string

def plot_shap_beeswarm(shap_result: dict, max_features=15) -> str:
    # Standard SHAP beeswarm using shap.summary_plot
    # show=False, save to buffer, return base64
    # Returns: base64 PNG string

def plot_waterfall_single(shap_result: dict, sample_index=0) -> str:
    # Waterfall plot for one representative sample
    # Shows contribution of each feature to that specific prediction
    # Returns: base64 PNG string

def plot_attribution_table(shap_result: dict) -> list[dict]:
    # Returns a list of dicts for rendering as HTML table
    # Each dict: {rank, feature, mean_abs_shap, pct_of_total, direction}
    # direction: "positive" if mean SHAP > 0, "negative" if < 0
    # Sorted by mean |SHAP| descending

def plot_prediction_distribution(model, profiles, shap_result) -> str:
    # Histogram of model predicted probabilities across all profiles
    # Coloured by predicted class (approved/denied)
    # Returns: base64 PNG string

def fig_to_base64(fig) -> str:
    # Convert matplotlib figure to base64 PNG string
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
```

---

## Step 7 — Flask Backend

### `app.py`

**Routes:**

```python
GET  /                  → serve index.html
POST /api/run-analysis  → run full pipeline, return JSON results
GET  /api/model-info    → return model_info.json contents
GET  /api/status        → return pipeline status (model loaded, data ready)
```

**POST /api/run-analysis — Request body:**
```json
{
  "n_profiles": 200,
  "use_synthetic": false,
  "sample_index_for_waterfall": 0
}
```

**POST /api/run-analysis — Response:**
```json
{
  "status": "success",
  "model_info": {
    "auc": 0.914,
    "accuracy": 0.871,
    "n_features": 14,
    "feature_names": [...]
  },
  "profiles_generated": 200,
  "shap_results": {
    "attribution_table": [
      {"rank": 1, "feature": "education_num", "mean_abs_shap": 0.142, "pct": 24.3, "direction": "positive"},
      ...
    ],
    "n_profiles_analysed": 200,
    "base_value": 0.231,
    "computation_time_seconds": 4.2
  },
  "plots": {
    "summary_bar": "<base64 PNG>",
    "beeswarm": "<base64 PNG>",
    "waterfall": "<base64 PNG>",
    "prediction_distribution": "<base64 PNG>"
  },
  "errors": []
}
```

**Pipeline execution in `/api/run-analysis`:**
```python
1. Load model from models/model.pkl
2. Load processed data from data/processed/
3. Generate profiles using ProfileGenerator
4. Assert profile dimensions match model expectations
5. Run ClassicalSHAPExplainer.explain(profiles)
6. Assert SHAP additivity check passes
7. Generate all plots
8. Build and return JSON response
```

**Error handling:** Wrap each step in try/except. Return errors in the `errors` list so the UI can display them without crashing. Never return HTTP 500 — always return JSON with status "error" and a message.

---

## Step 8 — Frontend UI

### `templates/index.html`

**Design direction:** Clean, data-focused, dark navy + white — like a professional financial audit tool. Sharp, precise, no decorative clutter. Every element earns its place.

**Colour palette:**
```css
--bg-primary: #0F172A;        /* dark navy */
--bg-secondary: #1E293B;      /* card background */
--bg-tertiary: #334155;       /* input/hover */
--accent: #3B82F6;            /* blue — Classical SHAP colour */
--accent-green: #10B981;      /* success */
--accent-red: #EF4444;        /* error/negative SHAP */
--text-primary: #F1F5F9;      /* white text */
--text-secondary: #94A3B8;    /* muted text */
--border: #334155;            /* borders */
```

**Layout — Single page, four sections:**

```
┌─────────────────────────────────────────────────────────┐
│  HEADER                                                  │
│  "LightGBM SHAP Analysis"  │  Model status badge        │
│  Mode: Classical SHAP      │  AUC: 0.914  Features: 14  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  CONTROL PANEL                                           │
│  Profiles: [200] slider    Use synthetic: [toggle]      │
│  Waterfall sample: [0]     [▶ Run Analysis] button      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RESULTS — shown after run                              │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │ Attribution Table│  │ Summary Bar Chart            │ │
│  │ Rank Feature  %  │  │ (base64 img)                 │ │
│  │  1  edu_num  24% │  │                              │ │
│  │  2  age      18% │  └──────────────────────────────┘ │
│  └──────────────────┘                                   │
│                                                         │
│  ┌──────────────────────┐  ┌────────────────────────┐   │
│  │ Beeswarm Plot        │  │ Waterfall (1 sample)   │   │
│  │ (base64 img)         │  │ (base64 img)           │   │
│  └──────────────────────┘  └────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Prediction Distribution                          │   │
│  │ (base64 img)                                     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  FOOTER                                                  │
│  Mode 1 of 6 — Classical SHAP │ Next: Causal SHAP       │
└─────────────────────────────────────────────────────────┘
```

**Behaviour:**

- On page load: call `GET /api/status` — show model info in header if loaded
- On "Run Analysis" click:
  - Show loading spinner with message "Running Classical SHAP analysis..."
  - POST to `/api/run-analysis` with current control values
  - On success: render all four plots + attribution table
  - On error: show error banner with message from response
- Attribution table: sortable by rank or by feature name
- Each plot: clicking opens full-size in a modal overlay
- Mutable features highlighted in light blue in attribution table
- Immutable features highlighted in light red

**Loading states:**
```
Step 1/4: Generating profiles...
Step 2/4: Running Classical SHAP...
Step 3/4: Generating visualisations...
Step 4/4: Building results...
```

Simulate step progress by streaming or polling — or just show a spinner with elapsed time counter.

---

## Step 9 — Requirements and Entry Point

### `requirements.txt`
```
lightgbm>=4.0
shap>=0.44
flask>=3.0
pandas>=2.0
scikit-learn>=1.3
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
pyyaml>=6.0
pyarrow>=14.0
ctgan>=0.9          # optional — only needed if use_synthetic=true
```

### `README.md`

```markdown
## Setup
pip install -r requirements.txt

## First run (trains model)
python data/preprocess.py
python models/train_lgbm.py

## Launch dashboard
python app.py
# Open http://localhost:5000
```

### `run_setup.py` — convenience script
```python
# Runs preprocess.py then train_lgbm.py in sequence
# Prints status at each step
# Run once before launching app.py
```

---

## Critical Implementation Rules

### Rule 1 — Feature Count Consistency
The single most important constraint. Every module must use the same feature list derived from `metadata.json`:

```python
# Load once, use everywhere
with open('data/processed/metadata.json') as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata['feature_names']
N_FEATURES = metadata['n_features']
```

Never hardcode feature names or counts anywhere in the codebase.

### Rule 2 — SHAP Additivity
Always verify SHAP values sum correctly after computation. If check fails, log the error and return it in the API response — do not silently continue.

### Rule 3 — Profile Dimension Assertion
Every time profiles are generated, assert shape matches model expectations before any model call.

### Rule 4 — Modular Explainer Interface
Each explainer must implement this interface so future modes can be added by dropping in a new file:

```python
class BaseExplainer:
    def explain(self, profiles: pd.DataFrame) -> dict:
        raise NotImplementedError

    def get_method_id(self) -> int:
        raise NotImplementedError

    def get_method_name(self) -> str:
        raise NotImplementedError
```

Classical SHAP (Method 2) inherits from this. Methods 3-6 will too.

### Rule 5 — No Silent Failures
Every exception must be caught, logged, and returned in the API response. The UI must always know what went wrong.

---

## What This Build Does NOT Include (Yet)

These are explicitly out of scope for this build — do not implement:

- Causal SHAP (Method 3)
- Layerwise SHAP (Method 4)
- Conditional SHAP (Method 5)
- Unified SHAP + Counterfactuals (Method 6)
- Debiasing module
- RL feedback loop
- Proxy detection
- Bias metrics (DPD/EOD/DI)
- Report PDF generation

The architecture must make all of these easy to add as the next modes — but they are not built here.

---

## Definition of Done

The build is complete when:

1. `python run_setup.py` completes without errors and prints AUC > 0.80
2. `python app.py` starts Flask on localhost:5000 without errors
3. Opening localhost:5000 shows the dashboard with model info in header
4. Clicking "Run Analysis" with default settings (200 profiles, no synthetic) completes successfully
5. All four plots render correctly in the UI
6. Attribution table shows all features ranked by mean |SHAP|
7. Mutable features are highlighted correctly
8. Feature count in UI matches `n_features` in model_info.json exactly
9. SHAP additivity assertion passes (logged in console)
10. No hardcoded feature names anywhere in the codebase
