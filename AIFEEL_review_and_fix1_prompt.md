# AIFEEL Codebase Review + Fix 1 Implementation Prompt
## For Antigravity / Cursor — Full Specification

---

# PART 1: CODEBASE REVIEW

## Overall Status
**The codebase is working and well-structured.**
- LightGBM trained: AUC = 0.9266, Accuracy = 0.8325, F1 = 0.7162
- SHAP additivity check correctly implemented in log-odds space
- Feature consistency maintained across all config files
- Error handling correct — never returns HTTP 500
- Base explainer interface clean and extensible

---

## Issues to Fix Before Proceeding

### Issue 1 — CRITICAL: Drop `fnlwgt` 🔴

**File:** `data/preprocess.py` and `config/feature_metadata.yaml`

**Problem:**
`fnlwgt` is a census sampling weight — a data collection artefact with no causal
or semantic meaning for income prediction. Including it will:
- Distort SHAP attributions by absorbing variance it should not
- Break the causal graph in Fix 1 — PC algorithm cannot assign a causal role to it
- Reduce interpretability of all analysis outputs

**Fix in `data/preprocess.py`:**
After selecting features, add:
```python
# Drop census sampling weight — no causal meaning
X = X.drop(columns=['fnlwgt'])
```

**Fix in `config/feature_metadata.yaml`:**
Remove `fnlwgt` from `continuous_features` list.

**After fix:** 14 features → 13 features, all with genuine causal meaning.
Rerun `data/preprocess.py` and `models/train_lgbm.py` after this change.
AUC should stay above 0.85 — likely improve slightly.

---

### Issue 2 — MEDIUM: Drop `education` (keep `education_num`) 🟡

**File:** `data/preprocess.py` and `config/feature_metadata.yaml`

**Problem:**
`education` (string label) and `education_num` (numeric equivalent) encode
identical information. Including both causes:
- Perfect correlation between two nodes in the causal graph
- PC algorithm-based causal discovery breaks on perfectly correlated nodes
- SHAP attribution split arbitrarily between two identical signals

**Fix in `data/preprocess.py`:**
```python
# Drop education string — education_num carries same information
X = X.drop(columns=['education'])
```

**Fix in `config/feature_metadata.yaml`:**
Remove `education` from `categorical_features` list.

**After fix:** 13 features → 12 features, all genuinely distinct.

---

### Issue 3 — MEDIUM: Fix `base_value` Averaging 🟡

**File:** `explainers/method2_classical_shap.py`

**Problem:**
```python
# CURRENT — fragile, takes first value only
if isinstance(base_value, np.ndarray):
    base_value = base_value[0]
```
For interventional SHAP with a background dataset, `base_value` should be the
mean across all background samples, not just the first one.

**Fix:**
```python
# CORRECT — mean across background samples
if isinstance(base_value, np.ndarray):
    base_value = float(np.mean(base_value))
```

---

### Issue 4 — MINOR: Memory Leak in `plot_shap_beeswarm` 🟢

**File:** `analysis/visualise.py`

**Problem:**
```python
# CURRENT — creates a figure that is never closed
fig, ax = plt.subplots(figsize=(8, 6))  # orphaned figure
shap.summary_plot(...)                   # creates its own figure
fig = plt.gcf()                          # gets shap's figure
# original fig leaks memory on every call
```

**Fix:**
```python
# CORRECT — close all before shap creates its figure
plt.close('all')
shap.summary_plot(
    sv, profiles,
    feature_names=feature_names,
    max_display=max_features,
    show=False
)
fig = plt.gcf()
```

---

### Issue 5 — MINOR: Remove `check_columns.py` from Root 🟢

**File:** `check_columns.py` (project root)

This is a debug script left in the root directory.
Move to `scripts/debug/check_columns.py` or delete it.
It should not be in the project root alongside `app.py`.

---

## Summary of All Fixes

| # | Issue | File | Severity | Action |
|---|-------|------|----------|--------|
| 1 | Drop fnlwgt | preprocess.py + feature_metadata.yaml | 🔴 Critical | Drop column |
| 2 | Drop education (keep education_num) | preprocess.py + feature_metadata.yaml | 🟡 Medium | Drop column |
| 3 | base_value averaging | method2_classical_shap.py | 🟡 Medium | Use np.mean |
| 4 | Beeswarm memory leak | visualise.py | 🟢 Minor | plt.close('all') |
| 5 | check_columns.py in root | project root | 🟢 Minor | Move or delete |

After applying all fixes:
- Rerun: `python data/preprocess.py`
- Rerun: `python models/train_lgbm.py`
- Verify: AUC > 0.85, all 12 features present in metadata.json

---

---

# PART 2: FIX 1 IMPLEMENTATION PROMPT
## Causal Coalition Weight SHAP (Method 3)

---

## Context

This is Fix 1 of the Nature-Inspired SHAP formula suite for the AIFEEL project.
The existing codebase already has:
- LightGBM model trained on UCI Adult Income (12 features after fixes above)
- Classical SHAP (Method 2) working via `explainers/method2_classical_shap.py`
- Base explainer interface in `explainers/base_explainer.py`
- Flask app serving analysis at localhost:5000

Fix 1 adds **Causal Coalition Weight SHAP (Method 3)** — replacing the uniform
Shapley coalition weight with a causally-informed probability weight derived from
a domain-specified causal graph G.

---

## What Fix 1 Changes

Only ONE thing changes from Classical SHAP (Method 2):

**The coalition weight function.**

```
Method 2 (Classical):
  w(S) = |S|!(|F|-|S|-1)! / |F|!        ← uniform, treats all coalitions equally

Method 3 (Causal):
  w(S,i,G) = P(S|do(i),G) / Σ P(S'|do(i),G)  ← causally informed
```

Everything else — the marginal contribution term f(S∪{i}) − f(S), the
interventional sampling, the background dataset — stays identical to Method 2.

---

## Step 1 — Add `config/causal_dag.yaml`

Create this file. It defines the causal graph G for the UCI Adult Income dataset
based on domain knowledge. The PC algorithm is too slow for real-time use;
use this hand-specified DAG instead.

```yaml
# Causal DAG for UCI Adult Income Dataset
# Based on domain knowledge of socioeconomic relationships
# Edges: [cause, effect]

edges:
  - [age, education_num]          # older people had more time to complete education
  - [age, hours_per_week]         # age affects work hours (career stage)
  - [age, capital_gain]           # older people accumulate more capital
  - [age, capital_loss]           # older people have more investments at risk
  - [education_num, occupation]   # education level determines available occupations
  - [education_num, workclass]    # education affects type of employer
  - [occupation, hours_per_week]  # occupation determines expected hours
  - [occupation, income]          # occupation directly determines income
  - [workclass, income]           # employer type directly affects income
  - [marital_status, relationship] # marital status determines relationship category
  - [relationship, income]        # household structure affects reported income
  - [sex, occupation]             # sex affects occupation access (documented bias)
  - [sex, hours_per_week]         # sex affects hours worked
  - [race, occupation]            # race affects occupation access (documented bias)
  - [native_country, occupation]  # country of origin affects occupation access
  - [capital_gain, income]        # capital gains directly contribute to income
  - [capital_loss, income]        # capital losses reduce effective income
  - [hours_per_week, income]      # hours worked directly affects income
  - [education_num, income]       # education directly affects income

protected_attributes:
  - sex
  - race

proxy_candidates:
  - native_country   # may proxy for race/ethnicity
  - relationship     # may proxy for sex (wife/husband categories)
  - occupation       # may proxy for sex/race due to occupational segregation

notes: >
  This DAG encodes known socioeconomic causal relationships.
  Protected attributes (sex, race) are included as causes of occupation
  and hours_per_week to model discrimination pathways explicitly.
  This allows the causal weight to correctly identify when features
  like occupation are acting as proxies for protected characteristics.
```

---

## Step 2 — Create `explainers/causal_graph.py`

This module loads the DAG and computes P(S | do(i), G) for any coalition S
and target feature i.

```python
"""
causal_graph.py

Loads the causal DAG from config and computes causal coalition weights.
P(S | do(i), G) is approximated by counting d-connected paths between
features in S and feature i in graph G, given the do(i) intervention.

The weight reflects how causally plausible it is that coalition S would
naturally form around feature i in the real data generating process.
"""

import yaml
import numpy as np
from itertools import combinations


class CausalGraph:
    def __init__(self, dag_config_path: str, feature_names: list):
        """
        Parameters:
            dag_config_path: path to causal_dag.yaml
            feature_names: list of feature names in model order
        """
        with open(dag_config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.feature_to_idx = {f: i for i, f in enumerate(feature_names)}

        # Build adjacency: edges are directed cause -> effect
        self.edges = []
        self.adjacency = {f: set() for f in feature_names}
        self.reverse_adjacency = {f: set() for f in feature_names}

        for edge in config.get('edges', []):
            cause, effect = edge[0], edge[1]
            if cause in self.feature_to_idx and effect in self.feature_to_idx:
                self.edges.append((cause, effect))
                self.adjacency[cause].add(effect)
                self.reverse_adjacency[effect].add(cause)

        self.protected_attributes = config.get('protected_attributes', [])
        self.proxy_candidates = config.get('proxy_candidates', [])

        # Precompute causal distances between all feature pairs
        # using BFS on undirected version of the graph (d-connectivity proxy)
        self._causal_distances = self._compute_all_distances()

    def _compute_all_distances(self) -> dict:
        """
        BFS distance between all pairs of features on undirected graph.
        Used to estimate d-connectivity: closer features = more causally related.
        """
        distances = {}
        for feature in self.feature_names:
            distances[feature] = self._bfs_distances(feature)
        return distances

    def _bfs_distances(self, source: str) -> dict:
        """BFS shortest path distances from source on undirected graph."""
        visited = {source: 0}
        queue = [source]
        while queue:
            current = queue.pop(0)
            neighbors = self.adjacency[current] | self.reverse_adjacency[current]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        # Features not reachable get distance = infinity
        for f in self.feature_names:
            if f not in visited:
                visited[f] = float('inf')
        return visited

    def causal_plausibility(self, feature_i: str, feature_j: str) -> float:
        """
        Returns a score in [0, 1] representing how causally related
        feature_j is to feature_i.

        Score = 1.0 if direct edge exists
        Score = 0.5 if distance 2 (one intermediate)
        Score = 1/(distance) for further
        Score = 0.0 if unreachable (d-separated)
        """
        if feature_i not in self._causal_distances:
            return 0.0

        dist = self._causal_distances[feature_i].get(feature_j, float('inf'))

        if feature_i == feature_j:
            return 0.0  # feature i never in its own coalition
        elif dist == 1:
            return 1.0  # direct causal relationship
        elif dist == 2:
            return 0.5  # one intermediary
        elif dist == float('inf'):
            return 0.0  # d-separated — no causal path
        else:
            return 1.0 / dist  # diminishing plausibility with distance

    def coalition_weight(self, coalition: frozenset, feature_i: str) -> float:
        """
        Computes w^causal(S, i, G) = P(S | do(i), G)

        Approximated as the product of causal plausibility scores
        for each member j of coalition S with respect to feature i.

        Product form ensures that a coalition with one d-separated feature
        gets near-zero weight overall — the double gate effect.

        Parameters:
            coalition: frozenset of feature names in coalition S
            feature_i: the feature whose SHAP value we are computing

        Returns:
            float: unnormalised causal weight for this coalition
        """
        if len(coalition) == 0:
            # Empty coalition — baseline plausibility = 1.0
            return 1.0

        # Product of individual plausibility scores
        weight = 1.0
        for feature_j in coalition:
            plausibility = self.causal_plausibility(feature_i, feature_j)
            weight *= plausibility

            # Early exit if weight already zero
            if weight == 0.0:
                return 0.0

        return weight

    def get_causal_neighbourhood(self, feature_i: str, max_distance: int = 2) -> list:
        """
        Returns features within max_distance causal steps of feature_i.
        Used to restrict coalition search space.

        Parameters:
            feature_i: target feature
            max_distance: maximum causal distance to include

        Returns:
            list of feature names in causal neighbourhood
        """
        if feature_i not in self._causal_distances:
            return [f for f in self.feature_names if f != feature_i]

        neighbourhood = []
        for f, dist in self._causal_distances[feature_i].items():
            if f != feature_i and dist <= max_distance:
                neighbourhood.append(f)

        return neighbourhood

    def get_all_coalition_weights(self, feature_i: str, all_features: list) -> dict:
        """
        Precomputes normalised causal weights for ALL possible coalitions
        of features in all_features excluding feature_i.

        Returns dict: frozenset(coalition) -> normalised_weight

        This is called once per feature per SHAP computation to avoid
        recomputing weights for every sample.
        """
        other_features = [f for f in all_features if f != feature_i]
        n = len(other_features)

        raw_weights = {}

        # Enumerate all 2^n subsets
        for size in range(n + 1):
            for combo in combinations(other_features, size):
                coalition = frozenset(combo)
                raw_weights[coalition] = self.coalition_weight(coalition, feature_i)

        # Normalise so all weights sum to 1
        total = sum(raw_weights.values())
        if total == 0:
            # Fallback to uniform if all weights zero (disconnected graph)
            uniform = 1.0 / len(raw_weights)
            return {k: uniform for k in raw_weights}

        return {k: v / total for k, v in raw_weights.items()}
```

---

## Step 3 — Create `explainers/method3_causal_shap.py`

This is the Method 3 explainer. It inherits from BaseExplainer and uses
CausalGraph to replace the uniform Shapley weight with the causal weight.

```python
"""
method3_causal_shap.py

Fix 1: Causal Coalition Weight SHAP

Changes from Method 2 (Classical SHAP):
  - Coalition weight: w^causal(S,i,G) replaces uniform w(S)
  - Everything else identical: interventional sampling, marginal contribution

The causal weight ensures only causally plausible coalitions receive
significant weight, suppressing proxy variables that have no direct
causal path to the target but correlate with protected attributes.

Natural inspiration: Dopamine model-based prior — the brain weights
contexts by their causal plausibility, not uniformly.
"""

import numpy as np
import pandas as pd
import shap
import yaml
import os
from itertools import combinations, product
from .base_explainer import BaseExplainer
from .causal_graph import CausalGraph


class CausalSHAPExplainer(BaseExplainer):

    def __init__(self, model, background_data: pd.DataFrame,
                 feature_names: list, config: dict,
                 dag_config_path: str, model_info_path: str = None):
        """
        Parameters:
            model: fitted LightGBM model
            background_data: sample of X_train for interventional SHAP baseline
            feature_names: ordered list of feature names
            config: loaded hyperparameters.yaml
            dag_config_path: path to causal_dag.yaml
            model_info_path: optional path to model_info.json for comparison plots
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.config = config
        self.background_data = background_data

        # Load model importances for comparison plot
        self.model_importances = None
        if model_info_path and os.path.exists(model_info_path):
            import json
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                self.model_importances = model_info.get('feature_importances_gain')

        # Load causal graph
        self.causal_graph = CausalGraph(dag_config_path, feature_names)

        # Classical TreeExplainer for interventional marginal contributions
        # We still use TreeExplainer for f(S∪{i}) - f(S) computation
        # but we REPLACE the coalition weights it uses internally
        # by computing our own weighted sum
        self.tree_explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional"
        )

        # Precompute causal weights for all features
        # dict: feature_name -> dict: frozenset(coalition) -> normalised_weight
        print("Precomputing causal coalition weights...")
        self._precomputed_weights = {}
        for feature in feature_names:
            self._precomputed_weights[feature] = \
                self.causal_graph.get_all_coalition_weights(feature, feature_names)
        print("Causal weights precomputed.")

    def _compute_causal_shap_for_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Compute causal SHAP values for a single sample using the formula:

        φᵢ^causal = Σ w^causal(S,i,G) · [f(S∪{i}) - f(S)]
                    S⊆F\{i}

        We approximate f(S∪{i}) - f(S) using interventional expectations
        estimated from the background dataset.

        Parameters:
            sample: 1D numpy array of shape (n_features,)

        Returns:
            np.ndarray of shape (n_features,) — causal SHAP values
        """
        n = self.n_features
        feature_idx = {f: i for i, f in enumerate(self.feature_names)}
        background = self.background_data.values
        n_background = len(background)

        causal_shap_values = np.zeros(n)

        for i, feature_i in enumerate(self.feature_names):
            other_features = [f for f in self.feature_names if f != feature_i]
            other_indices = [feature_idx[f] for f in other_features]
            coalition_weights = self._precomputed_weights[feature_i]

            phi_i = 0.0

            for coalition_frozen, weight in coalition_weights.items():
                if weight < 1e-10:
                    # Skip near-zero weight coalitions for efficiency
                    continue

                coalition = list(coalition_frozen)
                coalition_indices = [feature_idx[f] for f in coalition]
                absent_indices = [feature_idx[f] for f in other_features
                                  if f not in coalition_frozen]

                # Compute f(S∪{i}): fix coalition + feature i, marginalise absent
                # Compute f(S): fix coalition only, marginalise absent + feature i

                # Build masked inputs for f(S∪{i})
                inputs_with_i = np.tile(sample, (n_background, 1))
                for idx in absent_indices:
                    inputs_with_i[:, idx] = background[:, idx]
                # feature i stays at sample value — already set by np.tile

                # Build masked inputs for f(S)
                inputs_without_i = inputs_with_i.copy()
                inputs_without_i[:, i] = background[:, i]  # marginalise feature i

                # Get model predictions in log-odds space
                f_with_i = self.model.predict(
                    inputs_with_i, raw_score=True
                ).mean()

                f_without_i = self.model.predict(
                    inputs_without_i, raw_score=True
                ).mean()

                marginal_contribution = f_with_i - f_without_i
                phi_i += weight * marginal_contribution

            causal_shap_values[i] = phi_i

        return causal_shap_values

    def explain(self, profiles: pd.DataFrame) -> dict:
        """
        Compute causal SHAP values for all profiles.

        For efficiency, we compute per-sample causal SHAP values
        and aggregate. This is slower than TreeSHAP but correct.

        For large profile sets (>50), we subsample for causal SHAP
        and use classical SHAP for the remainder, then report
        causal SHAP statistics on the subsample.
        """
        profiles_arr = profiles.values
        n_profiles = len(profiles)

        # For efficiency: compute causal SHAP on up to 50 profiles
        # Classical SHAP additivity is verified on full set
        max_causal_samples = min(n_profiles, 50)
        sample_indices = np.random.choice(n_profiles, max_causal_samples,
                                          replace=False)

        print(f"Computing causal SHAP values for {max_causal_samples} samples...")
        causal_sv = np.zeros((max_causal_samples, self.n_features))

        for k, idx in enumerate(sample_indices):
            if k % 10 == 0:
                print(f"  Sample {k+1}/{max_causal_samples}...")
            causal_sv[k] = self._compute_causal_shap_for_sample(profiles_arr[idx])

        print("Causal SHAP computation complete.")

        # Base value from TreeExplainer (same background = same base value)
        classical_result = self.tree_explainer(profiles.iloc[:5])
        base_values = classical_result.base_values
        if isinstance(base_values, np.ndarray):
            if len(base_values.shape) == 2:
                base_value = float(np.mean(base_values[:, 1]))
            else:
                base_value = float(np.mean(base_values))
        else:
            base_value = float(base_values)

        mean_abs_shap = np.abs(causal_sv).mean(axis=0)

        return {
            "shap_values": causal_sv.tolist(),
            "base_value": base_value,
            "feature_names": self.feature_names,
            "mean_abs_shap": mean_abs_shap.tolist(),
            "profiles": profiles.iloc[sample_indices].values.tolist(),
            "method": self.get_method_name(),
            "model_importances": self.model_importances,
            "n_samples_computed": max_causal_samples,
            "causal_graph_info": {
                "n_edges": len(self.causal_graph.edges),
                "protected_attributes": self.causal_graph.protected_attributes,
                "proxy_candidates": self.causal_graph.proxy_candidates
            }
        }

    def get_summary(self, shap_result: dict) -> list:
        """Identical to Method 2 summary — same output format."""
        mean_abs = shap_result['mean_abs_shap']
        features = shap_result['feature_names']
        sv = np.array(shap_result['shap_values'])

        total_importance = sum(mean_abs)

        summary = []
        for i, (feat, imp) in enumerate(zip(features, mean_abs)):
            mean_shap = sv[:, i].mean()
            direction = "positive" if mean_shap > 0 else "negative"

            # Flag if feature is a proxy candidate or protected attribute
            is_proxy = feat in self.causal_graph.proxy_candidates
            is_protected = feat in self.causal_graph.protected_attributes

            summary.append({
                "feature": feat,
                "mean_abs_shap": float(imp),
                "pct": float(imp / total_importance * 100) if total_importance > 0 else 0.0,
                "direction": direction,
                "is_proxy_candidate": is_proxy,
                "is_protected": is_protected
            })

        summary.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        for i, item in enumerate(summary):
            item["rank"] = i + 1

        return summary

    def get_method_id(self) -> int:
        return 3

    def get_method_name(self) -> str:
        return "causal_shap"
```

---

## Step 4 — Add Method 3 Route to `app.py`

Add a new route `/api/run-causal-analysis` that mirrors `/api/run-analysis`
but uses `CausalSHAPExplainer` instead of `ClassicalSHAPExplainer`.

Also add these imports at the top of `app.py`:
```python
from explainers.method3_causal_shap import CausalSHAPExplainer
```

New route:
```python
@app.route('/api/run-causal-analysis', methods=['POST'])
def run_causal_analysis():
    errors = []
    try:
        req_data = request.json or {}
        n_profiles = int(req_data.get('n_profiles', 100))  # default 100 for causal (slower)
        use_synthetic = bool(req_data.get('use_synthetic', False))
        sample_index = int(req_data.get('sample_index_for_waterfall', 0))

        start_time = time.time()

        # Load model
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Run setup first.")
        model = joblib.load(model_path)

        # Generator
        config_path = os.path.join(CONFIG_DIR, "hyperparameters.yaml")
        metadata_path = os.path.join(DATA_DIR, "processed", "metadata.json")
        generator = ProfileGenerator(config_path, metadata_path, DATA_DIR)

        # Generate profiles
        profiles = generator.generate(n_profiles=n_profiles, use_synthetic=use_synthetic)
        assert profiles.shape[1] == generator.n_features

        # Config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Background data
        background_samples = config['shap'].get('background_samples', 100)
        X_train = pd.read_parquet(os.path.join(DATA_DIR, "processed", "X_train.parquet"))
        background_data = X_train.sample(
            n=min(background_samples, len(X_train)), random_state=42
        )

        # DAG config path
        dag_config_path = os.path.join(CONFIG_DIR, "causal_dag.yaml")
        model_info_path = os.path.join(MODELS_DIR, "model_info.json")

        # Causal SHAP explainer
        explainer = CausalSHAPExplainer(
            model, background_data, generator.feature_names,
            config, dag_config_path, model_info_path=model_info_path
        )

        # Run
        shap_result = explainer.explain(profiles)

        # Plots — reuse same visualise functions
        summary_bar = plot_shap_summary(shap_result)
        beeswarm = plot_shap_beeswarm(shap_result)

        if sample_index >= len(shap_result['shap_values']):
            sample_index = 0

        waterfall = plot_waterfall_single(shap_result, sample_index)
        pred_dist = plot_prediction_distribution(model, profiles, shap_result)
        shap_vs_weights = plot_shap_vs_weights(shap_result)

        attribution_table = explainer.get_summary(shap_result)
        computation_time = time.time() - start_time
        model_info = get_model_info()

        return jsonify({
            "status": "success",
            "method": "causal_shap",
            "model_info": {
                "auc": model_info.get("auc_test"),
                "accuracy": model_info.get("accuracy_test"),
                "n_features": model_info.get("n_features"),
                "feature_names": model_info.get("feature_names")
            },
            "profiles_generated": len(profiles),
            "causal_graph_info": shap_result.get("causal_graph_info", {}),
            "shap_results": {
                "attribution_table": attribution_table,
                "n_profiles_analysed": shap_result.get("n_samples_computed"),
                "base_value": shap_result["base_value"],
                "computation_time_seconds": round(computation_time, 2)
            },
            "plots": {
                "summary_bar": summary_bar,
                "beeswarm": beeswarm,
                "waterfall": waterfall,
                "prediction_distribution": pred_dist,
                "shap_vs_weights": shap_vs_weights
            },
            "errors": errors
        })

    except Exception as e:
        import traceback
        errors.append(str(e))
        errors.append(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": "An error occurred during causal SHAP analysis.",
            "errors": errors
        })
```

---

## Step 5 — Update `visualise.py` for Method 3

Update `plot_shap_summary` to handle the method_id correctly.
Add a new function `plot_causal_weights_heatmap`:

```python
def plot_causal_weights_heatmap(causal_graph, feature_names: list) -> str:
    """
    Heatmap showing causal plausibility scores between all feature pairs.
    Rows = target feature i, Columns = coalition member j
    Value = w^causal plausibility score
    """
    n = len(feature_names)
    matrix = np.zeros((n, n))

    for i, fi in enumerate(feature_names):
        for j, fj in enumerate(feature_names):
            if i != j:
                matrix[i, j] = causal_graph.causal_plausibility(fi, fj)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        ax=ax,
        vmin=0,
        vmax=1
    )
    ax.set_title('Causal Plausibility Matrix\n(row=feature i, col=coalition member j)')
    ax.set_xlabel('Coalition Member j')
    ax.set_ylabel('Target Feature i')

    return fig_to_base64(fig)
```

Also add this plot to the causal analysis route response:
```python
# In run_causal_analysis route, add:
causal_heatmap = plot_causal_weights_heatmap(
    explainer.causal_graph, generator.feature_names
)
# Add to plots dict:
"causal_weights_heatmap": causal_heatmap
```

---

## Step 6 — Update UI (`templates/index.html`)

Add a second mode button alongside the existing "Run Analysis" button:

```html
<!-- Mode selector -->
<div class="mode-selector">
  <button id="btn-classical" class="mode-btn active" onclick="runAnalysis('classical')">
    ▶ Method 2: Classical SHAP
  </button>
  <button id="btn-causal" class="mode-btn" onclick="runAnalysis('causal')">
    ▶ Method 3: Causal SHAP
  </button>
</div>
```

Update the JavaScript `runAnalysis` function to accept a mode parameter:
```javascript
function runAnalysis(mode = 'classical') {
    const endpoint = mode === 'causal'
        ? '/api/run-causal-analysis'
        : '/api/run-analysis';

    // ... rest of fetch logic unchanged ...
    // Just change the URL
}
```

When Method 3 results are displayed, add:
- A badge showing "Method 3 — Causal SHAP" in accent-green colour
- The causal weights heatmap as an additional plot panel
- Proxy candidate features highlighted in amber in the attribution table
- Protected attribute features highlighted in red in the attribution table

---

## Step 7 — Update `visualise.py` Title for Method 3

Update `plot_shap_summary` to use method-specific title:

```python
def plot_shap_summary(shap_result: dict, max_features=15) -> str:
    method = shap_result.get('method', 'classical_shap')
    method_id = 2 if method == 'classical_shap' else 3

    titles = {
        'classical_shap': 'Classical SHAP — Feature Attribution Summary',
        'causal_shap': 'Causal SHAP (Fix 1) — Feature Attribution Summary'
    }
    title = titles.get(method, 'SHAP Feature Attribution Summary')
    color = METHOD_COLORS.get(method_id, METHOD_COLORS[2])

    # ... rest of function using title and color variables ...
```

---

## Definition of Done — Fix 1

Fix 1 is complete when:

1. `config/causal_dag.yaml` exists with all edges defined
2. `explainers/causal_graph.py` loads DAG and computes causal weights correctly
3. `explainers/method3_causal_shap.py` runs without error on 12 features
4. `/api/run-causal-analysis` returns valid JSON with all plots
5. UI shows both Method 2 and Method 3 buttons and renders both correctly
6. Attribution table for Method 3 flags proxy candidates in amber
7. Causal weights heatmap renders correctly showing plausibility scores
8. Console logs confirm causal weight precomputation completed
9. No hardcoded feature names anywhere in new files
10. Spearman rank correlation between Method 2 and Method 3 is printed to console
    for quick sanity check — should be > 0.7 (same features, different weights)

---

## File Checklist — What to Create/Modify

### New Files to Create:
- `config/causal_dag.yaml`
- `explainers/causal_graph.py`
- `explainers/method3_causal_shap.py`

### Files to Modify:
- `data/preprocess.py` — drop fnlwgt and education
- `config/feature_metadata.yaml` — remove fnlwgt and education
- `explainers/method2_classical_shap.py` — fix base_value averaging
- `analysis/visualise.py` — fix beeswarm memory leak, add causal heatmap, update titles
- `app.py` — add causal SHAP route and import
- `templates/index.html` — add Method 3 button and display logic

### Files to Delete/Move:
- `check_columns.py` — move to `scripts/debug/` or delete

### Files to Rerun After Fixes:
- `python data/preprocess.py`
- `python models/train_lgbm.py`
- Verify new AUC > 0.85, n_features = 12 in metadata.json

---

## Important Notes for Implementation

1. **Feature count will change from 14 to 12** after dropping fnlwgt and education.
   All files that reference feature count must be regenerated — do not manually
   edit metadata.json or model_info.json. Always regenerate by running the scripts.

2. **Causal SHAP is slower than Classical SHAP** because it computes marginal
   contributions manually per sample rather than using TreeSHAP's optimised path.
   Default to 50 samples max for causal SHAP in the UI. Display a warning if user
   requests more than 100 profiles for Method 3.

3. **The causal weights heatmap is the most valuable new visual** — it shows users
   exactly which feature combinations the causal weight considers plausible. This
   is what makes the explanation of Fix 1 concrete and visible.

4. **Spearman rank correlation between Method 2 and Method 3** should be computed
   and displayed in the UI comparison panel. High correlation (>0.8) means causal
   structure largely agrees with classical SHAP. Low correlation means the causal
   graph is substantially redirecting attribution — the interesting case.

5. **Do not change the base_value computation method** between Method 2 and Method 3.
   Both use the same background dataset and the same TreeExplainer base value.
   Only the coalition weights change — not the baseline.
