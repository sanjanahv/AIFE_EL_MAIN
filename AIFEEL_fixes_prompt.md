# AIFEEL Codebase — Fix All 5 Issues
## For Antigravity / Cursor — Precise Fix Specification

---

## Context

This is the AIFEEL project — a LightGBM + multi-method SHAP analysis dashboard.
The codebase currently has Methods 2, 3, and 4 implemented and working.
Apply all 5 fixes below exactly as specified. Do not change anything else.

---

## Fix 1 — Remove unused `other_indices` variable
**File:** `explainers/method3_causal_shap.py`
**Severity:** Medium — dead code running 12 × n_coalitions times per sample

**Find this block inside `_compute_causal_shap_for_sample`:**
```python
for i, feature_i in enumerate(self.feature_names):
    other_features = [f for f in self.feature_names if f != feature_i]
    other_indices = [feature_idx[f] for f in other_features]
    coalition_weights = self._precomputed_weights[feature_i]
```

**Replace with:**
```python
for i, feature_i in enumerate(self.feature_names):
    other_features = [f for f in self.feature_names if f != feature_i]
    coalition_weights = self._precomputed_weights[feature_i]
```

Remove the `other_indices` line entirely. It is computed but never used anywhere
in the method. This removes unnecessary list comprehension overhead from the
innermost loop.

---

## Fix 2 — Remove unused imports in method4
**File:** `explainers/method4_layerwise_shap.py`
**Severity:** Medium — unused imports cause confusion and lint warnings

**Find this import line at the top of the file:**
```python
from itertools import combinations, product
```

**Remove it entirely.** Method 4 does not use `combinations` or `product`
anywhere. These were carried over from method3 by mistake.

Also check for and remove any other unused imports in this file:
- If `yaml` is imported but not used, remove it
- If `os` is imported but not used, remove it
- Keep: `numpy`, `pandas`, `shap`, `json`, `BaseExplainer`

---

## Fix 3 — Add safety clip to stage limits
**File:** `explainers/method4_layerwise_shap.py`
**Severity:** Medium — floating point can produce tree_limit > num_trees causing crash

**Find this block in `__init__`:**
```python
self.stage_limits = [
    int(np.ceil(self.num_trees * (d + 1) / self.n_stages))
    for d in range(self.n_stages)
]
```

**Replace with:**
```python
self.stage_limits = [
    min(int(np.ceil(self.num_trees * (d + 1) / self.n_stages)), self.num_trees)
    for d in range(self.n_stages)
]
```

The `min(..., self.num_trees)` clip ensures no stage limit ever exceeds the
total number of trained trees, preventing a potential LightGBM crash when
`tree_limit` is out of range.

---

## Fix 4 — Update misleading comment in method3
**File:** `explainers/method3_causal_shap.py`
**Severity:** Minor — comment is technically imprecise

**Find this comment block inside `_compute_causal_shap_for_sample`:**
```python
# Compute f(S∪{i}): fix coalition + feature i, marginalise absent
# Compute f(S): fix coalition only, marginalise absent + feature i
```

**Replace with:**
```python
# f(S∪{i}): sample value for coalition + feature i; background values for absent features
# f(S): sample value for coalition only; background values for absent features + feature i
```

This accurately describes what the code does — absent features receive
background dataset values (interventional marginalisation), not true
mathematical absence.

---

## Fix 5 — Add cross-method comparison plot
**Files:** `analysis/visualise.py` and `app.py`
**Severity:** Medium — critical visual missing for paper validation

### 5a — Add two functions to `analysis/visualise.py`

Add both functions at the end of `visualise.py`, before any `if __name__` block:

```python
def plot_method_comparison(results: dict) -> str:
    """
    Side-by-side grouped bar chart comparing mean |SHAP| per feature
    across all available methods.

    Parameters:
        results: dict of method_name -> shap_result_dict
                 e.g. {"Classical SHAP": result2,
                       "Causal SHAP": result3,
                       "Layerwise SHAP": result4}

    Returns:
        base64 PNG string
    """
    methods = list(results.keys())
    if not methods:
        return ""

    feature_names = results[methods[0]]['feature_names']
    n_features = len(feature_names)
    n_methods = len(methods)

    x = np.arange(n_features)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(12, n_features * 1.2), 6))

    method_color_map = {
        'Classical SHAP':   METHOD_COLORS[2],
        'Causal SHAP':      METHOD_COLORS[3],
        'Layerwise SHAP':   METHOD_COLORS[4],
        'Conditional SHAP': METHOD_COLORS[5],
        'Unified SHAP':     METHOD_COLORS[6],
    }

    for k, method_name in enumerate(methods):
        mean_abs = np.array(results[method_name]['mean_abs_shap'])
        offset = (k - n_methods / 2 + 0.5) * width
        color = method_color_map.get(method_name, '#6B7280')
        ax.bar(
            x + offset, mean_abs, width,
            label=method_name,
            color=color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Mean |SHAP Value|')
    ax.set_title('Attribution Comparison Across SHAP Methods',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-0.5, n_features - 0.5)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_attribution_shift_table(results: dict) -> str:
    """
    Heatmap showing how much attribution SHIFTS for each feature
    between consecutive methods.

    Rows = features
    Columns = method transitions (e.g. Classical->Causal, Causal->Layerwise)
    Colour = diverging (red = attribution increased, blue = decreased)

    Parameters:
        results: ordered dict of method_name -> shap_result_dict

    Returns:
        base64 PNG string
    """
    methods = list(results.keys())
    if len(methods) < 2:
        return ""

    feature_names = results[methods[0]]['feature_names']
    n_features = len(feature_names)
    n_transitions = len(methods) - 1

    shift_matrix = np.zeros((n_features, n_transitions))
    col_labels = []

    for t in range(n_transitions):
        prev_method = methods[t]
        curr_method = methods[t + 1]
        prev_abs = np.array(results[prev_method]['mean_abs_shap'])
        curr_abs = np.array(results[curr_method]['mean_abs_shap'])
        shift_matrix[:, t] = curr_abs - prev_abs
        col_labels.append(
            f"{prev_method.split()[0]}->{curr_method.split()[0]}"
        )

    fig, ax = plt.subplots(
        figsize=(max(8, n_transitions * 2.5), max(6, n_features * 0.5))
    )

    vmax = np.abs(shift_matrix).max()
    vmax = max(vmax, 0.001)

    import seaborn as sns
    sns.heatmap(
        shift_matrix,
        xticklabels=col_labels,
        yticklabels=feature_names,
        annot=True,
        fmt='.4f',
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        linewidths=0.3,
        linecolor='#334155'
    )

    ax.set_title(
        'Attribution Shift Per Method Transition\n'
        '(Red = attribution increased, Blue = decreased)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('Method Transition')
    ax.set_ylabel('Feature')

    plt.tight_layout()
    return fig_to_base64(fig)
```

### 5b — Add imports to `app.py`

Update the visualise import block in `app.py` to include both new functions:

```python
from analysis.visualise import (
    plot_shap_summary,
    plot_shap_beeswarm,
    plot_waterfall_single,
    plot_prediction_distribution,
    plot_shap_vs_weights,
    get_attribution_table,
    plot_causal_weights_heatmap,
    plot_layerwise_depth_heatmap,
    plot_method_comparison,
    plot_attribution_shift_table
)
```

### 5c — Add `/api/compare-methods` route to `app.py`

Add this route after the existing three routes, before `if __name__ == '__main__'`:

```python
@app.route('/api/compare-methods', methods=['POST'])
def compare_methods():
    """
    Runs all available SHAP methods and returns a side-by-side
    comparison plot and attribution shift heatmap.

    Request body (all optional):
    {
        "n_profiles": 100,
        "use_synthetic": false,
        "methods": ["classical", "causal", "layerwise"]
    }
    """
    errors = []
    try:
        req_data = request.json or {}
        n_profiles = int(req_data.get('n_profiles', 100))
        use_synthetic = bool(req_data.get('use_synthetic', False))
        requested_methods = req_data.get(
            'methods', ['classical', 'causal', 'layerwise']
        )

        start_time = time.time()

        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found. Run setup first.')
        model = joblib.load(model_path)

        config_path = os.path.join(CONFIG_DIR, 'hyperparameters.yaml')
        metadata_path = os.path.join(DATA_DIR, 'processed', 'metadata.json')
        generator = ProfileGenerator(config_path, metadata_path, DATA_DIR)

        profiles = generator.generate(
            n_profiles=n_profiles,
            use_synthetic=use_synthetic
        )
        assert profiles.shape[1] == generator.n_features

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        background_samples = config['shap'].get('background_samples', 100)
        X_train = pd.read_parquet(
            os.path.join(DATA_DIR, 'processed', 'X_train.parquet')
        )
        background_data = X_train.sample(
            n=min(background_samples, len(X_train)), random_state=42
        )

        dag_config_path = os.path.join(CONFIG_DIR, 'causal_dag.yaml')
        model_info_path = os.path.join(MODELS_DIR, 'model_info.json')

        method_results = {}
        method_labels = {
            'classical': 'Classical SHAP',
            'causal':    'Causal SHAP',
            'layerwise': 'Layerwise SHAP',
        }

        for method_key in requested_methods:
            try:
                if method_key == 'classical':
                    exp = ClassicalSHAPExplainer(
                        model, background_data,
                        generator.feature_names, config,
                        model_info_path=model_info_path
                    )
                    result = exp.explain(profiles)

                elif method_key == 'causal':
                    exp = CausalSHAPExplainer(
                        model, background_data,
                        generator.feature_names, config,
                        dag_config_path,
                        model_info_path=model_info_path
                    )
                    result = exp.explain(profiles)

                elif method_key == 'layerwise':
                    exp = LayerwiseSHAPExplainer(
                        model, background_data,
                        generator.feature_names, config,
                        model_info_path=model_info_path
                    )
                    result = exp.explain(profiles)

                else:
                    errors.append(f"Unknown method: {method_key}")
                    continue

                label = method_labels.get(method_key, method_key)
                method_results[label] = result

            except Exception as method_err:
                import traceback
                errors.append(
                    f"Method {method_key} failed: {str(method_err)}\n"
                    f"{traceback.format_exc()}"
                )

        if not method_results:
            raise ValueError("All requested methods failed. Check errors.")

        comparison_bar = plot_method_comparison(method_results)
        shift_heatmap = plot_attribution_shift_table(method_results)

        attribution_tables = {}
        for label, result in method_results.items():
            total = sum(result['mean_abs_shap'])
            table = []
            for feat, imp in zip(
                result['feature_names'], result['mean_abs_shap']
            ):
                table.append({
                    'feature': feat,
                    'mean_abs_shap': float(imp),
                    'pct': float(imp / total * 100) if total > 0 else 0.0
                })
            table.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
            for i, row in enumerate(table):
                row['rank'] = i + 1
            attribution_tables[label] = table

        computation_time = time.time() - start_time
        model_info = get_model_info()

        return jsonify({
            'status': 'success',
            'methods_run': list(method_results.keys()),
            'model_info': {
                'auc': model_info.get('auc_test'),
                'n_features': model_info.get('n_features'),
            },
            'profiles_generated': len(profiles),
            'computation_time_seconds': round(computation_time, 2),
            'attribution_tables': attribution_tables,
            'plots': {
                'method_comparison': comparison_bar,
                'attribution_shift': shift_heatmap
            },
            'errors': errors
        })

    except Exception as e:
        import traceback
        errors.append(str(e))
        errors.append(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Comparison analysis failed.',
            'errors': errors
        })
```

### 5d — Add Compare button to `templates/index.html`

Add a fourth button to the mode selector section alongside the existing
Method 2, 3, 4 buttons:

```html
<button id="btn-compare" class="mode-btn" onclick="runAnalysis('compare')">
    Compare All Methods
</button>
```

Update the JavaScript `runAnalysis` function endpoint mapping:

```javascript
const endpoints = {
    'classical':  '/api/run-analysis',
    'causal':     '/api/run-causal-analysis',
    'layerwise':  '/api/run-layerwise-analysis',
    'compare':    '/api/compare-methods'
};
const endpoint = endpoints[mode] || '/api/run-analysis';
```

When the compare endpoint returns successfully, display:
- The `method_comparison` grouped bar chart as the primary plot
- The `attribution_shift` heatmap directly below it
- A collapsible section per method showing its individual attribution table

---

## Verification Checklist

After applying all fixes, verify each item:

- [ ] `method3_causal_shap.py` — `other_indices` line removed from inner loop
- [ ] `method4_layerwise_shap.py` — `from itertools import combinations, product` removed
- [ ] `method4_layerwise_shap.py` — stage_limits uses `min(..., self.num_trees)`
- [ ] `method3_causal_shap.py` — comment says "background values for absent features"
- [ ] `analysis/visualise.py` — `plot_method_comparison` function exists and is callable
- [ ] `analysis/visualise.py` — `plot_attribution_shift_table` function exists and is callable
- [ ] `app.py` — both new functions imported from visualise
- [ ] `app.py` — `/api/compare-methods` route exists
- [ ] `templates/index.html` — Compare button exists and calls `runAnalysis('compare')`
- [ ] `python app.py` starts without import errors
- [ ] `/api/compare-methods` with default body returns status success
- [ ] Comparison bar chart renders with one bar group per feature per method
- [ ] Attribution shift heatmap renders with red/blue diverging colours

---

## What NOT to Change

Do not touch any of the following — they are correct and working:
- `data/preprocess.py`
- `data/processed/metadata.json`
- `models/train_lgbm.py`
- `models/model_info.json`
- `config/causal_dag.yaml`
- `config/feature_metadata.yaml`
- `config/hyperparameters.yaml`
- `explainers/base_explainer.py`
- `explainers/causal_graph.py`
- `profile_generator/generator.py`
- `explainers/method2_classical_shap.py`
- The core logic of method3 and method4 (only the specific lines listed above)
