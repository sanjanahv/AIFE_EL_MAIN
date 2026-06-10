"""
smoke_test.py — verifies the full pipeline end-to-end before running Flask.
Tests: imports, model loading, Method 2 (classical), Method 3 (causal), Method 4 (layerwise).
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import yaml

print("=" * 60)
print("AIFE SMOKE TEST")
print("=" * 60)

# ── 1. Imports ────────────────────────────────────────────────
print("\n[1] Testing imports...")
from explainers.method2_classical_shap import ClassicalSHAPExplainer
from explainers.method3_causal_shap import CausalSHAPExplainer
from explainers.method4_layerwise_shap import LayerwiseSHAPExplainer
from explainers.causal_graph import CausalGraph
from analysis.visualise import (
    plot_shap_summary, plot_shap_beeswarm, plot_waterfall_single,
    plot_prediction_distribution, plot_shap_vs_weights,
    plot_causal_weights_heatmap, plot_layerwise_depth_heatmap
)
from profile_generator.generator import ProfileGenerator
print("    All imports OK")

# ── 2. Load model ────────────────────────────────────────────
print("\n[2] Loading model...")
model = joblib.load("models/model.pkl")
with open("models/model_info.json") as f:
    info = json.load(f)
print("    AUC      :", info["auc_test"])
print("    Accuracy :", info["accuracy_test"])
print("    F1       :", info["f1_test"])
print("    Features :", info["n_features"], "->", info["feature_names"])

# ── 3. Load data ─────────────────────────────────────────────
print("\n[3] Loading processed data...")
X_train = pd.read_parquet("data/processed/X_train.parquet")
with open("config/hyperparameters.yaml") as f:
    config = yaml.safe_load(f)
background = X_train.sample(n=50, random_state=42)
print("    Background shape:", background.shape)

# ── 4. Generate profiles ──────────────────────────────────────
print("\n[4] Generating profiles...")
gen = ProfileGenerator("config/hyperparameters.yaml",
                       "data/processed/metadata.json", "data")
profiles = gen.generate(n_profiles=20)
print("    Profiles shape:", profiles.shape)
print("    Columns:", list(profiles.columns))

feature_names = gen.feature_names
dag_path = "config/causal_dag.yaml"
model_info_path = "models/model_info.json"

# ── 5. Method 2 — Classical SHAP ─────────────────────────────
print("\n[5] Testing Method 2 — Classical SHAP (20 profiles)...")
m2 = ClassicalSHAPExplainer(model, background, feature_names, config,
                             model_info_path=model_info_path)
r2 = m2.explain(profiles)
print("    shap_values shape :", len(r2["shap_values"]), "x", len(r2["shap_values"][0]))
print("    base_value        :", round(r2["base_value"], 4))
print("    top feature       :", r2["feature_names"][r2["mean_abs_shap"].index(max(r2["mean_abs_shap"]))])
print("    Plots...")
_ = plot_shap_summary(r2)
_ = plot_prediction_distribution(model, profiles, r2)
print("    Method 2 OK")

# ── 6. Method 3 — Causal SHAP ────────────────────────────────
print("\n[6] Testing Method 3 — Causal SHAP (5 profiles only — slow)...")
m3 = CausalSHAPExplainer(model, background, feature_names, config,
                          dag_config_path=dag_path,
                          model_info_path=model_info_path)
profiles_small = profiles.iloc[:5]
r3 = m3.explain(profiles_small)
print("    shap_values shape :", len(r3["shap_values"]), "x", len(r3["shap_values"][0]))
print("    base_value        :", round(r3["base_value"], 4))
print("    causal_graph_info :", r3["causal_graph_info"])
_ = plot_causal_weights_heatmap(m3.causal_graph, feature_names)
print("    Method 3 OK")

# ── 7. Method 4 — Layerwise SHAP ─────────────────────────────
print("\n[7] Testing Method 4 — Layerwise SHAP (10 profiles)...")
m4 = LayerwiseSHAPExplainer(model, background, feature_names, config,
                             model_info_path=model_info_path)
profiles_10 = profiles.iloc[:10]
r4 = m4.explain(profiles_10)
print("    shap_values shape        :", len(r4["shap_values"]), "x", len(r4["shap_values"][0]))
print("    max_depth                :", r4["max_depth"])
print("    n_samples_computed_lw    :", r4["n_samples_computed_layerwise"])
print("    depth keys               :", sorted(r4["layerwise_contributions"].keys()))
_ = plot_layerwise_depth_heatmap(r4)
print("    Method 4 OK")

print("\n" + "=" * 60)
print("ALL TESTS PASSED — pipeline is ready.")
print("Run: python app.py  ->  open http://localhost:5000")
print("=" * 60)
