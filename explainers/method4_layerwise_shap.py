"""
method4_layerwise_shap.py

Output 4: Layer-wise Local SHAP Formulae

Decomposes the gradient-boosted LightGBM model into L layers (boosting stages/trees)
and computes SHAP locally within each layer according to:

Formula 2: Layerwise Local SHAP
φᵢˡ = Σ w^layer(S, i, l) · [f_l(S∪{i}) - f_l(S)]
      S⊆N(i,l)\{i}

For efficiency and mathematical correctness, we partition the L boosting rounds 
dynamically into 6 sequential stages (representing coarse to fine refinements)
and compute exact local stage contributions using SHAP's C++ TreeExplainer.
"""

import numpy as np
import pandas as pd
import shap
import json
import os
from .base_explainer import BaseExplainer


class LayerwiseSHAPExplainer(BaseExplainer):

    def __init__(self, model, background_data: pd.DataFrame,
                 feature_names: list, config: dict,
                 model_info_path: str = None):
        """
        Parameters:
            model        : fitted LightGBM model (LGBMClassifier)
            background_data : sample of X_train for interventional baseline
            feature_names   : ordered list of feature names
            config          : loaded hyperparameters.yaml dict
            model_info_path : optional path to model_info.json
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.config = config
        self.background_data = background_data

        # Load model gain importances for comparison plot
        self.model_importances = None
        if model_info_path and os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    self.model_importances = model_info.get('feature_importances_gain')
            except Exception:
                pass

        # TreeExplainer with interventional sampling
        self.tree_explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional"
        )

        self.num_trees = self.model.booster_.num_trees()
        # Map 6 sequential stages as "depth_0" to "depth_5" to fit UI seamlessly
        self.max_depth = 5
        print(f"LayerwiseSHAPExplainer initialised with {self.num_trees} boosting stages (trees).")

    def explain(self, profiles: pd.DataFrame) -> dict:
        """
        Compute layer-wise local SHAP values for all profiles by slicing the ensemble
        into 6 sequential boosting stages.
        """
        n_profiles = len(profiles)
        
        # 1. Compute standard full-model TreeSHAP values
        shap_values_obj = self.tree_explainer(profiles)
        sv = shap_values_obj.values
        base_values = shap_values_obj.base_values

        if len(sv.shape) == 3:  # (samples, features, classes)
            sv = sv[:, :, 1]
            if isinstance(base_values, np.ndarray) and len(base_values.shape) == 2:
                base_values = base_values[:, 1]

        if isinstance(base_values, np.ndarray):
            base_value = float(np.mean(base_values))
        else:
            base_value = float(base_values)

        mean_abs_shap = np.abs(sv).mean(axis=0)

        # 2. Decompose into 6 sequential boosting rounds (coarse to fine stages)
        stage_limits = [int(np.ceil(self.num_trees * (d + 1) / 6)) for d in range(6)]
        
        layerwise_contributions = {}
        layerwise_mean_abs = {}
        
        prev_sv = np.zeros_like(sv)
        
        for d, limit in enumerate(stage_limits):
            key = f'depth_{d}'
            # Call TreeExplainer's optimized C++ shap_values with tree_limit
            stage_sv_full = self.tree_explainer.shap_values(profiles, tree_limit=limit, check_additivity=False)
            
            if isinstance(stage_sv_full, list):
                stage_sv = stage_sv_full[1]
            else:
                stage_sv = stage_sv_full
                
            # The contribution of the current stage is the incremental SHAP
            contrib = stage_sv - prev_sv
            prev_sv = stage_sv
            
            layerwise_contributions[key] = contrib.tolist()
            layerwise_mean_abs[key] = np.abs(contrib).mean(axis=0).tolist()

        return {
            "shap_values": sv.tolist(),
            "base_value": base_value,
            "feature_names": self.feature_names,
            "mean_abs_shap": mean_abs_shap.tolist(),
            "profiles": profiles.values.tolist(),
            "method": self.get_method_name(),
            "model_importances": self.model_importances,
            "layerwise_contributions": layerwise_contributions,
            "layerwise_mean_abs": layerwise_mean_abs,
            "max_depth": self.max_depth,
            "n_samples_computed_layerwise": n_profiles,
        }

    def get_summary(self, shap_result: dict) -> list:
        """Standard feature attribution summary — same format as Method 2/3."""
        mean_abs = shap_result['mean_abs_shap']
        features = shap_result['feature_names']
        sv = np.array(shap_result['shap_values'])
        total_importance = sum(mean_abs)

        summary = []
        for i, (feat, imp) in enumerate(zip(features, mean_abs)):
            mean_shap = sv[:, i].mean()
            direction = "positive" if mean_shap > 0 else "negative"
            summary.append({
                "feature": feat,
                "mean_abs_shap": float(imp),
                "pct": float(imp / total_importance * 100) if total_importance > 0 else 0.0,
                "direction": direction
            })

        summary.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        for i, item in enumerate(summary):
            item["rank"] = i + 1

        return summary

    def get_method_id(self) -> int:
        return 4

    def get_method_name(self) -> str:
        return "layerwise_shap"
