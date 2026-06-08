import shap
import numpy as np
import pandas as pd
import json
import os
from .base_explainer import BaseExplainer

class ClassicalSHAPExplainer(BaseExplainer):
    def __init__(self, model, background_data, feature_names, config, model_info_path=None):
        self.model = model
        self.feature_names = feature_names
        
        # Load model info to get actual feature importances (for comparison)
        self.model_importances = None
        if model_info_path and os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                self.model_importances = model_info.get('feature_importances_gain', None)
                
        # Interventional marginal sampling
        self.explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional"
        )
        self.config = config

    def explain(self, profiles: pd.DataFrame) -> dict:
        # profiles must have same columns as training data
        shap_values_obj = self.explainer(profiles)
        
        # TreeExplainer with data returns an Explanation object. 
        # For lightgbm binary classification, values shape might be (n_samples, n_features, 2) or (n_samples, n_features) depending on shap version.
        sv = shap_values_obj.values
        base_value = shap_values_obj.base_values
        
        if len(sv.shape) == 3: # (samples, features, classes)
            sv = sv[:, :, 1]
            if isinstance(base_value, np.ndarray) and len(base_value.shape) == 2:
                base_value = base_value[:, 1]
        
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0] # Usually same for all if background data is passed, or take mean

        # Validation check: SHAP values + base_value should approx equal model prediction (margin/log odds)
        # Note: model.predict_proba gives probability. SHAP gives log odds for TreeExplainer classifier.
        # We need to transform log odds back to proba for comparison, or use predict(raw_score=True)
        # But for additivity check, let's just do it in margin space or skip exact assertion if probability is requested.
        # Since LightGBM with shap outputs margin by default:
        predictions_margin = self.model.predict(profiles, raw_score=True)
        shap_sum = sv.sum(axis=1) + base_value
        max_error = np.abs(predictions_margin - shap_sum).max()
        assert max_error < 0.01, f"SHAP additivity check failed: max error {max_error}"

        mean_abs_shap = np.abs(sv).mean(axis=0)

        result = {
            "shap_values": sv.tolist(),
            "base_value": float(base_value),
            "feature_names": self.feature_names,
            "mean_abs_shap": mean_abs_shap.tolist(),
            "profiles": profiles.values.tolist(),
            "method": self.get_method_name(),
            "model_importances": self.model_importances # Pass actual weights for UI
        }
        
        return result

    def get_summary(self, shap_result: dict) -> list:
        mean_abs = shap_result['mean_abs_shap']
        features = shap_result['feature_names']
        sv = np.array(shap_result['shap_values'])
        
        total_importance = sum(mean_abs)
        
        summary = []
        for i, (feat, imp) in enumerate(zip(features, mean_abs)):
            # Determine direction: average SHAP value for this feature (or average of top 10% values)
            # simpler: mean of SHAP values
            mean_shap = sv[:, i].mean()
            direction = "positive" if mean_shap > 0 else "negative"
            
            summary.append({
                "feature": feat,
                "mean_abs_shap": float(imp),
                "pct": float(imp / total_importance * 100) if total_importance > 0 else 0.0,
                "direction": direction
            })
            
        # Sort by mean absolute SHAP descending
        summary.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        
        # Add rank
        for i, item in enumerate(summary):
            item["rank"] = i + 1
            
        return summary

    def get_method_id(self) -> int:
        return 2

    def get_method_name(self) -> str:
        return "classical_shap"
