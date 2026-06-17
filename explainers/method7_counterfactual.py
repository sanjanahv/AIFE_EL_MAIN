"""
method7_counterfactual.py

Output 7: Unified SHAP + Counterfactual Explanations

Uses phi_i^nature from Method 6 (Formula 4) to rank mutable features by
attribution magnitude, then greedily perturbs them to find the minimal
set of changes that flips the model's prediction.

Immutable features (age, sex, race, native_country, relationship) are
NEVER changed -- the counterfactual only modifies actionable features.

Reference: SHAP_Annotated_Formulas.docx
    "Every attribution that the counterfactual engine converts into a
     human-readable explanation is produced by phi_i^nature."
"""

import numpy as np
import pandas as pd
import json
import os
import yaml
from .base_explainer import BaseExplainer
from .method6_unified_shap import UnifiedSHAPExplainer


class CounterfactualExplainer(BaseExplainer):
    """
    Output 7 -- Counterfactual explanations guided by phi_i^nature.

    Steps:
    1. Compute unified SHAP values for the target sample
    2. Rank mutable features by |phi_i^nature| (most impactful first)
    3. Greedily perturb mutable features to flip the prediction
    4. Return the counterfactual instance + changes made
    """

    def __init__(self, model, background_data: pd.DataFrame,
                 feature_names: list, config: dict,
                 dag_config_path: str = "config/causal_dag.yaml",
                 feature_metadata_path: str = "config/feature_metadata.yaml",
                 model_info_path: str = None,
                 k_neighbours: int = 20):
        """
        Parameters
        ----------
        model                 : fitted LightGBM model
        background_data       : sample of X_train
        feature_names         : ordered list of feature names
        config                : loaded hyperparameters.yaml dict
        dag_config_path       : path to causal_dag.yaml
        feature_metadata_path : path to feature_metadata.yaml
        model_info_path       : optional path to model_info.json
        k_neighbours          : k for conditional sampling
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.config = config
        self.background_data = background_data

        # Load model importances
        self.model_importances = None
        if model_info_path and os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    self.model_importances = model_info.get(
                        'feature_importances_gain')
            except Exception:
                pass

        # Load mutable/immutable from feature_metadata.yaml
        with open(feature_metadata_path, 'r') as f:
            feat_meta = yaml.safe_load(f)

        self.mutable_features = feat_meta.get('mutable_features', [])
        self.immutable_features = feat_meta.get('immutable_features', [])
        self.continuous_features = feat_meta.get('continuous_features', [])

        # Build mutable/immutable index sets
        self.mutable_indices = [
            feature_names.index(f) for f in self.mutable_features
            if f in feature_names
        ]
        self.immutable_indices = [
            feature_names.index(f) for f in self.immutable_features
            if f in feature_names
        ]

        # Build the Unified SHAP explainer (Method 6) for attributions
        self.unified_explainer = UnifiedSHAPExplainer(
            model, background_data, feature_names, config,
            dag_config_path=dag_config_path,
            model_info_path=model_info_path,
            k_neighbours=k_neighbours,
        )

        # Precompute value distributions from background for perturbation
        bg_arr = background_data.values
        self._feature_values = {}
        for i, feat in enumerate(feature_names):
            unique_vals = np.unique(bg_arr[:, i])
            self._feature_values[i] = unique_vals

        print("CounterfactualExplainer initialised: "
              "%d mutable, %d immutable features." %
              (len(self.mutable_indices), len(self.immutable_indices)))

    # ================================================================== #
    # Counterfactual generation                                           #
    # ================================================================== #

    def _predict_class(self, sample: np.ndarray) -> int:
        """Predict binary class for a single sample."""
        proba = self.model.predict_proba(sample.reshape(1, -1))[0]
        return int(np.argmax(proba))

    def _predict_proba(self, sample: np.ndarray) -> float:
        """Predict probability of class 1."""
        proba = self.model.predict_proba(sample.reshape(1, -1))[0]
        return float(proba[1])

    def _generate_counterfactual(
        self, sample: np.ndarray, shap_values: np.ndarray
    ) -> dict:
        """
        Greedy counterfactual search:
        1. Rank mutable features by |phi_i^nature| descending
        2. For each mutable feature, try all plausible values
        3. Pick the value that moves prediction closest to flip
        4. Stop when prediction flips

        Returns dict with counterfactual details or None if not found.
        """
        original_class = self._predict_class(sample)
        original_proba = self._predict_proba(sample)
        target_class = 1 - original_class

        # Rank mutable features by absolute unified SHAP value (descending)
        mutable_shap = [
            (idx, abs(shap_values[idx])) for idx in self.mutable_indices
        ]
        mutable_shap.sort(key=lambda x: -x[1])

        # Greedy perturbation
        cf_sample = sample.copy()
        changes = []

        for feat_idx, shap_mag in mutable_shap:
            feat_name = self.feature_names[feat_idx]
            original_val = sample[feat_idx]
            candidate_values = self._feature_values[feat_idx]

            # For continuous features, add linspace between min and max
            # to find flip points between observed background values
            feat_name_check = self.feature_names[feat_idx]
            if feat_name_check in self.continuous_features and len(candidate_values) > 1:
                extra = np.linspace(
                    candidate_values.min(),
                    candidate_values.max(),
                    num=20
                )
                candidate_values = np.unique(
                    np.concatenate([candidate_values, extra])
                )

            # Skip if only one unique value
            if len(candidate_values) <= 1:
                continue

            # Try each candidate value, pick the one that best flips
            best_val = None
            best_proba_diff = 0.0

            for val in candidate_values:
                if abs(val - original_val) < 1e-10:
                    continue  # skip same value

                cf_sample[feat_idx] = val
                new_proba = self._predict_proba(cf_sample)

                if target_class == 1:
                    proba_diff = new_proba - original_proba
                else:
                    proba_diff = original_proba - new_proba

                if proba_diff > best_proba_diff:
                    best_proba_diff = proba_diff
                    best_val = val

            # Apply best perturbation (if any improvement found)
            if best_val is not None:
                cf_sample[feat_idx] = best_val
                changes.append({
                    "feature": feat_name,
                    "original": float(original_val),
                    "counterfactual": float(best_val),
                    "shap_magnitude": float(shap_mag),
                })

                # Check if prediction flipped
                new_class = self._predict_class(cf_sample)
                if new_class == target_class:
                    return {
                        "found": True,
                        "original_class": original_class,
                        "counterfactual_class": new_class,
                        "original_proba": original_proba,
                        "counterfactual_proba": self._predict_proba(cf_sample),
                        "changes": changes,
                        "n_changes": len(changes),
                        "counterfactual_sample": cf_sample.tolist(),
                    }
            else:
                # Reset to original if no improvement
                cf_sample[feat_idx] = original_val

        # If we exhaust all mutable features without flipping
        return {
            "found": False,
            "original_class": original_class,
            "counterfactual_class": self._predict_class(cf_sample),
            "original_proba": original_proba,
            "counterfactual_proba": self._predict_proba(cf_sample),
            "changes": changes,
            "n_changes": len(changes),
            "counterfactual_sample": cf_sample.tolist(),
        }

    def _compute_consistency_scores(
        self,
        shap_values: np.ndarray,
        counterfactuals: list
    ) -> dict:
        """
        Compute consistency score CSᵢ for each mutable feature.

        CSᵢ = 1 - rank(Δxᵢ^flip) / rank(|φᵢ^nature|)

        Where:
        - rank(|φᵢ^nature|): rank of feature i by SHAP magnitude
          (1 = highest attribution)
        - rank(Δxᵢ^flip): rank of feature i by how early it appeared
          in counterfactual changes (1 = changed first = smallest flip point)

        Returns dict: feature_name -> CS score (float in [-1, 1])
        High CS (close to 0 or positive) = consistent
        Low CS (close to -1) = contradiction
        """
        # Mean absolute SHAP per mutable feature
        mean_abs_shap = {}
        for idx in self.mutable_indices:
            feat = self.feature_names[idx]
            mean_abs_shap[feat] = float(np.abs(shap_values[:, idx]).mean())

        # Rank features by SHAP magnitude (1 = highest)
        shap_ranked = sorted(
            mean_abs_shap.keys(),
            key=lambda f: mean_abs_shap[f],
            reverse=True
        )
        shap_rank = {f: i + 1 for i, f in enumerate(shap_ranked)}

        # Count how often each feature appears in counterfactual changes
        # and what position it appears at (earlier = smaller flip point)
        feature_positions = {
            self.feature_names[idx]: [] for idx in self.mutable_indices
        }

        for cf in counterfactuals:
            if not cf.get('found', False):
                continue
            for pos, change in enumerate(cf.get('changes', [])):
                feat = change['feature']
                if feat in feature_positions:
                    feature_positions[feat].append(pos + 1)  # 1-indexed

        # Mean position (lower = changed earlier = smaller flip required)
        flip_point_score = {}
        for feat in feature_positions:
            positions = feature_positions[feat]
            if positions:
                flip_point_score[feat] = np.mean(positions)
            else:
                # Feature never changed in any counterfactual = hard flip
                flip_point_score[feat] = float(len(self.mutable_indices) + 1)

        # Rank by flip point (1 = changed earliest = easiest to flip)
        flip_ranked = sorted(
            flip_point_score.keys(),
            key=lambda f: flip_point_score[f]
        )
        flip_rank = {f: i + 1 for i, f in enumerate(flip_ranked)}

        # Compute CS per feature
        n_mutable = len(self.mutable_indices)
        consistency_scores = {}

        for feat in mean_abs_shap:
            sr = shap_rank.get(feat, n_mutable)
            fr = flip_rank.get(feat, n_mutable)
            # CS = 1 - (flip_rank / shap_rank)
            # If both ranks agree: CS close to 0 (neutral) or positive
            # If contradiction: CS negative
            cs = 1.0 - (fr / max(sr, 1))
            consistency_scores[feat] = round(float(cs), 4)

        return {
            'consistency_scores': consistency_scores,
            'shap_ranks': shap_rank,
            'flip_ranks': flip_rank,
            'mean_abs_shap': mean_abs_shap,
            'flip_point_scores': flip_point_score
        }

    # ================================================================== #
    # Main explain method                                                 #
    # ================================================================== #

    def explain(self, profiles: pd.DataFrame) -> dict:
        """
        Compute unified SHAP + counterfactuals for all profiles.
        """
        n_profiles = len(profiles)
        max_samples = min(n_profiles, 20)

        # Step 1: Compute unified SHAP values
        print("Step 1: Computing unified SHAP (phi_i^nature)...")
        unified_result = self.unified_explainer.explain(profiles)
        unified_sv = np.array(unified_result["shap_values"])

        # Step 2: Generate counterfactuals for each sample
        print("Step 2: Generating counterfactuals for %d samples..."
              % max_samples)
        profiles_arr = profiles.values
        sample_indices = np.random.RandomState(42).choice(
            n_profiles, max_samples, replace=False
        )

        counterfactuals = []
        found_count = 0

        for k, idx in enumerate(sample_indices):
            if k % 5 == 0:
                print("  Sample %d/%d..." % (k + 1, max_samples))

            cf = self._generate_counterfactual(
                profiles_arr[idx], unified_sv[k]
            )
            cf["sample_index"] = int(idx)
            counterfactuals.append(cf)

            if cf["found"]:
                found_count += 1

        success_rate = found_count / max_samples * 100
        print("Counterfactual generation complete: %d/%d found (%.1f%%)"
              % (found_count, max_samples, success_rate))

        # Verify immutability
        for cf in counterfactuals:
            for change in cf["changes"]:
                feat = change["feature"]
                if feat in self.immutable_features:
                    raise ValueError(
                        "IMMUTABLE FEATURE '%s' was changed!" % feat
                    )

        # Compute consistency scores
        print("Computing consistency scores (CSᵢ)...")
        cs_result = self._compute_consistency_scores(unified_sv, counterfactuals)
        print("Consistency scores: %s" % cs_result['consistency_scores'])

        return {
            # Unified SHAP values (same as Method 6)
            "shap_values": unified_result["shap_values"],
            "base_value": unified_result["base_value"],
            "feature_names": self.feature_names,
            "mean_abs_shap": unified_result["mean_abs_shap"],
            "profiles": unified_result["profiles"],
            "method": self.get_method_name(),
            "model_importances": self.model_importances,

            # Counterfactual results
            "counterfactuals": counterfactuals,
            "success_rate": success_rate,
            "n_counterfactuals_found": found_count,
            "n_counterfactuals_total": max_samples,
            
            # Consistency Scores
            "consistency_scores": cs_result['consistency_scores'],
            "shap_ranks": cs_result['shap_ranks'],
            "flip_ranks": cs_result['flip_ranks'],
            "flip_point_scores": cs_result['flip_point_scores'],

            # Metadata
            "mutable_features": self.mutable_features,
            "immutable_features": self.immutable_features,
            "causal_graph_info": unified_result["causal_graph_info"],
        }

    # ================================================================== #
    # Summary and metadata                                                #
    # ================================================================== #

    def get_summary(self, shap_result: dict) -> list:
        """Attribution summary from unified SHAP values with CS scores."""
        mean_abs = shap_result['mean_abs_shap']
        features = shap_result['feature_names']
        sv = np.array(shap_result['shap_values'])
        total_importance = sum(mean_abs)
        cs_scores = shap_result.get('consistency_scores', {})

        summary = []
        for i, (feat, imp) in enumerate(zip(features, mean_abs)):
            mean_shap = sv[:, i].mean()
            direction = "positive" if mean_shap > 0 else "negative"
            is_mutable = feat in self.mutable_features
            cs = cs_scores.get(feat, None)

            # Interpret CS score
            if cs is None:
                cs_label = "N/A"
            elif cs > 0.5:
                cs_label = "consistent"
            elif cs > 0.0:
                cs_label = "partial"
            elif cs > -0.5:
                cs_label = "contradiction"
            else:
                cs_label = "strong_contradiction"

            summary.append({
                "feature": feat,
                "mean_abs_shap": float(imp),
                "pct": float(imp / total_importance * 100)
                       if total_importance > 0 else 0.0,
                "direction": direction,
                "mutable": is_mutable,
                "consistency_score": cs,
                "cs_interpretation": cs_label,
            })

        summary.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        for i, item in enumerate(summary):
            item["rank"] = i + 1
        return summary

    def get_method_id(self) -> int:
        return 7

    def get_method_name(self) -> str:
        return "counterfactual_shap"
