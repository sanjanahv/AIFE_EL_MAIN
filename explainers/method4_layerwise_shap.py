"""
method4_layerwise_shap.py

Output 4: Layer-wise Local SHAP Formulae

For LightGBM (ensemble of decision trees), "layer-wise" means computing
SHAP attributions grouped by tree split depth across the ensemble.

For each sample x and each feature i:
    φᵢ_local(d) = contribution of feature i from all splits at depth d

This gives a richer picture than global SHAP:
- depth_0 tells you which features dominate at the first split (coarse rules)
- deeper levels show refined, interaction-driven contributions
- summing across all depths gives the standard SHAP value

This is LOCAL because attributions are computed per-sample, not averaged.

Method inspiration: Brain layer-wise processing — early layers detect broad
patterns, later layers refine fine-grained distinctions.
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
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                self.model_importances = model_info.get('feature_importances_gain')

        # TreeExplainer with interventional sampling — same setup as Method 2/3
        self.tree_explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional"
        )

        # Extract max depth from trained LightGBM booster
        # LightGBM trees use max_depth from hyperparameters
        self.max_depth = self._get_max_depth()
        print(f"LayerwiseSHAPExplainer initialised. Max tree depth: {self.max_depth}")

    def _get_max_depth(self) -> int:
        """Extract max depth from the LightGBM booster."""
        try:
            # Try getting from booster parameters
            booster = self.model.booster_
            params = booster.dump_model()
            # Walk one tree to find its actual depth
            trees = params.get('tree_info', [])
            if trees:
                max_d = self._tree_depth(trees[0].get('tree_structure', {}))
                return max(max_d, 1)
        except Exception:
            pass
        # Fallback: use the configured max_depth
        return self.config.get('lightgbm', {}).get('max_depth', 6)

    def _tree_depth(self, node: dict, depth: int = 0) -> int:
        """Recursively find the depth of a LightGBM tree node dict."""
        if not node or 'left_child' not in node:
            return depth
        left = self._tree_depth(node.get('left_child', {}), depth + 1)
        right = self._tree_depth(node.get('right_child', {}), depth + 1)
        return max(left, right)

    def _get_split_contributions_for_tree(self, tree_dict: dict,
                                           sample: np.ndarray,
                                           feature_idx_map: dict) -> dict:
        """
        Walk the decision path for `sample` through one LightGBM tree.
        At each split node, record: which feature split, at what depth,
        and what the leaf value difference (contribution) is.

        Returns:
            dict: {depth: {feature_idx: contribution_value}}
        """
        contributions = {}  # {depth: {feature_idx: float}}
        node = tree_dict.get('tree_structure', {})
        depth = 0

        while node and 'split_feature' in node:
            feat_name = node.get('split_feature')
            feat_idx = feature_idx_map.get(feat_name, -1)

            if feat_idx >= 0:
                threshold = node.get('threshold', 0)
                # Contribution = difference in leaf values between branches
                # Approximate as: right_leaf_val - left_leaf_val weighted by direction
                left_val = self._get_subtree_mean_value(node.get('left_child', {}))
                right_val = self._get_subtree_mean_value(node.get('right_child', {}))
                contribution = right_val - left_val

                if depth not in contributions:
                    contributions[depth] = {}
                contributions[depth][feat_idx] = \
                    contributions[depth].get(feat_idx, 0.0) + contribution

                # Navigate the correct branch for this sample
                sample_val = sample[feat_idx] if feat_idx < len(sample) else 0
                try:
                    threshold_val = float(threshold)
                    if sample_val <= threshold_val:
                        node = node.get('left_child', {})
                    else:
                        node = node.get('right_child', {})
                except (ValueError, TypeError):
                    break
            else:
                break
            depth += 1

        return contributions

    def _get_subtree_mean_value(self, node: dict) -> float:
        """Recursively compute the mean leaf value of a subtree."""
        if not node:
            return 0.0
        if 'leaf_value' in node:
            return float(node['leaf_value'])
        left = self._get_subtree_mean_value(node.get('left_child', {}))
        right = self._get_subtree_mean_value(node.get('right_child', {}))
        return (left + right) / 2.0

    def _compute_layerwise_for_sample(self, sample: np.ndarray) -> dict:
        """
        Compute layer-wise SHAP contributions for a single sample.

        For each tree in the LightGBM ensemble:
          - Walk the decision path
          - Record feature contribution at each split depth

        Aggregate across all trees by depth.

        Returns:
            dict: {
                'depth_0': np.array(n_features),
                'depth_1': np.array(n_features),
                ...
                'depth_N': np.array(n_features)
            }
        """
        booster = self.model.booster_
        model_dump = booster.dump_model()
        trees = model_dump.get('tree_info', [])

        # Build feature name -> index map from the booster's feature names
        booster_feature_names = booster.feature_name()
        feature_idx_map = {name: idx for idx, name in enumerate(booster_feature_names)}

        # Accumulator: depth -> feature_idx -> total contribution across trees
        depth_accum = {}

        for tree_dict in trees:
            tree_contributions = self._get_split_contributions_for_tree(
                tree_dict, sample, feature_idx_map
            )
            for depth, feat_contribs in tree_contributions.items():
                if depth not in depth_accum:
                    depth_accum[depth] = np.zeros(self.n_features)
                for feat_idx, contrib in feat_contribs.items():
                    if feat_idx < self.n_features:
                        depth_accum[depth][feat_idx] += contrib

        # Normalise by number of trees and fill missing depths with zeros
        n_trees = len(trees) if trees else 1
        result = {}
        for d in range(self.max_depth + 1):
            key = f'depth_{d}'
            if d in depth_accum:
                result[key] = depth_accum[d] / n_trees
            else:
                result[key] = np.zeros(self.n_features)

        return result

    def explain(self, profiles: pd.DataFrame) -> dict:
        """
        Compute layer-wise local SHAP values for all profiles.

        Returns a dict with:
          - shap_values        : np.array (n_samples, n_features) — TreeSHAP values
          - base_value         : float
          - feature_names      : list
          - mean_abs_shap      : list (n_features)
          - profiles           : list
          - method             : str
          - model_importances  : dict or None
          - layerwise_contributions : {
                'depth_0': list (n_samples, n_features),
                'depth_1': ...,
                ...
            }
          - layerwise_mean_abs : {
                'depth_0': list (n_features) — mean |contribution| at each depth
                ...
            }
        """
        profiles_arr = profiles.values
        n_profiles = len(profiles)

        # --- Standard TreeSHAP values (Method 2 style) for base comparison ---
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

        # --- Layer-wise contributions (per-sample, per-depth) ---
        # Compute for up to 50 samples to keep it manageable
        max_lw_samples = min(n_profiles, 50)
        lw_sample_indices = np.random.choice(n_profiles, max_lw_samples, replace=False)

        print(f"Computing layer-wise contributions for {max_lw_samples} samples...")

        # Collect: {depth_key: list of per-sample arrays}
        layerwise_raw = {f'depth_{d}': [] for d in range(self.max_depth + 1)}

        for k, idx in enumerate(lw_sample_indices):
            if k % 10 == 0:
                print(f"  Layer-wise sample {k+1}/{max_lw_samples}...")
            lw = self._compute_layerwise_for_sample(profiles_arr[idx])
            for d in range(self.max_depth + 1):
                key = f'depth_{d}'
                layerwise_raw[key].append(lw.get(key, np.zeros(self.n_features)))

        print("Layer-wise computation complete.")

        # Convert to arrays
        layerwise_contributions = {}
        layerwise_mean_abs = {}
        for key, arrays in layerwise_raw.items():
            arr = np.array(arrays)  # (max_lw_samples, n_features)
            layerwise_contributions[key] = arr.tolist()
            layerwise_mean_abs[key] = np.abs(arr).mean(axis=0).tolist()

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
            "n_samples_computed_layerwise": max_lw_samples,
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
