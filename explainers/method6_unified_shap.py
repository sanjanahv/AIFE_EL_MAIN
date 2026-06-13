"""
method6_unified_shap.py

Output 6: Unified Formula phi_i^nature  (Formula 4 from SHAP_Annotated_Formulas.docx)

Combines all three fixes into one coherent attribution function:
  - Fix 1 (Causal Coalition Weight) from Formula 1
  - Fix 2 (Layerwise Decomposition)  from Formula 2
  - Fix 3 (Conditional Expectations)  from Formula 3

Full formula:

    phi_i^nature = SUM_l {
        SUM_{S in N(i,G)\\{i}}
            w_unified(S,i,G,l) * [ E_{x|x_{S u {i}}}[f_l] - E_{x|x_S}[f_l] ]
    } * J[l]

Where:
    w_unified(S,i,G,l) = [w_causal(S,i,G) * w_layer(S,i,l)]
                        / SUM_{S'} [w_causal(S',i,G) * w_layer(S',i,l)]

    w_causal = "global causal plausibility"  (Formula 1)
    w_layer  = "local structural fairness"   (Formula 2)
    J[l]     = Jacobian chain                (Formula 2, Step 2)

Reference: SHAP_Annotated_Formulas.docx -- Formula 4
"""

import numpy as np
import pandas as pd
import shap
import json
import os
from math import factorial
from itertools import combinations
from .base_explainer import BaseExplainer
from .causal_graph import CausalGraph
from .method5_conditional_shap import ConditionalSampler


class UnifiedSHAPExplainer(BaseExplainer):
    """
    Output 6 -- Unified SHAP (phi_i^nature).

    Two-pass architecture:
      Pass 1: For each stage l, for each feature i, compute causal-weighted
              conditional SHAP restricted to the causal neighbourhood N(i,G).
      Pass 2: Propagate across stages via Jacobian chain J[l].
    """

    def __init__(self, model, background_data: pd.DataFrame,
                 feature_names: list, config: dict,
                 dag_config_path: str = "config/causal_dag.yaml",
                 model_info_path: str = None,
                 k_neighbours: int = 20,
                 max_causal_distance: int = 2):
        """
        Parameters
        ----------
        model               : fitted LightGBM model (LGBMClassifier)
        background_data     : sample of X_train
        feature_names       : ordered list of feature names
        config              : loaded hyperparameters.yaml dict
        dag_config_path     : path to causal_dag.yaml
        model_info_path     : optional path to model_info.json
        k_neighbours        : k for kNN conditional sampling (Fix 3)
        max_causal_distance : max hops in causal graph for neighbourhood
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.config = config
        self.background_data = background_data

        # Load model importances for comparison
        self.model_importances = None
        if model_info_path and os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    self.model_importances = model_info.get(
                        'feature_importances_gain')
            except Exception:
                pass

        # ── Fix 1: Causal Graph ──────────────────────────────────────────
        self.causal_graph = CausalGraph(dag_config_path, feature_names)
        self.max_causal_distance = max_causal_distance

        # Precompute causal neighbourhoods for all features
        self._neighbourhoods = {}
        for feat in feature_names:
            self._neighbourhoods[feat] = (
                self.causal_graph.get_causal_neighbourhood(
                    feat, max_distance=max_causal_distance
                )
            )

        # ── Fix 2: Layerwise decomposition ───────────────────────────────
        self.num_trees = self.model.booster_.num_trees()
        self.n_stages = 6
        self.stage_limits = [
            min(int(np.ceil(self.num_trees * (d + 1) / self.n_stages)),
                self.num_trees)
            for d in range(self.n_stages)
        ]

        # Jacobian chain J[l] = PROD_{k=l+1}^{L} beta_k
        print("UnifiedSHAPExplainer: computing stage Jacobians...")
        self._jacobian_chain = self._compute_jacobian_chain()
        print("  Jacobian chain: %s" %
              [round(j, 4) for j in self._jacobian_chain])

        # ── Fix 3: Conditional sampler ───────────────────────────────────
        self.sampler = ConditionalSampler(background_data, k=k_neighbours)

        # Base value from TreeExplainer
        self.tree_explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional"
        )

        print("UnifiedSHAPExplainer initialised: "
              "%d features, %d stages, k=%d neighbours." %
              (self.n_features, self.n_stages, k_neighbours))

    # ================================================================== #
    # Jacobian chain (same logic as Method 4, re-implemented inline)      #
    # ================================================================== #

    def _stage_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Cumulative raw-score predictions at each stage boundary."""
        preds = np.zeros((len(X), self.n_stages))
        for d, limit in enumerate(self.stage_limits):
            preds[:, d] = self.model.predict(
                X, raw_score=True, num_iteration=limit
            )
        return preds

    def _compute_jacobian_chain(self) -> list:
        """
        beta_k = dot(F_{k-1}, F_k) / dot(F_{k-1}, F_{k-1})
        J[l] = PROD_{k=l+1}^{L} beta_k
        """
        stage_preds = self._stage_predictions(self.background_data)
        betas = []
        for k in range(1, self.n_stages):
            f_prev = stage_preds[:, k - 1]
            f_curr = stage_preds[:, k]
            denom = np.dot(f_prev, f_prev)
            if denom < 1e-12:
                betas.append(1.0)
            else:
                betas.append(float(np.dot(f_prev, f_curr) / denom))

        jacobian_chain = [1.0] * self.n_stages
        for l in range(self.n_stages - 2, -1, -1):
            jacobian_chain[l] = betas[l] * jacobian_chain[l + 1]
        return jacobian_chain

    # ================================================================== #
    # Unified weight computation                                          #
    # ================================================================== #

    @staticmethod
    def _shapley_weight(s_size: int, n_features: int) -> float:
        """Local structural fairness weight (Formula 2 layer weight).

        w(S) = |S|!(|N|-|S|-1)! / |N|!

        When |S| == |N| (coalition is full neighbourhood), weight is 0
        because there's no room for feature i to join.
        """
        if n_features <= 0 or s_size >= n_features:
            return 0.0
        return (factorial(s_size) * factorial(n_features - s_size - 1)
                / factorial(n_features))

    def _compute_unified_weights(self, feature_i: str,
                                 neighbourhood: list) -> dict:
        """
        Compute normalised unified weights for all coalitions S subset
        of neighbourhood (excluding feature_i).

        w_unified(S) = [w_causal(S,i,G) * w_layer(S,i)] / Z

        Where:
          - w_causal = CausalGraph.coalition_weight()
          - w_layer  = |S|!(|N|-|S|-1)! / |N|!  (Shapley weight within N)
          - Z = sum over all S' of (w_causal * w_layer)

        Returns dict: frozenset -> normalised weight
        """
        n_local = len(neighbourhood)  # |N(i,G)| excluding i
        raw_weights = {}

        for s_size in range(n_local + 1):
            w_layer = self._shapley_weight(s_size, n_local)

            for combo in combinations(neighbourhood, s_size):
                coalition = frozenset(combo)
                w_causal = self.causal_graph.coalition_weight(
                    coalition, feature_i
                )
                raw_weights[coalition] = w_causal * w_layer

        # Normalise
        total = sum(raw_weights.values())
        if total < 1e-15:
            uniform = 1.0 / max(len(raw_weights), 1)
            return {k: uniform for k, v in raw_weights.items()}

        return {k: v / total for k, v in raw_weights.items()}

    # ================================================================== #
    # Per-sample, per-stage computation                                   #
    # ================================================================== #

    def _compute_unified_shap_for_sample(
        self, sample: np.ndarray
    ) -> np.ndarray:
        """
        Compute phi_i^nature for a single sample.

        Two-pass:
          Pass 1: For each stage l, for each feature i:
            phi_i^l = SUM_{S in N(i,G)} w_unified(S) *
                      [E_{x|x_{S u {i}}}[f_l] - E_{x|x_S}[f_l]]
          Pass 2: phi_i^nature = SUM_l phi_i^l * J[l]
        """
        n = self.n_features
        feature_indices = list(range(n))

        # Pass 2 accumulator
        phi_nature = np.zeros(n)

        for stage_idx, limit in enumerate(self.stage_limits):
            j_factor = self._jacobian_chain[stage_idx]

            # Pass 1: compute phi_i^l for each feature at this stage
            phi_stage = np.zeros(n)

            for i in range(n):
                feat_i = self.feature_names[i]
                neighbourhood = self._neighbourhoods[feat_i]

                if len(neighbourhood) == 0:
                    # Isolated feature -- fallback to direct marginal
                    continue

                # Get neighbourhood indices
                neigh_indices = [
                    self.feature_names.index(f) for f in neighbourhood
                ]

                # Unified weights for all coalitions within neighbourhood
                unified_weights = self._compute_unified_weights(
                    feat_i, neighbourhood
                )

                phi_i_l = 0.0

                for coalition_set, w in unified_weights.items():
                    if w < 1e-12:
                        continue

                    # Map coalition feature names to indices
                    S_indices = [
                        self.feature_names.index(f) for f in coalition_set
                    ]
                    S_with_i = S_indices + [i]
                    absent_with_i = [
                        j for j in feature_indices if j not in S_with_i
                    ]
                    absent_without_i = [
                        j for j in feature_indices if j not in S_indices
                    ]

                    # E_{x|x_{S u {i}}}[f_l]
                    if len(absent_with_i) == 0:
                        f_with = self.model.predict(
                            sample.reshape(1, -1),
                            raw_score=True, num_iteration=limit
                        ).mean()
                    else:
                        completed_with = self.sampler.sample_conditional(
                            sample, S_with_i, absent_with_i
                        )
                        f_with = self.model.predict(
                            completed_with,
                            raw_score=True, num_iteration=limit
                        ).mean()

                    # E_{x|x_S}[f_l]
                    if len(absent_without_i) == 0:
                        f_without = self.model.predict(
                            sample.reshape(1, -1),
                            raw_score=True, num_iteration=limit
                        ).mean()
                    else:
                        completed_without = self.sampler.sample_conditional(
                            sample, S_indices, absent_without_i
                        )
                        f_without = self.model.predict(
                            completed_without,
                            raw_score=True, num_iteration=limit
                        ).mean()

                    phi_i_l += w * (f_with - f_without)

                phi_stage[i] = phi_i_l

            # Pass 2: weight by Jacobian
            phi_nature += phi_stage * j_factor

        return phi_nature

    # ================================================================== #
    # Main explain method                                                 #
    # ================================================================== #

    def explain(self, profiles: pd.DataFrame) -> dict:
        """
        Compute phi_i^nature for all profiles.

        For efficiency, processes up to 20 profiles.
        """
        profiles_arr = profiles.values
        n_profiles = len(profiles)

        max_samples = min(n_profiles, 20)
        sample_indices = np.random.RandomState(42).choice(
            n_profiles, max_samples, replace=False
        )

        print("Computing unified SHAP (phi_i^nature) for %d samples "
              "across %d stages..." % (max_samples, self.n_stages))

        unified_sv = np.zeros((max_samples, self.n_features))

        for k, idx in enumerate(sample_indices):
            if k % 2 == 0:
                print("  Sample %d/%d..." % (k + 1, max_samples))
            unified_sv[k] = self._compute_unified_shap_for_sample(
                profiles_arr[idx]
            )

        print("Unified SHAP computation complete.")

        # Base value
        classical_result = self.tree_explainer(profiles.iloc[:5])
        base_values = classical_result.base_values
        if isinstance(base_values, np.ndarray):
            if len(base_values.shape) == 2:
                base_value = float(np.mean(base_values[:, 1]))
            else:
                base_value = float(np.mean(base_values))
        else:
            base_value = float(base_values)

        mean_abs_shap = np.abs(unified_sv).mean(axis=0)

        # Neighbourhood info for display
        neigh_info = {}
        for feat in self.feature_names:
            neigh_info[feat] = self._neighbourhoods[feat]

        return {
            "shap_values": unified_sv.tolist(),
            "base_value": base_value,
            "feature_names": self.feature_names,
            "mean_abs_shap": mean_abs_shap.tolist(),
            "profiles": profiles.iloc[sample_indices].values.tolist(),
            "method": self.get_method_name(),
            "model_importances": self.model_importances,
            "n_samples_computed": max_samples,
            "n_stages": self.n_stages,
            "jacobian_chain": self._jacobian_chain,
            "causal_neighbourhoods": neigh_info,
            "causal_graph_info": {
                "n_edges": len(self.causal_graph.edges),
                "protected_attributes": self.causal_graph.protected_attributes,
                "proxy_candidates": self.causal_graph.proxy_candidates,
            },
            "sampling_method": "knn_conditional",
            "k_neighbours": self.sampler.k,
        }

    # ================================================================== #
    # Summary and metadata                                                #
    # ================================================================== #

    def get_summary(self, shap_result: dict) -> list:
        """Standard feature attribution summary."""
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
                "pct": float(imp / total_importance * 100)
                       if total_importance > 0 else 0.0,
                "direction": direction
            })

        summary.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        for i, item in enumerate(summary):
            item["rank"] = i + 1
        return summary

    def get_method_id(self) -> int:
        return 6

    def get_method_name(self) -> str:
        return "unified_shap"
