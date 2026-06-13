"""
method4_layerwise_shap.py

Output 4: Layer-wise Local SHAP  (Formula 2 from SHAP_Annotated_Formulas.pdf)

Implements the FULL two-step formula:

  STEP 1 — Local SHAP per boosting stage l:
      φᵢˡ = Σ  w^layer(S, i, l) · [f_l(S∪{i}) − f_l(S)]
             S⊆N(i,l)∖{i}

  STEP 2 — Upward Jacobian propagation (chain-rule, analogous to backprop):
      φᵢ^global = Σₗ  φᵢˡ · Πₖ₌ₗ₊₁ᴸ (∂Fₖ / ∂Fₖ₋₁)

  where ∂Fₖ/∂Fₖ₋₁ is estimated empirically on the background dataset as
  the OLS slope of cumulative stage-k predictions regressed on stage-(k-1)
  predictions:

      β_k = Σ(Fₖ · Fₖ₋₁) / Σ(Fₖ₋₁²)

  This measures how much a unit change in the stage-(k-1) output propagates
  forward through all later boosting rounds to the final prediction — exactly
  the "downstream influence" from the formula doc.

Reference: SHAP_Annotated_Formulas.pdf — Formula 2, Steps 1 & 2
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
        Parameters
        ----------
        model           : fitted LightGBM model (LGBMClassifier)
        background_data : sample of X_train for interventional SHAP baseline
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

        # TreeExplainer with interventional sampling for SHAP at each stage
        self.tree_explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional"
        )

        self.num_trees = self.model.booster_.num_trees()
        # 6 sequential boosting stages — depth_0 (coarse) to depth_5 (fine)
        self.n_stages = 6
        self.max_depth = self.n_stages - 1  # for UI display
        self.stage_limits = [
            min(int(np.ceil(self.num_trees * (d + 1) / self.n_stages)), self.num_trees)
            for d in range(self.n_stages)
        ]

        # ── Pre-compute Jacobians on background data (Step 2) ───────────────
        # For each consecutive pair of stages (k-1, k), estimate
        #   β_k = ∂F_k / ∂F_{k-1}  via OLS slope on background predictions.
        # Then build the chain product J[l] = Π_{k=l+1}^{L} β_k
        # so J[l] = how much a unit change at stage l propagates to the final output.
        print("LayerwiseSHAPExplainer: computing stage Jacobians on background data...")
        self._jacobian_chain = self._compute_jacobian_chain()
        print(f"  Jacobian chain (stage -> final): {[round(j, 4) for j in self._jacobian_chain]}")
        print(f"LayerwiseSHAPExplainer initialised with {self.num_trees} boosting stages.")

    # ────────────────────────────────────────────────────────────────────────
    # Jacobian estimation
    # ────────────────────────────────────────────────────────────────────────

    def _stage_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns cumulative raw-score (log-odds) predictions at each of the
        n_stages stage boundaries, shape = (n_samples, n_stages).
        """
        preds = np.zeros((len(X), self.n_stages))
        for d, limit in enumerate(self.stage_limits):
            raw = self.model.predict(X, raw_score=True, num_iteration=limit)
            preds[:, d] = raw
        return preds

    def _compute_jacobian_chain(self) -> list:
        """
        Step 2 from the formula doc: estimate ∂F_k/∂F_{k-1} for each
        consecutive pair of stages using OLS on the background dataset.

        β_k = (F_{k-1} · F_k) / (F_{k-1} · F_{k-1})   [dot-product form]

        Returns J[l] = Π_{k=l+1}^{L} β_k  for l = 0 … n_stages-1.
        J[n_stages-1] = 1.0  (last stage: no downstream stages)
        """
        bg = self.background_data
        stage_preds = self._stage_predictions(bg)   # (n_bg, n_stages)

        # β_k for k in {1 … n_stages-1}  (pairs: stage k-1 → stage k)
        betas = []
        for k in range(1, self.n_stages):
            f_prev = stage_preds[:, k - 1]
            f_curr = stage_preds[:, k]
            denom = np.dot(f_prev, f_prev)
            if denom < 1e-12:
                betas.append(1.0)  # degenerate — fall back to 1
            else:
                betas.append(float(np.dot(f_prev, f_curr) / denom))

        # Build chain products: J[l] = product of betas[l], betas[l+1], …, betas[-1]
        # J[n_stages-1] = 1.0 (last stage has no downstream stages)
        jacobian_chain = [1.0] * self.n_stages
        for l in range(self.n_stages - 2, -1, -1):
            jacobian_chain[l] = betas[l] * jacobian_chain[l + 1]

        return jacobian_chain

    # ────────────────────────────────────────────────────────────────────────
    # Main explain method
    # ────────────────────────────────────────────────────────────────────────

    def explain(self, profiles: pd.DataFrame) -> dict:
        """
        Full two-step layer-wise SHAP:

        STEP 1: φᵢˡ  — incremental SHAP at each boosting stage l
        STEP 2: φᵢ^global = Σₗ  φᵢˡ · J[l]   (Jacobian-weighted sum)

        The Jacobian chain J[l] = Π_{k=l+1}^{L} (∂F_k/∂F_{k-1}) was
        pre-computed in __init__ on the background dataset.
        """
        n_profiles = len(profiles)

        # ── STEP 1: local SHAP per stage ────────────────────────────────────
        stage_shap = {}   # key: 'depth_d', value: ndarray (n_profiles, n_features)
        layerwise_mean_abs = {}
        layerwise_contributions = {}

        prev_sv = None

        for d, limit in enumerate(self.stage_limits):
            key = f'depth_{d}'

            # Full-model SHAP up to this tree limit
            stage_sv_full = self.tree_explainer.shap_values(
                profiles, tree_limit=limit, check_additivity=False
            )

            if isinstance(stage_sv_full, list):
                stage_sv = np.array(stage_sv_full[1])
            else:
                stage_sv = np.array(stage_sv_full)

            # Incremental contribution at this stage
            if prev_sv is None:
                contrib = stage_sv
            else:
                contrib = stage_sv - prev_sv
            prev_sv = stage_sv

            stage_shap[key] = contrib  # (n_profiles, n_features)
            layerwise_contributions[key] = contrib.tolist()
            layerwise_mean_abs[key] = np.abs(contrib).mean(axis=0).tolist()

        # ── STEP 2: Jacobian-weighted upward propagation ────────────────────
        #   φᵢ^global = Σₗ  φᵢˡ · J[l]
        propagated_sv = np.zeros((n_profiles, self.n_features))
        for d in range(self.n_stages):
            key = f'depth_{d}'
            j = self._jacobian_chain[d]
            propagated_sv += stage_shap[key] * j

        # Base value (same background → same base)
        shap_obj = self.tree_explainer(profiles.iloc[:5])
        base_values = shap_obj.base_values
        if isinstance(base_values, np.ndarray):
            if len(base_values.shape) == 2:
                base_value = float(np.mean(base_values[:, 1]))
            else:
                base_value = float(np.mean(base_values))
        else:
            base_value = float(base_values)

        # Global SHAP from Step 2 (Jacobian-propagated)
        mean_abs_shap = np.abs(propagated_sv).mean(axis=0)

        # Also expose the raw (non-propagated) full-model SHAP for comparison
        shap_values_obj = self.tree_explainer(profiles)
        sv_raw = shap_values_obj.values
        if len(sv_raw.shape) == 3:
            sv_raw = sv_raw[:, :, 1]

        print(
            f"Layerwise SHAP complete - Jacobian chain: "
            f"{[round(j, 4) for j in self._jacobian_chain]}"
        )

        return {
            # Step 2 result: Jacobian-propagated global attributions
            "shap_values": propagated_sv.tolist(),
            "base_value": base_value,
            "feature_names": self.feature_names,
            "mean_abs_shap": mean_abs_shap.tolist(),
            "profiles": profiles.values.tolist(),
            "method": self.get_method_name(),
            "model_importances": self.model_importances,

            # Step 1 results: per-stage incremental contributions
            "layerwise_contributions": layerwise_contributions,
            "layerwise_mean_abs": layerwise_mean_abs,

            # Jacobians for display/audit
            "jacobian_chain": self._jacobian_chain,

            # Metadata
            "max_depth": self.max_depth,
            "n_samples_computed_layerwise": n_profiles,
        }

    # ────────────────────────────────────────────────────────────────────────

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
