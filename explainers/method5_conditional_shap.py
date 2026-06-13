"""
method5_conditional_shap.py

Fix 3: Conditional SHAP  (Formula 3 from SHAP_Annotated_Formulas.docx)

Changes from Method 2 (Classical SHAP):
  - Expectation method: E_{x_absent | x_present}[f] replaces E_{x_absent}[f]
  - Everything else identical: uniform coalition weight, full feature set

Classical SHAP samples missing features from their unconditional (marginal)
distribution, ignoring correlations with observed features.  This produces
unrealistic data points (e.g., income=0 with a perfect credit score).

Conditional SHAP fixes this by conditioning on observed feature values when
computing expectations.  The formula (from SHAP_Annotated_Formulas.docx):

    phi_i^cond = SUM_{S subset F\\{i}}  w_uniform(S)
                 * [ E_{x_{F\\(S u {i})} | x_{S u {i}}}[f]
                   - E_{x_{F\\S} | x_S}[f] ]

Where:
  - w_uniform(S) = |S|!(|F|-|S|-1)! / |F|!   (standard Shapley weight)
  - E_{x_absent | x_present}[f]  is estimated via kNN conditional sampling

Natural inspiration: Ecological conditional sampling -- organisms sample
environmental states conditioned on their current context, not uniformly.
"""

import numpy as np
import pandas as pd
import shap
import json
import os
from math import factorial
from itertools import combinations
from .base_explainer import BaseExplainer


# ======================================================================= #
#  ConditionalSampler -- reusable component for Fix 3 and Fix 1+2+3       #
# ======================================================================= #

class ConditionalSampler:
    """
    Estimates P(x_absent | x_present) using k-nearest-neighbour lookup
    on a background dataset.

    For a given set of present features and their observed values, finds
    the k closest rows in the background data (by Euclidean distance on
    present features), then returns those rows' values for the absent
    features.  This ensures that sampled absent-feature values are
    realistic given what is already observed.

    Reference: SHAP_Annotated_Formulas.docx -- Formula 3
        "Conditional SHAP always samples from a distribution conditioned
         on observed coalition features.  Impossible combinations are
         naturally suppressed."
    """

    def __init__(self, background_data: pd.DataFrame, k: int = 20):
        """
        Parameters
        ----------
        background_data : pd.DataFrame
            Reference dataset (typically a sample of X_train) in encoded
            feature space.
        k : int
            Number of nearest neighbours to draw from.
        """
        self.background = background_data.values.astype(np.float64)
        self.columns = list(background_data.columns)
        self.n_bg = len(self.background)
        self.k = min(k, self.n_bg)

        # Precompute per-feature standard deviations for distance scaling
        self._std = np.std(self.background, axis=0)
        self._std[self._std < 1e-12] = 1.0  # avoid division by zero

    def sample_conditional(
        self,
        sample: np.ndarray,
        present_indices: list,
        absent_indices: list,
    ) -> np.ndarray:
        """
        Draw k background rows whose present-feature values are closest
        to the observed sample, then return full rows with present features
        fixed to the sample's values and absent features filled from
        those neighbours.

        Parameters
        ----------
        sample         : 1-D array, shape (n_features,)
        present_indices: list of int -- column indices that are observed
        absent_indices : list of int -- column indices that are missing

        Returns
        -------
        np.ndarray, shape (k, n_features)
            Each row is a plausible completion of the sample.
        """
        if len(present_indices) == 0:
            # No observed features -- return random background rows
            idx = np.random.choice(self.n_bg, size=self.k, replace=True)
            return self.background[idx].copy()

        if len(absent_indices) == 0:
            # All features observed -- tile the sample
            return np.tile(sample, (self.k, 1))

        # Compute scaled Euclidean distance on present features only
        present_vals = sample[present_indices]  # (|S|,)
        bg_present = self.background[:, present_indices]  # (n_bg, |S|)
        scales = self._std[present_indices]

        diff = (bg_present - present_vals) / scales
        dists = np.sum(diff ** 2, axis=1)  # (n_bg,)

        # Find k nearest neighbours
        k_actual = min(self.k, self.n_bg)
        nn_idx = np.argpartition(dists, k_actual)[:k_actual]

        # Build completed rows
        result = np.tile(sample, (k_actual, 1)).astype(np.float64)
        result[:, absent_indices] = self.background[nn_idx][:, absent_indices]
        return result


# ======================================================================= #
#  Coalition weight helpers                                                #
# ======================================================================= #

def _shapley_weight(s_size: int, n_features: int) -> float:
    """
    Standard Shapley coalition weight:
        w(S) = |S|! (|F| - |S| - 1)! / |F|!

    Reference: Formula 0 from SHAP_Annotated_Formulas.docx
    """
    return (factorial(s_size) * factorial(n_features - s_size - 1)
            / factorial(n_features))


# ======================================================================= #
#  Main explainer class                                                    #
# ======================================================================= #

class ConditionalSHAPExplainer(BaseExplainer):
    """
    Output 5 -- Conditional SHAP (Fix 3 / Formula 3).

    Uses standard uniform Shapley weights but replaces marginal sampling
    with kNN-based conditional sampling so that absent features are drawn
    from P(x_absent | x_present) rather than P(x_absent).

    For computational tractability with 12 features (2^11 = 2048 coalitions
    per feature), we enumerate all coalitions but skip those with very small
    Shapley weight to save time.
    """

    def __init__(self, model, background_data: pd.DataFrame,
                 feature_names: list, config: dict,
                 model_info_path: str = None, k_neighbours: int = 20):
        """
        Parameters
        ----------
        model           : fitted LightGBM model (LGBMClassifier)
        background_data : sample of X_train for conditional sampling
        feature_names   : ordered list of feature names
        config          : loaded hyperparameters.yaml dict
        model_info_path : optional path to model_info.json
        k_neighbours    : number of neighbours for conditional sampling
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.config = config
        self.background_data = background_data

        # Load model importances for comparison plot
        self.model_importances = None
        if model_info_path and os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    self.model_importances = model_info.get(
                        'feature_importances_gain')
            except Exception:
                pass

        # Build conditional sampler (reusable by Method 6)
        self.sampler = ConditionalSampler(background_data, k=k_neighbours)

        # Base value via TreeExplainer (same background = same baseline)
        self.tree_explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional"
        )

        print("ConditionalSHAPExplainer initialised with k=%d neighbours."
              % k_neighbours)

    # ------------------------------------------------------------------ #
    # Per-sample computation                                              #
    # ------------------------------------------------------------------ #

    def _compute_conditional_shap_for_sample(
        self, sample: np.ndarray
    ) -> np.ndarray:
        """
        Compute conditional SHAP values for a single sample.

        Formula 3 from SHAP_Annotated_Formulas.docx:

            phi_i^cond = SUM_{S subset F\\{i}} w_uniform(S)
                         * [ E_{x_absent | x_{S u {i}}}[f]
                           - E_{x_absent | x_S}[f] ]

        Parameters
        ----------
        sample : 1-D numpy array, shape (n_features,)

        Returns
        -------
        np.ndarray of shape (n_features,) -- conditional SHAP values
        """
        n = self.n_features
        feature_indices = list(range(n))

        cond_shap_values = np.zeros(n)

        for i in range(n):
            other_indices = [j for j in feature_indices if j != i]

            phi_i = 0.0

            # Enumerate all 2^(n-1) coalitions S subset F\{i}
            for s_size in range(n):  # s_size = 0 .. n-1
                w = _shapley_weight(s_size, n)

                if w < 1e-12:
                    continue

                for combo in combinations(other_indices, s_size):
                    S = list(combo)
                    S_with_i = S + [i]
                    absent_with_i = [j for j in feature_indices
                                     if j not in S_with_i]
                    absent_without_i = [j for j in feature_indices
                                        if j not in S]

                    # E[f | x_{S u {i}}] -- conditional expectation
                    # with feature i in the observed set
                    if len(absent_with_i) == 0:
                        # All features observed
                        f_with_i = self.model.predict(
                            sample.reshape(1, -1), raw_score=True
                        ).mean()
                    else:
                        completed_with = self.sampler.sample_conditional(
                            sample, S_with_i, absent_with_i
                        )
                        f_with_i = self.model.predict(
                            completed_with, raw_score=True
                        ).mean()

                    # E[f | x_S] -- conditional expectation
                    # without feature i in the observed set
                    if len(absent_without_i) == 0:
                        # This only happens when S = F\{i} and we add i
                        # But S doesn't include i, so absent = {i}
                        # Actually this case can't happen since
                        # absent_without_i always includes at least i
                        f_without_i = self.model.predict(
                            sample.reshape(1, -1), raw_score=True
                        ).mean()
                    else:
                        completed_without = self.sampler.sample_conditional(
                            sample, S, absent_without_i
                        )
                        f_without_i = self.model.predict(
                            completed_without, raw_score=True
                        ).mean()

                    phi_i += w * (f_with_i - f_without_i)

            cond_shap_values[i] = phi_i

        return cond_shap_values

    # ------------------------------------------------------------------ #
    # Main explain method                                                 #
    # ------------------------------------------------------------------ #

    def explain(self, profiles: pd.DataFrame) -> dict:
        """
        Compute conditional SHAP values for all profiles.

        For efficiency, computes on up to 50 profiles (subsampled if more).
        """
        profiles_arr = profiles.values
        n_profiles = len(profiles)

        max_samples = min(n_profiles, 50)
        sample_indices = np.random.RandomState(42).choice(
            n_profiles, max_samples, replace=False
        )

        print("Computing conditional SHAP values for %d samples..."
              % max_samples)
        cond_sv = np.zeros((max_samples, self.n_features))

        for k, idx in enumerate(sample_indices):
            if k % 5 == 0:
                print("  Sample %d/%d..." % (k + 1, max_samples))
            cond_sv[k] = self._compute_conditional_shap_for_sample(
                profiles_arr[idx]
            )

        print("Conditional SHAP computation complete.")

        # Base value from TreeExplainer
        classical_result = self.tree_explainer(profiles.iloc[:5])
        base_values = classical_result.base_values
        if isinstance(base_values, np.ndarray):
            if len(base_values.shape) == 2:
                base_value = float(np.mean(base_values[:, 1]))
            else:
                base_value = float(np.mean(base_values))
        else:
            base_value = float(base_values)

        mean_abs_shap = np.abs(cond_sv).mean(axis=0)

        return {
            "shap_values": cond_sv.tolist(),
            "base_value": base_value,
            "feature_names": self.feature_names,
            "mean_abs_shap": mean_abs_shap.tolist(),
            "profiles": profiles.iloc[sample_indices].values.tolist(),
            "method": self.get_method_name(),
            "model_importances": self.model_importances,
            "n_samples_computed": max_samples,
            "sampling_method": "knn_conditional",
            "k_neighbours": self.sampler.k,
        }

    # ------------------------------------------------------------------ #
    # Summary and metadata                                                #
    # ------------------------------------------------------------------ #

    def get_summary(self, shap_result: dict) -> list:
        """Standard feature attribution summary -- same format as M2/M3."""
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
        return 5

    def get_method_name(self) -> str:
        return "conditional_shap"
