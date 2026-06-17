# Detailed Empirical Trace: Unified SHAP + Counterfactuals (Method 7)

This trace demonstrates the operation of the Counterfactual Explainer on a single rejected loan applicant profile (Original Class 0, Proba: 0.2202).

It first computes the Unified SHAP attributions ($\phi_i^{nature}$) to identify the causal impact of each feature, and then uses those magnitudes to drive a greedy counterfactual search across the **mutable** features to find the minimal changes needed to flip the prediction to Class 1 (Approval).

### 1. Unified SHAP Computation (Fix 1, 2, 3 Union)
The algorithm first computes $\phi_i^{nature}$ for all features (mutable and immutable) to understand the structural decision of the model.

```text
Original Profile Probabilities: [0.7798 0.2202]
Original Class: 0

[Step 1] Running Unified SHAP (Union of Fix 1, 2, 3)
Computing unified SHAP (phi_i^nature) for 1 samples across 6 stages...

Unified SHAP Magnitude per feature:
  workclass: 3.9667
  marital_status: 7.9935
  occupation: 4.6471
  relationship: 5.2319
  race: 1.7633
  sex: 3.4014
  native_country: 2.2990
  age: 8.7971
  education_num: 3.8911
  capital_gain: 5.2023
  capital_loss: 2.7300
  hours_per_week: 0.8747
```

### 2. Counterfactual Greedy Search
The algorithm filters out the immutable features (`age`, `sex`, `race`, `native_country`, `relationship`) and ranks the remaining actionable features by their Unified SHAP magnitude.

```text
[Step 2] Counterfactual Greedy Search
Ranked Mutable Features to perturb (highest impact first):
  marital_status (Magnitude: 7.9935)
  capital_gain (Magnitude: 5.2023)
  occupation (Magnitude: 4.6471)
  workclass (Magnitude: 3.9667)
  education_num (Magnitude: 3.8911)
  capital_loss (Magnitude: 2.7300)
  hours_per_week (Magnitude: 0.8747)
```

### 3. Feature Evaluation and Flip Sequence
The engine steps through the ranked mutable features one by one, scanning plausible background values to find the perturbation that maximally increases the probability of Class 1. It applies the change and moves to the next feature until the $0.5$ threshold is crossed.

```text
Evaluating marital_status (Rank 1) - Current Proba: 0.2202
  -> Best flip value: 2.00 (Proba changed by +0.1284)
  -> New Proba: 0.3485

Evaluating capital_gain (Rank 2) - Current Proba: 0.3485
  -> Best flip value: 1.96 (Proba changed by +0.7796)
  -> New Proba: 0.9998
  *** Prediction Flipped to Class 1! ***
```

### Conclusion
By changing just two features—guided by the causal unified SHAP rankings—the model's prediction successfully flips from a firm rejection (22%) to a confident approval (99%). This flip sequence drives the calculation of the **Consistency Score (CSᵢ)**.
