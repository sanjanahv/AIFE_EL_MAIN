"""
train_lgbm.py

Trains a LightGBM classifier on the preprocessed UCI Adult Income data.
Saves model.pkl and model_info.json to the models/ directory.
"""
import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report
)
from lightgbm import LGBMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    # Load config
    with open(os.path.join(CONFIG_DIR, "hyperparameters.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    lgb_params = config['lightgbm']
    seed = config['random_seed']

    # Load processed data
    print("Loading processed data...")
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "y_train.parquet"))['target']
    y_test = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "y_test.parquet"))['target']

    print("  Train: %d samples, %d features" % (len(X_train), X_train.shape[1]))
    print("  Test : %d samples" % len(X_test))

    # Train LightGBM
    print("Training LightGBM classifier...")
    early_stopping = lgb_params.pop('early_stopping_rounds', 50)

    model = LGBMClassifier(
        random_state=seed,
        verbose=-1,
        **lgb_params
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            __import__('lightgbm').early_stopping(early_stopping, verbose=False),
            __import__('lightgbm').log_evaluation(period=0)
        ]
    )

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("  AUC      : %.4f" % auc)
    print("  Accuracy : %.4f" % acc)
    print("  F1       : %.4f" % f1)

    # Feature importances (gain)
    gain = model.booster_.feature_importance(importance_type='gain')
    gain_normalised = (gain / gain.sum()).tolist()

    feature_names = list(X_train.columns)

    # Save model
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print("Model saved to %s" % model_path)

    # Save model info
    info = {
        "auc_test": round(auc, 4),
        "accuracy_test": round(acc, 4),
        "f1_test": round(f1, 4),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_estimators_actual": model.booster_.num_trees(),
        "feature_importances_gain": {
            feat: round(g, 6) for feat, g in zip(feature_names, gain_normalised)
        }
    }

    info_path = os.path.join(MODELS_DIR, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print("Model info saved to %s" % info_path)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
