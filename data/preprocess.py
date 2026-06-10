import os
import json
import yaml
import joblib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_config():
    with open(os.path.join(CONFIG_DIR, "hyperparameters.yaml"), 'r') as f:
        hyperparams = yaml.safe_load(f)
    with open(os.path.join(CONFIG_DIR, "feature_metadata.yaml"), 'r') as f:
        metadata = yaml.safe_load(f)
    return hyperparams, metadata

def main():
    print("Loading configurations...")
    hyperparams, metadata_config = load_config()
    
    print("Fetching UCI Adult Income dataset...")
    # adult dataset from OpenML (ID 1590)
    data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    df = data.frame
    
    print("Cleaning data...")
    # Replace dashes with underscores in column names to match metadata
    df.columns = [col.replace('-', '_') for col in df.columns]
    
    # Drop rows with missing values (NaN or '?' based on OpenML dataset)
    df = df.replace('?', pd.NA)
    df = df.dropna()
    
    # Strip whitespace from categorical string values
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
    target_col = metadata_config['target_column']
    if target_col == 'income':
        target_col = 'class' # OpenML adult target is named 'class'
        
    X = df.drop(columns=[target_col])
    
    # Drop census sampling weight — no causal meaning
    X = X.drop(columns=['fnlwgt'])
    # Drop education string — education_num carries same information
    X = X.drop(columns=['education'])
    
    y = df[target_col]
    
    # Encode target: '>50K' -> 1, '<=50K' -> 0
    y = (y == '>50K').astype(int)
    
    # Select features based on metadata
    cat_features = metadata_config['categorical_features']
    cont_features = metadata_config['continuous_features']
    
    X = X[cat_features + cont_features]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=hyperparams['test_size'], 
        random_state=hyperparams['random_seed'],
        stratify=y if hyperparams['stratify'] else None
    )
    
    print("Encoding and Scaling...")
    # OrdinalEncoder for categoricals
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat = pd.DataFrame(ordinal_encoder.fit_transform(X_train[cat_features]), columns=cat_features, index=X_train.index)
    X_test_cat = pd.DataFrame(ordinal_encoder.transform(X_test[cat_features]), columns=cat_features, index=X_test.index)
    
    # StandardScaler for continuous
    scaler = StandardScaler()
    X_train_cont = pd.DataFrame(scaler.fit_transform(X_train[cont_features]), columns=cont_features, index=X_train.index)
    X_test_cont = pd.DataFrame(scaler.transform(X_test[cont_features]), columns=cont_features, index=X_test.index)
    
    # Combine back
    X_train_processed = pd.concat([X_train_cat, X_train_cont], axis=1)
    X_test_processed = pd.concat([X_test_cat, X_test_cont], axis=1)
    
    # Ensure column order matches exactly the original order of config or concatenation
    feature_names = list(X_train_processed.columns)
    
    print("Saving processed data...")
    X_train_processed.to_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test_processed.to_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train.to_frame(name='target').to_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet"))
    y_test.to_frame(name='target').to_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet"))
    
    print("Saving encoders...")
    encoders = {
        'ordinal_encoder': ordinal_encoder,
        'scaler': scaler
    }
    joblib.dump(encoders, os.path.join(PROCESSED_DIR, "encoders.pkl"))
    
    print("Saving metadata.json...")
    # Find indices for features
    mutable_features = metadata_config.get('mutable_features', [])
    immutable_features = metadata_config.get('immutable_features', [])
    sensitive_features = metadata_config.get('sensitive_features', [])
    
    mutable_indices = [feature_names.index(f) for f in mutable_features if f in feature_names]
    immutable_indices = [feature_names.index(f) for f in immutable_features if f in feature_names]
    sensitive_indices = [feature_names.index(f) for f in sensitive_features if f in feature_names]
    
    metadata = {
        "dataset": metadata_config['dataset'],
        "n_train": len(X_train_processed),
        "n_test": len(X_test_processed),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_names_original": feature_names,
        "mutable_indices": mutable_indices,
        "immutable_indices": immutable_indices,
        "sensitive_indices": sensitive_indices,
        "class_balance": {
            "0": int((y_train == 0).sum()),
            "1": int((y_train == 1).sum())
        },
        "categorical_features": cat_features,
        "continuous_features": cont_features
    }
    
    with open(os.path.join(PROCESSED_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
