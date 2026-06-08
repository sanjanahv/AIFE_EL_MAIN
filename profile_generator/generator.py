import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np

class ProfileGenerator:
    def __init__(self, config_path, metadata_path, data_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['profile_generator']
            
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.n_features = self.metadata['n_features']
        self.feature_names = self.metadata['feature_names']
        
        # Load encoders
        self.encoders = joblib.load(os.path.join(data_path, 'processed', 'encoders.pkl'))
        self.ordinal_encoder = self.encoders['ordinal_encoder']
        self.scaler = self.encoders['scaler']
        
        # Load reference data
        self.X_test = pd.read_parquet(os.path.join(data_path, 'processed', 'X_test.parquet'))
        self.X_train = pd.read_parquet(os.path.join(data_path, 'processed', 'X_train.parquet'))
        
    def generate(self, n_profiles=None, use_synthetic=None) -> pd.DataFrame:
        if n_profiles is None:
            n_profiles = self.config['n_real_profiles']
        if use_synthetic is None:
            use_synthetic = self.config['use_synthetic']
            
        if use_synthetic:
            # Generate synthetic profiles
            profiles = self._generate_synthetic(n_profiles, self.X_train)
        else:
            # Sample real profiles
            profiles = self._sample_real(n_profiles)
            
        # CRITICAL dimension safety check
        assert profiles.shape[1] == self.n_features, \
            f"Profile shape mismatch: got {profiles.shape[1]}, expected {self.n_features}"
        assert list(profiles.columns) == self.feature_names, \
            "Feature name mismatch between profiles and model"
            
        return profiles

    def _sample_real(self, n) -> pd.DataFrame:
        return self.X_test.sample(n=n, random_state=42, replace=True).reset_index(drop=True)

    def _generate_synthetic(self, n, reference_df) -> pd.DataFrame:
        # We'll use simple perturbation for synthetic generation if CTGAN fails/not used
        try:
            from ctgan import CTGAN
            print("Using CTGAN for synthetic profile generation...")
            discrete_columns = self.metadata['categorical_features']
            ctgan = CTGAN(epochs=10) # Quick training
            ctgan.fit(reference_df, discrete_columns)
            synthetic_data = ctgan.sample(n)
        except ImportError:
            print("CTGAN not installed. Using Gaussian noise perturbation instead...")
            synthetic_data = reference_df.sample(n=n, replace=True, random_state=42).copy()
            noise_factor = self.config.get('synthetic_noise_factor', 0.05)
            
            # Continuous features: add gaussian noise
            cont_cols = self.metadata['continuous_features']
            for col in cont_cols:
                std = reference_df[col].std()
                noise = np.random.normal(0, std * noise_factor, n)
                synthetic_data[col] += noise
                
            # Categorical features: occasional swap
            cat_cols = self.metadata['categorical_features']
            for col in cat_cols:
                # Swap 5% of categories randomly
                swap_idx = np.random.rand(n) < 0.05
                if swap_idx.any():
                    unique_vals = reference_df[col].unique()
                    synthetic_data.loc[swap_idx, col] = np.random.choice(unique_vals, size=swap_idx.sum())
                    
        return synthetic_data.reset_index(drop=True)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_inv = df.copy()
        
        cat_features = self.metadata['categorical_features']
        cont_features = self.metadata['continuous_features']
        
        # Inverse continuous
        df_inv[cont_features] = self.scaler.inverse_transform(df[cont_features])
        
        # Inverse categorical
        # OrdinalEncoder returns float by default if there are NaN, cast back to int if needed
        # but Inverse transform expects 2D array
        df_inv[cat_features] = self.ordinal_encoder.inverse_transform(df[cat_features])
        
        return df_inv
