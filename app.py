import os
import json
import time
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import yaml

from profile_generator.generator import ProfileGenerator
from explainers.method2_classical_shap import ClassicalSHAPExplainer
from explainers.method3_causal_shap import CausalSHAPExplainer
from explainers.method4_layerwise_shap import LayerwiseSHAPExplainer
from explainers.method5_conditional_shap import ConditionalSHAPExplainer
from explainers.method6_unified_shap import UnifiedSHAPExplainer
from explainers.method7_counterfactual import CounterfactualExplainer
from analysis.visualise import (
    plot_shap_summary,
    plot_shap_beeswarm,
    plot_waterfall_single,
    plot_prediction_distribution,
    plot_shap_vs_weights,
    get_attribution_table,
    plot_causal_weights_heatmap,
    plot_layerwise_depth_heatmap,
    plot_method_comparison,
    plot_attribution_shift_table
)

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def get_model_info():
    info_path = os.path.join(MODELS_DIR, "model_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return None

def get_metadata():
    metadata_path = os.path.join(DATA_DIR, "processed", "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def status():
    model_info = get_model_info()
    metadata = get_metadata()
    
    if model_info and metadata:
        return jsonify({
            "status": "ready",
            "model_info": model_info,
            "metadata": metadata
        })
    else:
        return jsonify({
            "status": "not_ready",
            "message": "Model or data not found. Please run setup script first."
        })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    info = get_model_info()
    if info:
        return jsonify(info)
    return jsonify({"error": "Model info not found"}), 404

@app.route('/api/run-analysis', methods=['POST'])
def run_analysis():
    errors = []
    try:
        req_data = request.json or {}
        n_profiles = int(req_data.get('n_profiles', 200))
        use_synthetic = bool(req_data.get('use_synthetic', False))
        sample_index = int(req_data.get('sample_index_for_waterfall', 0))
        
        start_time = time.time()
        
        # 1. Load Model
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Run setup first.")
        model = joblib.load(model_path)
        
        # 2. Generator
        config_path = os.path.join(CONFIG_DIR, "hyperparameters.yaml")
        metadata_path = os.path.join(DATA_DIR, "processed", "metadata.json")
        generator = ProfileGenerator(config_path, metadata_path, DATA_DIR)
        
        # 3. Generate profiles
        profiles = generator.generate(n_profiles=n_profiles, use_synthetic=use_synthetic)
        
        # 4. Assert profile dimensions (handled in generator, but double check)
        assert profiles.shape[1] == generator.n_features
        
        # 5. Explainer
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Background data for TreeExplainer
        background_samples = config['shap'].get('background_samples', 100)
        X_train = pd.read_parquet(os.path.join(DATA_DIR, "processed", "X_train.parquet"))
        background_data = X_train.sample(n=min(background_samples, len(X_train)), random_state=42)
        
        model_info_path = os.path.join(MODELS_DIR, "model_info.json")
        explainer = ClassicalSHAPExplainer(model, background_data, generator.feature_names, config, model_info_path=model_info_path)
        
        # Run SHAP
        shap_result = explainer.explain(profiles)
        
        # 6. Generate Plots
        summary_bar = plot_shap_summary(shap_result)
        beeswarm = plot_shap_beeswarm(shap_result)
        
        if sample_index >= len(profiles):
            sample_index = 0
            
        waterfall = plot_waterfall_single(shap_result, sample_index)
        pred_dist = plot_prediction_distribution(model, profiles, shap_result)
        shap_vs_weights = plot_shap_vs_weights(shap_result)
        
        attribution_table = explainer.get_summary(shap_result)
        
        computation_time = time.time() - start_time
        
        model_info = get_model_info()
        
        return jsonify({
            "status": "success",
            "model_info": {
                "auc": model_info.get("auc_test"),
                "accuracy": model_info.get("accuracy_test"),
                "n_features": model_info.get("n_features"),
                "feature_names": model_info.get("feature_names")
            },
            "profiles_generated": len(profiles),
            "shap_results": {
                "attribution_table": attribution_table,
                "n_profiles_analysed": len(profiles),
                "base_value": shap_result["base_value"],
                "computation_time_seconds": round(computation_time, 2)
            },
            "plots": {
                "summary_bar": summary_bar,
                "beeswarm": beeswarm,
                "waterfall": waterfall,
                "prediction_distribution": pred_dist,
                "shap_vs_weights": shap_vs_weights
            },
            "errors": errors
        })
        
    except Exception as e:
        errors.append(str(e))
        return jsonify({
            "status": "error",
            "message": "An error occurred during analysis.",
            "errors": errors
        })

@app.route('/api/run-causal-analysis', methods=['POST'])
def run_causal_analysis():
    errors = []
    try:
        req_data = request.json or {}
        n_profiles = int(req_data.get('n_profiles', 100))  # default 100 for causal (slower)
        use_synthetic = bool(req_data.get('use_synthetic', False))
        sample_index = int(req_data.get('sample_index_for_waterfall', 0))

        start_time = time.time()

        # Load model
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Run setup first.")
        model = joblib.load(model_path)

        # Generator
        config_path = os.path.join(CONFIG_DIR, "hyperparameters.yaml")
        metadata_path = os.path.join(DATA_DIR, "processed", "metadata.json")
        generator = ProfileGenerator(config_path, metadata_path, DATA_DIR)

        # Generate profiles
        profiles = generator.generate(n_profiles=n_profiles, use_synthetic=use_synthetic)
        assert profiles.shape[1] == generator.n_features

        # Config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Background data
        background_samples = config['shap'].get('background_samples', 100)
        X_train = pd.read_parquet(os.path.join(DATA_DIR, "processed", "X_train.parquet"))
        background_data = X_train.sample(
            n=min(background_samples, len(X_train)), random_state=42
        )

        # DAG config path
        dag_config_path = os.path.join(CONFIG_DIR, "causal_dag.yaml")
        model_info_path = os.path.join(MODELS_DIR, "model_info.json")

        # Causal SHAP explainer
        explainer = CausalSHAPExplainer(
            model, background_data, generator.feature_names,
            config, dag_config_path, model_info_path=model_info_path
        )

        # Run
        shap_result = explainer.explain(profiles)

        # Plots — reuse same visualise functions
        summary_bar = plot_shap_summary(shap_result)
        beeswarm = plot_shap_beeswarm(shap_result)

        if sample_index >= len(shap_result['shap_values']):
            sample_index = 0

        waterfall = plot_waterfall_single(shap_result, sample_index)
        pred_dist = plot_prediction_distribution(model, profiles, shap_result)
        shap_vs_weights = plot_shap_vs_weights(shap_result)
        
        causal_heatmap = plot_causal_weights_heatmap(
            explainer.causal_graph, generator.feature_names
        )

        attribution_table = explainer.get_summary(shap_result)
        computation_time = time.time() - start_time
        model_info = get_model_info()

        return jsonify({
            "status": "success",
            "method": "causal_shap",
            "model_info": {
                "auc": model_info.get("auc_test"),
                "accuracy": model_info.get("accuracy_test"),
                "n_features": model_info.get("n_features"),
                "feature_names": model_info.get("feature_names")
            },
            "profiles_generated": len(profiles),
            "causal_graph_info": shap_result.get("causal_graph_info", {}),
            "shap_results": {
                "attribution_table": attribution_table,
                "n_profiles_analysed": shap_result.get("n_samples_computed"),
                "base_value": shap_result["base_value"],
                "computation_time_seconds": round(computation_time, 2)
            },
            "plots": {
                "summary_bar": summary_bar,
                "beeswarm": beeswarm,
                "waterfall": waterfall,
                "prediction_distribution": pred_dist,
                "shap_vs_weights": shap_vs_weights,
                "causal_weights_heatmap": causal_heatmap
            },
            "errors": errors
        })

    except Exception as e:
        import traceback
        errors.append(str(e))
        errors.append(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": "An error occurred during causal SHAP analysis.",
            "errors": errors
        })

@app.route('/api/run-layerwise-analysis', methods=['POST'])
def run_layerwise_analysis():
    """Output 4: Layer-wise Local SHAP analysis."""
    errors = []
    try:
        req_data = request.json or {}
        n_profiles = int(req_data.get('n_profiles', 100))
        use_synthetic = bool(req_data.get('use_synthetic', False))
        sample_index = int(req_data.get('sample_index_for_waterfall', 0))

        start_time = time.time()

        # Load model
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Run setup first.")
        model = joblib.load(model_path)

        # Generator
        config_path = os.path.join(CONFIG_DIR, "hyperparameters.yaml")
        metadata_path = os.path.join(DATA_DIR, "processed", "metadata.json")
        generator = ProfileGenerator(config_path, metadata_path, DATA_DIR)

        # Generate profiles
        profiles = generator.generate(n_profiles=n_profiles, use_synthetic=use_synthetic)
        assert profiles.shape[1] == generator.n_features

        # Config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Background data
        background_samples = config['shap'].get('background_samples', 100)
        X_train = pd.read_parquet(os.path.join(DATA_DIR, "processed", "X_train.parquet"))
        background_data = X_train.sample(
            n=min(background_samples, len(X_train)), random_state=42
        )

        model_info_path = os.path.join(MODELS_DIR, "model_info.json")

        # Layer-wise SHAP explainer (Output 4)
        explainer = LayerwiseSHAPExplainer(
            model, background_data, generator.feature_names,
            config, model_info_path=model_info_path
        )

        # Run
        shap_result = explainer.explain(profiles)

        # Plots
        summary_bar = plot_shap_summary(shap_result)
        beeswarm = plot_shap_beeswarm(shap_result)

        if sample_index >= len(shap_result['shap_values']):
            sample_index = 0

        waterfall = plot_waterfall_single(shap_result, sample_index)
        pred_dist = plot_prediction_distribution(model, profiles, shap_result)
        shap_vs_weights = plot_shap_vs_weights(shap_result)
        layerwise_heatmap = plot_layerwise_depth_heatmap(shap_result)

        attribution_table = explainer.get_summary(shap_result)
        computation_time = time.time() - start_time
        model_info = get_model_info()

        return jsonify({
            "status": "success",
            "method": "layerwise_shap",
            "model_info": {
                "auc": model_info.get("auc_test"),
                "accuracy": model_info.get("accuracy_test"),
                "n_features": model_info.get("n_features"),
                "feature_names": model_info.get("feature_names")
            },
            "profiles_generated": len(profiles),
            "layerwise_info": {
                "max_depth": shap_result.get("max_depth"),
                "n_samples_computed_layerwise": shap_result.get("n_samples_computed_layerwise")
            },
            "shap_results": {
                "attribution_table": attribution_table,
                "n_profiles_analysed": len(profiles),
                "base_value": shap_result["base_value"],
                "computation_time_seconds": round(computation_time, 2)
            },
            "layerwise_mean_abs": shap_result.get("layerwise_mean_abs", {}),
            "jacobian_chain": shap_result.get("jacobian_chain", []),
            "plots": {
                "summary_bar": summary_bar,
                "beeswarm": beeswarm,
                "waterfall": waterfall,
                "prediction_distribution": pred_dist,
                "shap_vs_weights": shap_vs_weights,
                "layerwise_depth_heatmap": layerwise_heatmap
            },
            "errors": errors
        })

    except Exception as e:
        import traceback
        errors.append(str(e))
        errors.append(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": "An error occurred during layer-wise SHAP analysis.",
            "errors": errors
        })


@app.route('/api/run-conditional-analysis', methods=['POST'])
def run_conditional_analysis():
    """Output 5: Conditional SHAP (Fix 3 / Formula 3)."""
    errors = []
    try:
        req_data = request.json or {}
        n_profiles = int(req_data.get('n_profiles', 50))
        use_synthetic = bool(req_data.get('use_synthetic', False))

        start_time = time.time()

        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found. Run setup first.')
        model = joblib.load(model_path)

        config_path = os.path.join(CONFIG_DIR, 'hyperparameters.yaml')
        metadata_path = os.path.join(DATA_DIR, 'processed', 'metadata.json')
        generator = ProfileGenerator(config_path, metadata_path, DATA_DIR)
        profiles = generator.generate(n_profiles=n_profiles, use_synthetic=use_synthetic)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        background_samples = config['shap'].get('background_samples', 100)
        X_train = pd.read_parquet(os.path.join(DATA_DIR, 'processed', 'X_train.parquet'))
        background_data = X_train.sample(n=min(background_samples, len(X_train)), random_state=42)
        model_info_path = os.path.join(MODELS_DIR, 'model_info.json')

        explainer = ConditionalSHAPExplainer(
            model, background_data, generator.feature_names, config,
            model_info_path=model_info_path
        )

        shap_result = explainer.explain(profiles)
        summary_bar = plot_shap_summary(shap_result)
        attribution_table = explainer.get_summary(shap_result)
        computation_time = time.time() - start_time
        model_info = get_model_info()

        return jsonify({
            'status': 'success',
            'method': 'conditional_shap',
            'model_info': {
                'auc': model_info.get('auc_test'),
                'n_features': model_info.get('n_features'),
                'feature_names': model_info.get('feature_names')
            },
            'profiles_generated': len(profiles),
            'shap_results': {
                'attribution_table': attribution_table,
                'n_profiles_analysed': shap_result.get('n_samples_computed'),
                'base_value': shap_result['base_value'],
                'computation_time_seconds': round(computation_time, 2)
            },
            'plots': {'summary_bar': summary_bar},
            'errors': errors
        })
    except Exception as e:
        import traceback
        errors.append(str(e))
        errors.append(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'Conditional SHAP analysis failed.', 'errors': errors})


@app.route('/api/run-unified-analysis', methods=['POST'])
def run_unified_analysis():
    """Output 6: Unified SHAP phi_i^nature (Formula 4)."""
    errors = []
    try:
        req_data = request.json or {}
        n_profiles = int(req_data.get('n_profiles', 20))
        use_synthetic = bool(req_data.get('use_synthetic', False))

        start_time = time.time()

        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found. Run setup first.')
        model = joblib.load(model_path)

        config_path = os.path.join(CONFIG_DIR, 'hyperparameters.yaml')
        metadata_path = os.path.join(DATA_DIR, 'processed', 'metadata.json')
        generator = ProfileGenerator(config_path, metadata_path, DATA_DIR)
        profiles = generator.generate(n_profiles=n_profiles, use_synthetic=use_synthetic)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        background_samples = config['shap'].get('background_samples', 100)
        X_train = pd.read_parquet(os.path.join(DATA_DIR, 'processed', 'X_train.parquet'))
        background_data = X_train.sample(n=min(background_samples, len(X_train)), random_state=42)

        dag_config_path = os.path.join(CONFIG_DIR, 'causal_dag.yaml')
        model_info_path = os.path.join(MODELS_DIR, 'model_info.json')

        explainer = UnifiedSHAPExplainer(
            model, background_data, generator.feature_names, config,
            dag_config_path=dag_config_path, model_info_path=model_info_path
        )

        shap_result = explainer.explain(profiles)
        summary_bar = plot_shap_summary(shap_result)
        attribution_table = explainer.get_summary(shap_result)
        computation_time = time.time() - start_time
        model_info = get_model_info()

        return jsonify({
            'status': 'success',
            'method': 'unified_shap',
            'model_info': {
                'auc': model_info.get('auc_test'),
                'n_features': model_info.get('n_features'),
                'feature_names': model_info.get('feature_names')
            },
            'profiles_generated': len(profiles),
            'causal_graph_info': shap_result.get('causal_graph_info', {}),
            'shap_results': {
                'attribution_table': attribution_table,
                'n_profiles_analysed': shap_result.get('n_samples_computed'),
                'base_value': shap_result['base_value'],
                'computation_time_seconds': round(computation_time, 2)
            },
            'unified_info': {
                'n_stages': shap_result.get('n_stages'),
                'jacobian_chain': shap_result.get('jacobian_chain'),
                'causal_neighbourhoods': shap_result.get('causal_neighbourhoods'),
            },
            'plots': {'summary_bar': summary_bar},
            'errors': errors
        })
    except Exception as e:
        import traceback
        errors.append(str(e))
        errors.append(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'Unified SHAP analysis failed.', 'errors': errors})


@app.route('/api/run-counterfactual-analysis', methods=['POST'])
def run_counterfactual_analysis():
    """Output 7: Unified SHAP + Counterfactuals."""
    errors = []
    try:
        req_data = request.json or {}
        n_profiles = int(req_data.get('n_profiles', 10))
        use_synthetic = bool(req_data.get('use_synthetic', False))

        start_time = time.time()

        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found. Run setup first.')
        model = joblib.load(model_path)

        config_path = os.path.join(CONFIG_DIR, 'hyperparameters.yaml')
        metadata_path = os.path.join(DATA_DIR, 'processed', 'metadata.json')
        generator = ProfileGenerator(config_path, metadata_path, DATA_DIR)
        profiles = generator.generate(n_profiles=n_profiles, use_synthetic=use_synthetic)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        background_samples = config['shap'].get('background_samples', 100)
        X_train = pd.read_parquet(os.path.join(DATA_DIR, 'processed', 'X_train.parquet'))
        background_data = X_train.sample(n=min(background_samples, len(X_train)), random_state=42)

        dag_config_path = os.path.join(CONFIG_DIR, 'causal_dag.yaml')
        feature_metadata_path = os.path.join(CONFIG_DIR, 'feature_metadata.yaml')
        model_info_path = os.path.join(MODELS_DIR, 'model_info.json')

        explainer = CounterfactualExplainer(
            model, background_data, generator.feature_names, config,
            dag_config_path=dag_config_path,
            feature_metadata_path=feature_metadata_path,
            model_info_path=model_info_path
        )

        result = explainer.explain(profiles)
        summary_bar = plot_shap_summary(result)
        attribution_table = explainer.get_summary(result)
        computation_time = time.time() - start_time
        model_info = get_model_info()

        return jsonify({
            'status': 'success',
            'method': 'counterfactual_shap',
            'model_info': {
                'auc': model_info.get('auc_test'),
                'n_features': model_info.get('n_features'),
                'feature_names': model_info.get('feature_names')
            },
            'profiles_generated': len(profiles),
            'shap_results': {
                'attribution_table': attribution_table,
                'n_profiles_analysed': result.get('n_counterfactuals_total'),
                'base_value': result['base_value'],
                'computation_time_seconds': round(computation_time, 2)
            },
            'counterfactual_results': {
                'counterfactuals': result['counterfactuals'],
                'success_rate': result['success_rate'],
                'n_found': result['n_counterfactuals_found'],
                'n_total': result['n_counterfactuals_total'],
                'mutable_features': result['mutable_features'],
                'immutable_features': result['immutable_features'],
            },
            'plots': {'summary_bar': summary_bar},
            'errors': errors
        })
    except Exception as e:
        import traceback
        errors.append(str(e))
        errors.append(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'Counterfactual analysis failed.', 'errors': errors})


@app.route('/api/compare-methods', methods=['POST'])
def compare_methods():
    """
    Runs all available SHAP methods and returns a side-by-side
    comparison plot and attribution shift heatmap.

    Request body (all optional):
    {
        "n_profiles": 100,
        "use_synthetic": false,
        "methods": ["classical", "causal", "layerwise", "conditional", "unified"]
    }
    """
    errors = []
    try:
        req_data = request.json or {}
        n_profiles = int(req_data.get('n_profiles', 100))
        use_synthetic = bool(req_data.get('use_synthetic', False))
        requested_methods = req_data.get(
            'methods', ['classical', 'causal', 'layerwise', 'conditional', 'unified']
        )

        start_time = time.time()

        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found. Run setup first.')
        model = joblib.load(model_path)

        config_path = os.path.join(CONFIG_DIR, 'hyperparameters.yaml')
        metadata_path = os.path.join(DATA_DIR, 'processed', 'metadata.json')
        generator = ProfileGenerator(config_path, metadata_path, DATA_DIR)

        profiles = generator.generate(
            n_profiles=n_profiles,
            use_synthetic=use_synthetic
        )
        assert profiles.shape[1] == generator.n_features

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        background_samples = config['shap'].get('background_samples', 100)
        X_train = pd.read_parquet(
            os.path.join(DATA_DIR, 'processed', 'X_train.parquet')
        )
        background_data = X_train.sample(
            n=min(background_samples, len(X_train)), random_state=42
        )

        dag_config_path = os.path.join(CONFIG_DIR, 'causal_dag.yaml')
        model_info_path = os.path.join(MODELS_DIR, 'model_info.json')

        method_results = {}
        method_labels = {
            'classical':   'Classical SHAP',
            'causal':      'Causal SHAP',
            'layerwise':   'Layerwise SHAP',
            'conditional': 'Conditional SHAP',
            'unified':     'Unified SHAP',
        }

        for method_key in requested_methods:
            try:
                if method_key == 'classical':
                    exp = ClassicalSHAPExplainer(
                        model, background_data,
                        generator.feature_names, config,
                        model_info_path=model_info_path
                    )
                    result = exp.explain(profiles)

                elif method_key == 'causal':
                    exp = CausalSHAPExplainer(
                        model, background_data,
                        generator.feature_names, config,
                        dag_config_path,
                        model_info_path=model_info_path
                    )
                    result = exp.explain(profiles)

                elif method_key == 'layerwise':
                    exp = LayerwiseSHAPExplainer(
                        model, background_data,
                        generator.feature_names, config,
                        model_info_path=model_info_path
                    )
                    result = exp.explain(profiles)

                elif method_key == 'conditional':
                    exp = ConditionalSHAPExplainer(
                        model, background_data,
                        generator.feature_names, config,
                        model_info_path=model_info_path
                    )
                    result = exp.explain(profiles)

                elif method_key == 'unified':
                    exp = UnifiedSHAPExplainer(
                        model, background_data,
                        generator.feature_names, config,
                        dag_config_path=dag_config_path,
                        model_info_path=model_info_path
                    )
                    result = exp.explain(profiles)

                else:
                    errors.append(f"Unknown method: {method_key}")
                    continue

                label = method_labels.get(method_key, method_key)
                method_results[label] = result

            except Exception as method_err:
                import traceback
                errors.append(
                    f"Method {method_key} failed: {str(method_err)}\n"
                    f"{traceback.format_exc()}"
                )

        if not method_results:
            raise ValueError("All requested methods failed. Check errors.")

        comparison_bar = plot_method_comparison(method_results)
        shift_heatmap = plot_attribution_shift_table(method_results)

        attribution_tables = {}
        for label, result in method_results.items():
            total = sum(result['mean_abs_shap'])
            table = []
            for feat, imp in zip(
                result['feature_names'], result['mean_abs_shap']
            ):
                table.append({
                    'feature': feat,
                    'mean_abs_shap': float(imp),
                    'pct': float(imp / total * 100) if total > 0 else 0.0
                })
            table.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
            for i, row in enumerate(table):
                row['rank'] = i + 1
            attribution_tables[label] = table

        computation_time = time.time() - start_time
        model_info = get_model_info()

        return jsonify({
            'status': 'success',
            'methods_run': list(method_results.keys()),
            'model_info': {
                'auc': model_info.get('auc_test'),
                'n_features': model_info.get('n_features'),
            },
            'profiles_generated': len(profiles),
            'computation_time_seconds': round(computation_time, 2),
            'attribution_tables': attribution_tables,
            'plots': {
                'method_comparison': comparison_bar,
                'attribution_shift': shift_heatmap
            },
            'errors': errors
        })

    except Exception as e:
        import traceback
        errors.append(str(e))
        errors.append(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Comparison analysis failed.',
            'errors': errors
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
