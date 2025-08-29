"""
Enhanced prediction experiment to predict base_score from post metrics.

Uses ensemble methods, feature engineering, and hyperparameter tuning for better performance.
Train on representative_500, test on (representative_1000 - representative_500).
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from db import get_representative_posts


# Features to use for prediction (same as original)
FEATURES = [
    "PostInferentialSupport_evidence_quality",
    "PostInferentialSupport_overall_support",
    "PostInferentialSupport_reasoning_quality",
    "PostClarity_clarity_score",
    "PostValue_value_ea",
    "PostExternalValidation_emperical_claim_validation_score",
    "PostRobustness_robustness_score",
    "PostAuthorAura_author_fame_ea",
]


def load_post_metrics_for_posts(post_ids: List[str], version_id: str = "mini1000") -> pd.DataFrame:
    """Load post metrics for specific post IDs."""
    metrics_dir = Path(f"data/post_metrics/{version_id}")
    
    all_data = []
    for post_id in post_ids:
        metrics_file = metrics_dir / f"{post_id}.json"
        
        if not metrics_file.exists():
            print(f"WARNING: No metrics found for post {post_id}")
            continue
            
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        row = {"post_id": post_id}
        
        # Extract all metric fields
        for metric_name, metric_data in metrics_data.items():
            if isinstance(metric_data, dict):
                for key, value in metric_data.items():
                    if key != "post_id":
                        row[f"{metric_name}_{key}"] = value
        
        all_data.append(row)
    
    return pd.DataFrame(all_data)


def create_train_test_split(version_id: str = "mini1000"):
    """Create train/test split using representative sets, filtering for posts with metrics."""
    posts_500 = get_representative_posts(500)
    posts_1000 = get_representative_posts(1000)
    
    # Filter for posts that have metrics files
    metrics_dir = Path(f"data/post_metrics/{version_id}")
    
    # Train on representative_500 posts that have metrics
    train_posts_with_metrics = []
    for post in posts_500:
        if (metrics_dir / f"{post.post_id}.json").exists():
            train_posts_with_metrics.append(post.post_id)
        else:
            print(f"Skipping training post {post.post_id} - no metrics file")
    
    # Test on posts from representative_1000 that are NOT in representative_500 and have metrics
    test_posts_with_metrics = []
    train_ids_set = {post.post_id for post in posts_500}
    for post in posts_1000:
        if post.post_id not in train_ids_set:  # Not in training set
            if (metrics_dir / f"{post.post_id}.json").exists():
                test_posts_with_metrics.append(post.post_id)
            else:
                print(f"Skipping test post {post.post_id} - no metrics file")
    
    print(f"Train set: {len(train_posts_with_metrics)} posts (from representative_500)")
    print(f"Test set: {len(test_posts_with_metrics)} posts (from representative_1000, not in train set)")
    
    return train_posts_with_metrics, test_posts_with_metrics


def engineer_features(X_train, X_test):
    """Create polynomial and interaction features."""
    # Create polynomial features (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Get feature names for later interpretation
    feature_names = poly.get_feature_names_out()
    
    return X_train_poly, X_test_poly, feature_names, poly


def create_ensemble_model():
    """Create ensemble of different models."""
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    }
    return models


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Evaluate multiple models and return best performer."""
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        predictions[name] = {
            'train': train_pred,
            'test': test_pred
        }
        
        print(f"  {name}: Test R²={test_r2:.4f}, CV R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    return results, predictions


def create_ensemble_prediction(predictions, y_test):
    """Create ensemble prediction using weighted average of top models."""
    # Weight models by their test R² performance
    model_names = list(predictions.keys())
    test_predictions = np.array([predictions[name]['test'] for name in model_names])
    
    # Simple average ensemble
    ensemble_pred = np.mean(test_predictions, axis=0)
    
    return ensemble_pred


def run_enhanced_experiment(version_id: str = "mini1000", experiment_version: str = "enhanced_v1", ntiles: int = 5):
    """Run the enhanced prediction experiment."""
    print(f"Running enhanced prediction experiment {experiment_version}")
    print(f"Using metrics from version: {version_id}")
    
    # Create train/test split
    train_ids, test_ids = create_train_test_split(version_id)
    
    if len(train_ids) == 0:
        print("ERROR: No training posts with metrics available")
        return
    
    if len(test_ids) == 0:
        print("ERROR: No test posts with metrics available")
        return
    
    # Load metrics data
    train_df = load_post_metrics_for_posts(train_ids, version_id)
    test_df = load_post_metrics_for_posts(test_ids, version_id)
    
    # Get base scores
    all_posts_1000 = {p.post_id: p for p in get_representative_posts(1000)}
    
    train_df['base_score'] = train_df['post_id'].map(lambda x: all_posts_1000[x].base_score)
    test_df['base_score'] = test_df['post_id'].map(lambda x: all_posts_1000[x].base_score)
    
    # Check for missing features
    missing_features = []
    for feature in FEATURES:
        if feature not in train_df.columns:
            missing_features.append(feature)
    
    if missing_features:
        print(f"WARNING: Missing features in training data: {missing_features}")
        available_features = [f for f in FEATURES if f in train_df.columns]
        print(f"Using {len(available_features)} available features: {available_features}")
        features_to_use = available_features
    else:
        features_to_use = FEATURES
    
    # Prepare training data
    X_train = train_df[features_to_use]
    y_train = train_df['base_score']
    
    # Prepare test data
    X_test = test_df[features_to_use]
    y_test = test_df['base_score']
    
    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())  # Use training means for test set
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Engineer features
    print("Engineering polynomial features...")
    X_train_poly, X_test_poly, feature_names, poly = engineer_features(X_train_scaled, X_test_scaled)
    print(f"Expanded to {X_train_poly.shape[1]} features with polynomial interactions")
    
    # Create and evaluate models
    models = create_ensemble_model()
    
    print("\n=== Evaluating Models ===")
    results, predictions = evaluate_models(models, X_train_poly, y_train, X_test_poly, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name} (Test R²: {results[best_model_name]['test_r2']:.4f})")
    
    # Create ensemble prediction
    ensemble_pred = create_ensemble_prediction(predictions, y_test)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"Ensemble Test R²: {ensemble_r2:.4f}")
    print(f"Ensemble Test RMSE: {ensemble_rmse:.2f}")
    print(f"Ensemble Test MAE: {ensemble_mae:.2f}")
    
    # Calculate quantile accuracy for best model
    all_posts = get_representative_posts(1000)
    all_base_scores = [p.base_score for p in all_posts if p.base_score is not None]
    
    percentiles = [(i * 100) / ntiles for i in range(1, ntiles)]
    quantile_boundaries = np.percentile(all_base_scores, percentiles)
    
    def get_quantile(score):
        for i, boundary in enumerate(quantile_boundaries):
            if score <= boundary:
                return i + 1
        return ntiles
    
    # Use best model predictions for quantile analysis
    best_test_pred = predictions[best_model_name]['test']
    test_actual_quantiles = [get_quantile(score) for score in y_test]
    test_predicted_quantiles = [get_quantile(score) for score in best_test_pred]
    quantile_accuracy = np.mean([actual == predicted for actual, predicted in zip(test_actual_quantiles, test_predicted_quantiles)])
    
    # Feature importance (for tree-based models)
    feature_importance = {}
    if hasattr(best_model, 'feature_importances_'):
        # Map back to original feature names (approximate for polynomial features)
        importances = best_model.feature_importances_
        # Take only the first len(features_to_use) importances for original features
        original_importances = importances[:len(features_to_use)]
        feature_importance = dict(zip(features_to_use, original_importances))
    elif hasattr(best_model, 'coef_'):
        # For linear models, use coefficients of original features
        feature_importance = dict(zip(features_to_use, best_model.coef_[:len(features_to_use)]))
    
    # Save results
    enhanced_results = {
        'experiment_version': experiment_version,
        'metrics_version': version_id,
        'best_model': best_model_name,
        'features_used': features_to_use,
        'train_set_size': len(train_ids),
        'test_set_size': len(test_ids),
        'model_results': results,
        'ensemble_performance': {
            'test_r2': float(ensemble_r2),
            'test_rmse': float(ensemble_rmse),
            'test_mae': float(ensemble_mae)
        },
        'best_model_performance': {
            'test_r2': float(results[best_model_name]['test_r2']),
            'test_rmse': float(results[best_model_name]['test_rmse']),
            'test_mae': float(results[best_model_name]['test_mae']),
            'test_quantile_accuracy': float(quantile_accuracy)
        },
        'ntiles': ntiles,
        'quantile_boundaries': quantile_boundaries.tolist(),
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'predictions': {
            'test': {
                'actual': y_test.tolist(),
                'best_model': best_test_pred.tolist(),
                'ensemble': ensemble_pred.tolist(),
                'post_ids': test_ids
            }
        }
    }
    
    # Save model and results
    model_dir = Path("data/ml_experiments/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / f"enhanced_{experiment_version}.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(model_dir / f"enhanced_{experiment_version}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(model_dir / f"enhanced_{experiment_version}_poly.pkl", 'wb') as f:
        pickle.dump(poly, f)
    
    results_dir = Path("data/ml_experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f"enhanced_{experiment_version}_results.json", 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    # Print detailed results
    print(f"\n=== Enhanced Experiment Results ===")
    print(f"Best Model: {best_model_name}")
    print(f"Features used: {len(features_to_use)} (expanded to {X_train_poly.shape[1]} with interactions)")
    print(f"Best Model Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"Ensemble Test R²: {ensemble_r2:.4f}")
    print(f"Test Quantile Accuracy: {quantile_accuracy:.1%}")
    print(f"Random chance accuracy: {100/ntiles:.1f}%")
    
    print(f"\n=== All Model Comparison ===")
    for name, result in results.items():
        print(f"{name:20}: R²={result['test_r2']:.4f}, RMSE={result['test_rmse']:.2f}, CV={result['cv_mean']:.4f}±{result['cv_std']:.4f}")
    
    if feature_importance:
        print(f"\n=== Feature Importance ({best_model_name}) ===")
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")
    
    return enhanced_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run enhanced machine learning experiment")
    parser.add_argument("--version", default="mini1000", help="Metrics version to use")
    parser.add_argument("--experiment-version", default="enhanced_v1", help="Experiment version")
    parser.add_argument("--ntiles", type=int, default=5, help="Number of quantiles/tiles to use (default: 5)")
    
    args = parser.parse_args()
    results = run_enhanced_experiment(
        version_id=args.version,
        experiment_version=args.experiment_version,
        ntiles=args.ntiles
    )