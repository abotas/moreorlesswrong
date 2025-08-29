"""
Linear regression experiment to predict base_score from post metrics.

Train on representative_500, test on (representative_1000 - representative_500).
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from db import get_representative_posts


# Features to use for prediction (based on correlation analysis)
FEATURES = [
    # "PostInferentialSupport_evidence_quality",
    "PostInferentialSupport_overall_support",
    # "PostInferentialSupport_reasoning_quality",
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


def run_experiment(version_id: str = "mini1000", experiment_version: str = "v2", ntiles: int = 5):
    """Run the linear regression experiment."""
    print(f"Running linear regression experiment {experiment_version}")
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
    
    # Scale features for linear regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Feature importance (coefficients)
    feature_importance = dict(zip(features_to_use, model.coef_))
    
    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # Calculate quantile accuracy
    # Get quantile boundaries from all posts in the database
    all_posts = get_representative_posts(1000)  # Get large sample for quantile boundaries
    all_base_scores = [p.base_score for p in all_posts if p.base_score is not None]
    
    # Calculate percentiles for n-tiles
    percentiles = [(i * 100) / ntiles for i in range(1, ntiles)]
    quantile_boundaries = np.percentile(all_base_scores, percentiles)
    
    def get_quantile(score):
        """Get n-tile for a base score."""
        for i, boundary in enumerate(quantile_boundaries):
            if score <= boundary:
                return i + 1
        return ntiles  # Highest quantile
    
    # Calculate quantile accuracy for test set
    test_actual_quantiles = [get_quantile(score) for score in y_test]
    test_predicted_quantiles = [get_quantile(score) for score in test_pred]
    quantile_accuracy = np.mean([actual == predicted for actual, predicted in zip(test_actual_quantiles, test_predicted_quantiles)])
    
    # Calculate quantile accuracy for train set  
    train_actual_quantiles = [get_quantile(score) for score in y_train]
    train_predicted_quantiles = [get_quantile(score) for score in train_pred]
    train_quantile_accuracy = np.mean([actual == predicted for actual, predicted in zip(train_actual_quantiles, train_predicted_quantiles)])
    
    
    results = {
        'experiment_version': experiment_version,
        'metrics_version': version_id,
        'model_type': 'linear_regression',
        'features_used': features_to_use,
        'train_set_size': len(train_ids),
        'test_set_size': len(test_ids),
        'train_post_ids': train_ids,
        'test_post_ids': test_ids,
        'performance': {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_quantile_accuracy': float(train_quantile_accuracy),
            'test_quantile_accuracy': float(quantile_accuracy)
        },
        'ntiles': ntiles,
        'quantile_boundaries': quantile_boundaries.tolist(),
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'predictions': {
            'train': {
                'actual': y_train.tolist(),
                'predicted': train_pred.tolist(),
                'post_ids': train_ids
            },
            'test': {
                'actual': y_test.tolist(),
                'predicted': test_pred.tolist(),
                'post_ids': test_ids
            }
        },
        'model_intercept': float(model.intercept_)
    }
    
    # Save model and scaler
    model_dir = Path("data/ml_experiments/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / f"linear_regression_{experiment_version}.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(model_dir / f"linear_regression_{experiment_version}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save results
    results_dir = Path("data/ml_experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f"linear_regression_{experiment_version}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    

    
    print(f"\n=== Predictions ===")
    print("Test set predictions (with quantiles):")
    for i, (actual, pred, post_id) in enumerate(zip(y_test, test_pred, test_ids)):
        error = abs(actual - pred)
        actual_q = get_quantile(actual)
        pred_q = get_quantile(pred)
        correct = "✓" if actual_q == pred_q else "✗"
        print(f"  {post_id}: actual={actual:.1f}(T{actual_q}), predicted={pred:.1f}(T{pred_q}) {correct}, error={error:.1f}")

    # Print results
    print(f"\n=== Experiment Results ===")
    print(f"Model: Linear Regression")
    print(f"Features used: {len(features_to_use)}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Train Quantile Accuracy: {train_quantile_accuracy:.1%}")
    print(f"Test Quantile Accuracy: {quantile_accuracy:.1%}")
    print(f"\n{ntiles}-tile Boundaries: {quantile_boundaries}")
    print(f"Random chance accuracy: {100/ntiles:.1f}%")
    
    print(f"\n=== Feature Importance (Coefficients) ===")
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, coef in sorted_features:
        print(f"{feature}: {coef:.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run machine learning experiment")
    parser.add_argument("--version", default="mini1000", help="Metrics version to use")
    parser.add_argument("--experiment-version", default="v2", help="Experiment version")
    parser.add_argument("--ntiles", type=int, default=5, help="Number of quantiles/tiles to use (default: 5)")
    
    args = parser.parse_args()
    results = run_experiment(
        version_id=args.version,
        experiment_version=args.experiment_version,
        ntiles=args.ntiles
    )