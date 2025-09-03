"""
Linear regression experiment to predict post karma from v2 metrics.

Uses time-based train/test split: first 50% of posts by time for training, 
second 50% for testing. Uses active v2 metrics from v2-mini data.
"""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


# V2 features to use (active metrics from pipeline)
V2_FEATURES = [
    "ValueV2_value_score",
    "CooperativenessV2_cooperativeness_score", 
    "ClarityV2_clarity_score",
    "PrecisionV2_precision_score",
    "AuthorAuraV2_ea_fame_score",
    "ExternalValidationV2_external_validation_score",
    "RobustnessV2_robustness_score",
    "ReasoningQualityV2_reasoning_quality_score",
    "TitleClickabilityV2_title_clickability_score",
    "ControversyTemperatureV2_controversy_temperature_score",
    "EmpiricalEvidenceQualityV2_empirical_evidence_quality_score"
]


def load_v2_metrics_from_db() -> pd.DataFrame:
    """Load posts with v2 metrics from database, ordered by time."""
    engine = create_engine(DATABASE_URL)
    
    query = """
    SELECT 
        post_id,
        base_score,
        posted_at,
        -- Extract all v2 metrics from JSONB
        (alej_v2_metrics->>'value_score')::int as value_score,
        (alej_v2_metrics->>'cooperativeness_score')::int as cooperativeness_score,
        (alej_v2_metrics->>'clarity_score')::int as clarity_score,
        (alej_v2_metrics->>'precision_score')::int as precision_score,
        (alej_v2_metrics->>'ea_fame_score')::int as ea_fame_score,
        (alej_v2_metrics->>'external_validation_score')::int as external_validation_score,
        (alej_v2_metrics->>'robustness_score')::int as robustness_score,
        (alej_v2_metrics->>'reasoning_quality_score')::int as reasoning_quality_score,
        (alej_v2_metrics->>'title_clickability_score')::int as title_clickability_score,
        (alej_v2_metrics->>'controversy_temperature_score')::int as controversy_temperature_score,
        (alej_v2_metrics->>'empirical_evidence_quality_score')::int as empirical_evidence_quality_score
    FROM fellowship_mvp
    WHERE alej_v2_metrics IS NOT NULL 
    AND posted_at IS NOT NULL
    ORDER BY posted_at
    """
    
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    
    return df




def time_based_train_test_split(df: pd.DataFrame, test_size: float = 0.5):
    """Split data by time: first (1-test_size)% for train, last test_size% for test."""
    # Sort by time
    df_sorted = df.sort_values('posted_at').reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"Time-based split:")
    print(f"  Training: {len(train_df)} posts ({train_df['posted_at'].min()} to {train_df['posted_at'].max()})")
    print(f"  Testing: {len(test_df)} posts ({test_df['posted_at'].min()} to {test_df['posted_at'].max()})")
    
    return train_df, test_df


def ntile_analysis(y_true, y_pred, n_quantiles=5, dataset_name="Dataset"):
    """Perform n-tile analysis comparing true vs predicted quantiles."""
    # Calculate quantile thresholds based on true values
    quantile_thresholds = np.percentile(y_true, np.linspace(0, 100, n_quantiles + 1))
    
    # Assign true quantiles
    true_quantiles = pd.cut(y_true, bins=quantile_thresholds, labels=False, include_lowest=True, duplicates='drop')
    
    # Assign predicted quantiles using same thresholds
    pred_quantiles = pd.cut(y_pred, bins=quantile_thresholds, labels=False, include_lowest=True, duplicates='drop')
    
    # Handle cases where cut returns NaN (shouldn't happen but just in case)
    true_quantiles = pd.Series(true_quantiles).fillna(0).astype(int)
    pred_quantiles = pd.Series(pred_quantiles).fillna(0).astype(int)
    
    # Calculate accuracy (exact quantile match)
    exact_match_accuracy = accuracy_score(true_quantiles, pred_quantiles)
    
    # Calculate "close" accuracy (within 1 quantile)
    within_one_accuracy = np.mean(np.abs(true_quantiles - pred_quantiles) <= 1)
    
    # Create confusion matrix
    confusion_matrix = pd.crosstab(
        true_quantiles, pred_quantiles,
        rownames=['True Quantile'], colnames=['Predicted Quantile'],
        dropna=False
    ).fillna(0).astype(int)
    
    print(f"\n{dataset_name} - {n_quantiles}-Quantile Analysis:")
    print(f"  Exact quantile accuracy: {exact_match_accuracy:.3f}")
    print(f"  Within-1 quantile accuracy: {within_one_accuracy:.3f}")
    
    print(f"\n  Confusion Matrix ({dataset_name}):")
    print(f"  Rows = True Quantile, Columns = Predicted Quantile")
    print(confusion_matrix)
    
    # Show quantile ranges for interpretation
    print(f"\n  Quantile Ranges ({dataset_name}):")
    for i in range(len(quantile_thresholds) - 1):
        count = np.sum(true_quantiles == i)
        print(f"    Q{i}: [{quantile_thresholds[i]:.1f}, {quantile_thresholds[i+1]:.1f}] ({count} posts)")
    
    return {
        'exact_accuracy': exact_match_accuracy,
        'within_one_accuracy': within_one_accuracy,
        'confusion_matrix': confusion_matrix,
        'quantile_thresholds': quantile_thresholds,
        'true_quantiles': true_quantiles,
        'pred_quantiles': pred_quantiles
    }


def run_experiment():
    """Run the linear regression experiment."""
    print("=" * 60)
    print("V2 Linear Regression Experiment")
    print("=" * 60)
    
    # Load data from database
    print("Loading v2 metrics from database...")
    df = load_v2_metrics_from_db()
    
    print(f"Loaded {len(df)} posts with v2 metrics")
    
    # Prepare features
    feature_columns = [f.split('_', 1)[1] for f in V2_FEATURES]  # Remove prefix for column names
    available_features = [f for f in feature_columns if f in df.columns]
    
    print(f"Available features ({len(available_features)}):")
    for f in available_features:
        print(f"  - {f}")
    
    if len(available_features) == 0:
        print("ERROR: No features available!")
        return
    
    # Remove rows with missing feature data or missing target
    df_clean = df[['post_id', 'posted_at', 'base_score'] + available_features].dropna()
    print(f"After removing missing data: {len(df_clean)} posts")
    
    if len(df_clean) < 20:
        print("ERROR: Not enough clean data for modeling!")
        return
    
    # Time-based train/test split
    train_df, test_df = time_based_train_test_split(df_clean, test_size=0.5)
    
    # Prepare training data
    X_train = train_df[available_features].values
    y_train = train_df['base_score'].values
    
    X_test = test_df[available_features].values
    y_test = test_df['base_score'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear regression
    print("\nTraining linear regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    print(f"Training Performance:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE: {train_mae:.2f}")
    
    print(f"\nTest Performance:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    
    print(f"\nOverfitting Check:")
    print(f"  R² Difference: {train_r2 - test_r2:.4f}")
    if train_r2 - test_r2 > 0.1:
        print("  ⚠️  Possible overfitting (train R² >> test R²)")
    else:
        print("  ✓ Model generalizes reasonably well")
    
    # Feature importance (coefficients)
    print(f"\nFeature Importance (Standardized Coefficients):")
    feature_importance = list(zip(available_features, model.coef_))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, coef in feature_importance:
        print(f"  {feature:35} {coef:8.4f}")
    
    # Summary stats
    print(f"\nTarget Variable (base_score) Stats:")
    print(f"  Train Mean: {y_train.mean():.2f} ± {y_train.std():.2f}")
    print(f"  Test Mean: {y_test.mean():.2f} ± {y_test.std():.2f}")
    print(f"  Overall Range: [{df_clean['base_score'].min():.0f}, {df_clean['base_score'].max():.0f}]")
    
    # Perform n-tile analysis
    print("\n" + "=" * 50)
    print("QUANTILE ANALYSIS")
    print("=" * 50)
    
    # 5-quantile analysis (quintiles)
    train_ntile = ntile_analysis(y_train, y_train_pred, n_quantiles=5, dataset_name="Training")
    test_ntile = ntile_analysis(y_test, y_test_pred, n_quantiles=5, dataset_name="Test")
    
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'feature_importance': feature_importance,
        'n_features': len(available_features),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'train_quantile_accuracy': train_ntile['exact_accuracy'],
        'test_quantile_accuracy': test_ntile['exact_accuracy'],
        'train_quantile_within_one': train_ntile['within_one_accuracy'],
        'test_quantile_within_one': test_ntile['within_one_accuracy']
    }


def main():
    results = run_experiment()
    if results:
        # Calculate random chance baseline
        exact_random_chance = 1/5  # 20% for 5 quantiles
        
        print(f"\n🎯 Test MAE: {results['test_mae']:.2f} karma points")
        print(f"🎯 Test Quantile Accuracy: {results['test_quantile_accuracy']:.3f}")
        print(f"🎯 Random Chance Baseline: {exact_random_chance:.3f}")
        print(f"🎯 Improvement over Random: {results['test_quantile_accuracy']/exact_random_chance:.1f}x")


if __name__ == "__main__":
    main()