"""Streamlit app for visualizing post metrics from local files."""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
from scipy import stats
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

from db import get_chronological_sample_posts
from models import Post

# Get version ID from command line args if provided
if len(sys.argv) > 1:
    DEFAULT_VERSION_ID = sys.argv[1]
else:
    DEFAULT_VERSION_ID = "post_v1"

st.set_page_config(page_title="EA Forum Metrics Analysis (Local)", layout="wide")

def get_human_readable_name(metric_name: str) -> str:
    """Convert metric column name to human readable format."""
    v3_mapping = {
        'ValueV3_value_score': 'Value (V3)',
        'ReasoningQualityV3_reasoning_quality_score': 'Reasoning Quality (V3)',
        'CooperativenessV3_cooperativeness_score': 'Cooperativeness (V3)',
        'PrecisionV3_precision_score': 'Precision (V3)',
        'EmpiricalEvidenceQualityV3_empirical_evidence_quality_score': 'Empirical Evidence Quality (V3)',
        'RobustnessV3_robustness_score': 'Robustness (V3)',
        'AuthorAuraV3_ea_fame_score': 'Author EA Fame (V3)',
        'ControversyTemperatureV3_controversy_temperature_score': 'Controversy Temperature (V3)',
        'MemeticPotentialV3_memetic_potential_score': 'Memetic Potential (V3)',
        'OverallEpistemicQualityV3_overall_epistemic_quality_score': 'Overall Epistemic Quality (V3)',
        'OverallKarmaPredictorV3_predicted_karma_score': 'Predicted Karma Score (V3)',
    }
    
    v2_mapping = {
        'TruthfulnessV2_truthfulness_score': 'Truthfulness',
        'ValueV2_value_score': 'Value',
        'CooperativenessV2_cooperativeness_score': 'Cooperativeness',
        'CoherenceV2_coherence_score': 'Coherence',
        'ClarityV2_clarity_score': 'Clarity',
        'PrecisionV2_precision_score': 'Precision',
        'HonestyV2_honesty_score': 'Honesty',
        'AuthorAuraV2_ea_fame_score': 'Author EA Fame',
        'ExternalValidationV2_external_validation_score': 'External Validation',
        'RobustnessV2_robustness_score': 'Robustness',
        'ReasoningQualityV2_reasoning_quality_score': 'Reasoning Quality',
        'MemeticPotentialV2_memetic_potential_score': 'Memetic Potential',
        'TitleClickabilityV2_title_clickability_score': 'Title Clickability',
        'ControversyTemperatureV2_controversy_temperature_score': 'Controversy Temperature',
        'EmpiricalEvidenceQualityV2_empirical_evidence_quality_score': 'Empirical Evidence Quality',
    }

    v1_mapping = {
        'PostValue_value_ea': 'Value to EA',
        'PostValue_value_humanity': 'Value to Humanity',
        'PostRobustness_robustness_score': 'Robustness',
        'PostAuthorAura_author_fame_ea': 'Author Fame (EA)',
        'PostAuthorAura_author_fame_humanity': 'Author Fame (Humanity)',
        'PostClarity_clarity_score': 'Clarity',
        'PostNovelty_novelty_ea': 'Novelty (EA)',
        'PostNovelty_novelty_humanity': 'Novelty (Humanity)',
        'PostInferentialSupport_reasoning_quality': 'Reasoning Quality',
        'PostInferentialSupport_evidence_quality': 'Evidence Quality',
        'PostInferentialSupport_overall_support': 'Overall Support',
        'PostEmpiricalClaimExternalValidation_emperical_claim_validation_score': 'Empirical Validation',
        'base_score': 'Post Karma'
    }
    
    # Check v3 first, then v2, then v1
    if metric_name in v3_mapping:
        return v3_mapping[metric_name]
    elif metric_name in v2_mapping:
        return v2_mapping[metric_name]
    elif metric_name in v1_mapping:
        return v1_mapping[metric_name]
    else:
        # Fallback: clean up the name
        return metric_name.replace('_', ' ').title()

# Cache data loading - shorter TTL for auto-refresh
@st.cache_data(ttl=60)  # Cache for 1 minute
def load_metrics_data(version_id: str, v3_only: bool = False) -> pd.DataFrame:
    """Load posts with metrics from local JSON files."""
    # Check both possible locations for data
    metrics_dir = Path(f"data/post_metrics/{version_id}")
    if not metrics_dir.exists():
        # Try parent directory (when running from moreorlesswrong subdirectory)
        metrics_dir = Path(f"../data/post_metrics/{version_id}")
    
    if not metrics_dir.exists():
        return pd.DataFrame()
    
    all_data = []
    v3_files_loaded = 0
    
    for metrics_file in metrics_dir.glob("*.json"):
        post_id = metrics_file.stem
        
        with open(metrics_file, 'r') as f:
            file_content = f.read()
            metrics_data = json.loads(file_content)
        
        # Filter for V3 metrics if requested
        if v3_only and 'V3"' not in file_content:
            continue
        
        if v3_only:
            v3_files_loaded += 1
        
        row = {"post_id": post_id}
        
        # Flatten all metrics
        for metric_name, metric_data in metrics_data.items():
            if isinstance(metric_data, dict):
                for key, value in metric_data.items():
                    if key != "post_id":  # Skip duplicate
                        row[f"{metric_name}_{key}"] = value
        
        all_data.append(row)
    
    if v3_only:
        print(f"Loaded {v3_files_loaded} JSON files with V3 metrics")
    
    df = pd.DataFrame(all_data)
    
    # Get posts metadata
    posts = get_chronological_sample_posts(4, datetime(2024, 1, 1))
    print(len(posts))
    post_scores = {p.post_id: p.base_score for p in posts}
    post_titles = {p.post_id: p.title[:60] + "..." if len(p.title) > 60 else p.title for p in posts}
    post_authors = {p.post_id: p.author_display_name for p in posts}
    
    df["base_score"] = df["post_id"].map(post_scores)
    df["title"] = df["post_id"].map(post_titles) 
    df["author"] = df["post_id"].map(post_authors)
    
    # Filter out posts without base_score
    df = df.dropna(subset=['base_score'])
    
    return df

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between all metrics and base_score."""
    # Get all numeric columns that are not metadata
    metric_columns = [
        col for col in df.columns 
        if col not in ['post_id', 'title', 'author', 'base_score'] 
        and df[col].dtype in ['int64', 'float64']
    ]
    
    correlations = []
    
    for col in metric_columns:
        valid_data = df[[col, 'base_score']].dropna()
        # Count total entries that have this particular metric (regardless of base_score)
        metric_entries = df[col].notna().sum()
        
        if len(valid_data) > 1:
            correlation, p_value = stats.pearsonr(valid_data[col], valid_data['base_score'])
            if not np.isnan(correlation):
                correlations.append({
                    'Metric': get_human_readable_name(col),
                    'Correlation': correlation,
                    'P-value': p_value,
                    'N': metric_entries,  # Total entries with this metric
                    'N_corr': len(valid_data),  # Entries used for correlation (with both metric and base_score)
                    'Significant': '✓' if p_value < 0.05 else '',
                    'metric_col': col  # Keep original column name
                })
    
    corr_df = pd.DataFrame(correlations)
    if not corr_df.empty:
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    
    return corr_df

def calculate_v3_vs_v2_comparison(df: pd.DataFrame, corr_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comparison between V3 and V2 metric correlations."""
    # Mapping of V3 metrics to their V2 counterparts
    v3_to_v2_mapping = {
        'ValueV3_value_score': 'ValueV2_value_score',
        'CooperativenessV3_cooperativeness_score': 'CooperativenessV2_cooperativeness_score', 
        'PrecisionV3_precision_score': 'PrecisionV2_precision_score',
        'AuthorAuraV3_ea_fame_score': 'AuthorAuraV2_ea_fame_score',
        'ReasoningQualityV3_reasoning_quality_score': 'ReasoningQualityV2_reasoning_quality_score',
        'EmpiricalEvidenceQualityV3_empirical_evidence_quality_score': 'EmpiricalEvidenceQualityV2_empirical_evidence_quality_score',
        'ControversyTemperatureV3_controversy_temperature_score': 'ControversyTemperatureV2_controversy_temperature_score'
    }
    
    comparisons = []
    
    for v3_col, v2_col in v3_to_v2_mapping.items():
        # Check if both metrics exist in the data
        if v3_col in df.columns and v2_col in df.columns:
            # Get correlation data from the correlation dataframe
            v3_data = corr_df[corr_df['metric_col'] == v3_col]
            v2_data = corr_df[corr_df['metric_col'] == v2_col]
            
            if not v3_data.empty and not v2_data.empty:
                v3_corr = v3_data.iloc[0]['Correlation']
                v3_n = v3_data.iloc[0]['N_corr'] if 'N_corr' in v3_data.columns else v3_data.iloc[0]['N']
                v2_corr = v2_data.iloc[0]['Correlation']
                v2_n = v2_data.iloc[0]['N_corr'] if 'N_corr' in v2_data.columns else v2_data.iloc[0]['N']
                
                difference = v3_corr - v2_corr
                
                # Extract base metric name (remove version suffix)
                base_name = v3_col.replace('V3_', '').replace('_score', '').replace('_', ' ').title()
                if base_name.endswith(' Fame'):
                    base_name = base_name.replace(' Fame', ' EA Fame')
                
                comparisons.append({
                    'Metric': get_human_readable_name(v3_col).replace(' (V3)', ''),
                    'V2_Correlation': v2_corr,
                    'V2_N': v2_n,
                    'V3_Correlation': v3_corr, 
                    'V3_N': v3_n,
                    'Difference': difference,
                    'Improvement': '↑ Better' if difference > 0.01 else '↓ Worse' if difference < -0.01 else '≈ Similar'
                })
    
    return pd.DataFrame(comparisons)

def aggregate_by_author(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Aggregate metrics by author."""
    # Group by author and calculate mean for numeric columns
    numeric_cols = [col for col in metrics if col in df.columns]
    
    if not numeric_cols:
        return pd.DataFrame()
    
    author_data = df.groupby('author')[numeric_cols].mean()
    
    # Add post count per author
    post_counts = df.groupby('author').size().reset_index(name='post_count')
    author_data = author_data.reset_index().merge(post_counts, on='author')
    
    return author_data

def main():
    st.title("EA Forum Metrics Analysis (Local)")
    st.caption(f"Analyzing V3 metrics from run: {DEFAULT_VERSION_ID}")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    st.sidebar.write(f"**Run ID:** `{DEFAULT_VERSION_ID}`")
    st.sidebar.caption("Run ID is set via CLI argument when launching the app")
    
    # Auto-refresh control
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh (30s)",
        value=True,
        help="Automatically refresh data every 30 seconds to show new metrics as they're computed"
    )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh mechanism using st.rerun with timer
    if auto_refresh:
        import time
        # Initialize session state for refresh timing
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        # Check if 30 seconds have passed
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 30:
            st.session_state.last_refresh = current_time
            st.cache_data.clear()  # Clear cache to get fresh data
            st.rerun()
        
        # Show countdown timer
        seconds_until_refresh = 30 - int(current_time - st.session_state.last_refresh)
        if seconds_until_refresh > 0:
            st.sidebar.caption(f"⏱️ Auto-refresh in {seconds_until_refresh}s")
    
    # Load data - always use V3 only mode
    df = load_metrics_data(DEFAULT_VERSION_ID, v3_only=True)
    
    if df.empty:
        st.warning(f"No V3 metrics found for run ID '{DEFAULT_VERSION_ID}'. Please check the run ID or run the post_metric_pipeline first.")
        st.code("uv run python moreorlesswrong/post_metric_pipeline.py --threads 4 --model gpt-5-mini")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Posts", len(df))
    with col2:
        st.metric("Unique Authors", df['author'].nunique())
    with col3:
        # Count metrics (columns that contain scores)
        score_cols = [col for col in df.columns if 'score' in col.lower()]
        st.metric("Metrics Available", len(score_cols))
    with col4:
        st.metric("Avg Post Karma", f"{df['base_score'].mean():.1f}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Correlations", 
        "👥 Author Leaderboards", 
        "📈 Distributions"
    ])
    
    with tab1:
        st.header("Correlations with Post Karma")
        
        # Calculate correlations
        corr_df = calculate_correlations(df)
        
        if not corr_df.empty:
            # Bar chart of correlations
            fig = px.bar(
                corr_df,
                x='Correlation',
                y='Metric',
                orientation='h',
                color='Correlation',
                color_continuous_scale='RdBu',
                range_color=[-1, 1],
                title="Correlation with Post Karma (Base Score)",
                hover_data={
                    'Correlation': ':.3f',
                    'P-value': ':.4f', 
                    'N': True,
                    'N_corr': True,
                    'Significant': True
                }
            )
            fig.update_layout(height=max(400, len(corr_df) * 30))
            st.plotly_chart(fig, use_container_width=True)
            
            # V3 vs V2 Comparison
            st.subheader("V3 vs V2 Correlation Comparison")
            st.caption("Comparing V3 metrics (with synthesized context) to their V2 counterparts")
            
            comparison_df = calculate_v3_vs_v2_comparison(df, corr_df)
            if not comparison_df.empty:
                # Format the comparison dataframe for display
                display_comparison = comparison_df.copy()
                display_comparison['V2 Correlation'] = display_comparison.apply(
                    lambda row: f"{row['V2_Correlation']:.3f} (N={row['V2_N']})", axis=1
                )
                display_comparison['V3 Correlation'] = display_comparison.apply(
                    lambda row: f"{row['V3_Correlation']:.3f} (N={row['V3_N']})", axis=1
                )
                display_comparison['Difference'] = display_comparison['Difference'].apply(
                    lambda x: f"{x:+.3f}"
                )
                
                # Select columns for display
                final_display = display_comparison[['Metric', 'V2 Correlation', 'V3 Correlation', 'Difference', 'Improvement']]
                
                # Apply conditional formatting for the differences
                def highlight_improvements(row):
                    if '↑ Better' in str(row['Improvement']):
                        return [''] * len(row.index[:-2]) + ['background-color: #d4edda; color: #155724'] + ['background-color: #d4edda; color: #155724']  # Green
                    elif '↓ Worse' in str(row['Improvement']):
                        return [''] * len(row.index[:-2]) + ['background-color: #f8d7da; color: #721c24'] + ['background-color: #f8d7da; color: #721c24']  # Red
                    else:
                        return [''] * len(row.index[:-2]) + ['background-color: #fff3cd; color: #856404'] + ['background-color: #fff3cd; color: #856404']  # Yellow
                
                styled_df = final_display.style.apply(highlight_improvements, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                improvements = sum(1 for x in comparison_df['Difference'] if x > 0.01)
                degradations = sum(1 for x in comparison_df['Difference'] if x < -0.01)
                similar = len(comparison_df) - improvements - degradations
                total_delta = comparison_df['Difference'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🟢 Improvements", improvements)
                with col2:
                    st.metric("🔴 Degradations", degradations)
                with col3:
                    st.metric("🟡 Similar", similar)
                with col4:
                    delta_color = "🟢" if total_delta > 0 else "🔴" if total_delta < 0 else "🟡"
                    st.metric(f"{delta_color} Total Δ (V3-V2)", f"{total_delta:+.3f}")
            else:
                st.info("No V2/V3 comparable metrics found in the data.")
            
            # Correlation table
            st.subheader("Detailed Correlation Data")
            display_df = corr_df[['Metric', 'Correlation', 'P-value', 'N', 'N_corr', 'Significant']].copy()
            display_df['Correlation'] = display_df['Correlation'].round(3)
            display_df['P-value'] = display_df['P-value'].round(4)
            display_df = display_df.rename(columns={
                'N': 'Total N',
                'N_corr': 'Correlation N'
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Top 3 scatter plots
            st.subheader("Top Correlations (Scatter Plots)")
            
            top_metrics = corr_df.nlargest(3, 'Correlation', keep='first')
            
            for _, row in top_metrics.iterrows():
                metric_col = row['metric_col']
                metric_name = row['Metric']
                correlation = row['Correlation']
                
                fig = px.scatter(
                    df,
                    x=metric_col,
                    y='base_score',
                    title=f"{metric_name} vs Post Karma (r={correlation:.3f})",
                    hover_data=['title', 'author'],
                    trendline="ols"
                )
                fig.update_xaxes(title=metric_name)
                fig.update_yaxes(title="Post Karma")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Author Leaderboards")
        
        # Configuration
        col1, col2 = st.columns(2)
        with col1:
            min_posts = st.number_input(
                "Minimum posts per author", 
                min_value=1, 
                value=3,
                key="author_min_posts"
            )
        with col2:
            sort_order = st.selectbox(
                "Show", 
                ["Top", "Bottom"],
                key="author_sort"
            )
        
        # Get top correlating metrics
        corr_df = calculate_correlations(df)
        if not corr_df.empty:
            top_metrics = corr_df.nlargest(5, 'Correlation')['metric_col'].tolist()
        else:
            # Fallback to common metric patterns
            score_cols = [col for col in df.columns if 'score' in col.lower() and col != 'base_score']
            top_metrics = score_cols[:3]
        
        # Always show post karma first
        st.subheader("📊 Post Karma by Author")
        
        author_data = aggregate_by_author(df, ['base_score'])
        author_data = author_data[author_data['post_count'] >= min_posts]
        
        if not author_data.empty:
            if sort_order == "Top":
                display_data = author_data.nlargest(10, 'base_score')
            else:
                display_data = author_data.nsmallest(10, 'base_score')
            
            display_data = display_data[['author', 'base_score']].copy()
            display_data.columns = ['Author', 'Avg Post Karma']
            display_data['Avg Post Karma'] = display_data['Avg Post Karma'].round(1)
            
            st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        # Show top correlating metrics
        for metric in top_metrics[:3]:
            if metric in df.columns:
                st.subheader(f"📈 {get_human_readable_name(metric)} by Author")
                
                author_data = aggregate_by_author(df, [metric])
                author_data = author_data[author_data['post_count'] >= min_posts]
                
                if not author_data.empty:
                    if sort_order == "Top":
                        display_data = author_data.nlargest(10, metric)
                    else:
                        display_data = author_data.nsmallest(10, metric)
                    
                    display_data = display_data[['author', metric]].copy()
                    display_data.columns = ['Author', get_human_readable_name(metric)]
                    display_data[get_human_readable_name(metric)] = display_data[get_human_readable_name(metric)].round(2)
                    
                    st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("Metric Distributions")
        
        # Get all metric columns (those with scores)
        metric_columns = [
            col for col in df.columns 
            if col not in ['post_id', 'title', 'author', 'base_score'] 
            and df[col].dtype in ['int64', 'float64']
        ]
        
        selected_metric = st.selectbox(
            "Select metric",
            metric_columns,
            format_func=get_human_readable_name,
            key="dist_metric"
        )
        
        if selected_metric:
            # Overall distribution
            fig = px.histogram(
                df,
                x=selected_metric,
                nbins=20,
                title=f"Distribution of {get_human_readable_name(selected_metric)}",
                labels={selected_metric: get_human_readable_name(selected_metric)}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            stats = df[selected_metric].describe()
            
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
            with col2:
                st.metric("Std Dev", f"{stats['std']:.2f}")
            with col3:
                st.metric("Min", f"{stats['min']:.0f}")
            with col4:
                st.metric("Max", f"{stats['max']:.0f}")

if __name__ == "__main__":
    main()