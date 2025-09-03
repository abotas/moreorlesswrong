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
    
    # Check v2 first, then v1
    if metric_name in v2_mapping:
        return v2_mapping[metric_name]
    elif metric_name in v1_mapping:
        return v1_mapping[metric_name]
    else:
        # Fallback: clean up the name
        return metric_name.replace('_', ' ').title()

# Cache data loading
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_metrics_data(version_id: str) -> pd.DataFrame:
    """Load posts with metrics from local JSON files."""
    # Check both possible locations for data
    metrics_dir = Path(f"data/post_metrics/{version_id}")
    if not metrics_dir.exists():
        # Try parent directory (when running from moreorlesswrong subdirectory)
        metrics_dir = Path(f"../data/post_metrics/{version_id}")
    
    if not metrics_dir.exists():
        return pd.DataFrame()
    
    all_data = []
    
    for metrics_file in metrics_dir.glob("*.json"):
        post_id = metrics_file.stem
        
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        row = {"post_id": post_id}
        
        # Flatten all metrics
        for metric_name, metric_data in metrics_data.items():
            if isinstance(metric_data, dict):
                for key, value in metric_data.items():
                    if key != "post_id":  # Skip duplicate
                        row[f"{metric_name}_{key}"] = value
        
        all_data.append(row)
    
    df = pd.DataFrame(all_data)
    
    # Get posts metadata
    posts = get_chronological_sample_posts(8, datetime(2024, 1, 1))
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
        
        if len(valid_data) > 1:
            correlation, p_value = stats.pearsonr(valid_data[col], valid_data['base_score'])
            if not np.isnan(correlation):
                correlations.append({
                    'Metric': get_human_readable_name(col),
                    'Correlation': correlation,
                    'P-value': p_value,
                    'Significant': '✓' if p_value < 0.05 else '',
                    'metric_col': col  # Keep original column name
                })
    
    corr_df = pd.DataFrame(correlations)
    if not corr_df.empty:
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    
    return corr_df

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
    st.caption("Analyzing metrics from local JSON files")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    version_id = st.sidebar.text_input(
        "Run ID",
        value=DEFAULT_VERSION_ID,
        help="The run ID of the pipeline run"
    )
    
    # Load data
    df = load_metrics_data(version_id)
    
    if df.empty:
        st.warning(f"No metrics found for run ID '{version_id}'. Please check the run ID or run the post_metric_pipeline first.")
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
                title="Correlation with Post Karma (Base Score)"
            )
            fig.update_layout(height=max(400, len(corr_df) * 30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation table
            display_df = corr_df[['Metric', 'Correlation', 'P-value', 'Significant']].copy()
            display_df['Correlation'] = display_df['Correlation'].round(3)
            display_df['P-value'] = display_df['P-value'].round(4)
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