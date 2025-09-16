"""Streamlit app for exploring V0 metrics from local JSON files."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from scipy import stats
import numpy as np
import json
from pathlib import Path
import os
from datetime import datetime, date

st.set_page_config(page_title="V0 Metrics Explorer", layout="wide")

# Define the data directories - relative to project root
V0_DATA_DIR = Path(__file__).parent.parent / "data" / "post_metrics" / "v0"
V0_ADJ_DATA_DIR = Path(__file__).parent.parent / "data" / "post_metrics" / "v0-adj"
V3_DATA_DIR = Path(__file__).parent.parent / "data" / "post_metrics" / "v3-clust12"

def load_v0_metrics_from_json(version: str = "v0-adj") -> pd.DataFrame:
    """Load V0 metrics from local JSON files."""

    data_dir = V0_ADJ_DATA_DIR if version == "v0-adj" else V0_DATA_DIR

    if not data_dir.exists():
        st.error(f"Data directory {data_dir} does not exist!")
        return pd.DataFrame()

    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        st.warning(f"No JSON files found in {data_dir}")
        return pd.DataFrame()
    
    data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                metrics = json.load(f)
                
            # Extract post_id from filename
            post_id = json_file.stem
            
            # Create a flattened record
            record = {'post_id': post_id}
            
            # Extract V0 metrics - handle both epistemic and quality versions
            if version == "v0-adj":
                # Use quality metrics for v0-adj
                if 'GptNanoOBOQualityV0' in metrics:
                    record['nano_epistemic_score'] = metrics['GptNanoOBOQualityV0'].get('quality_score')
                    record['nano_explanation'] = metrics['GptNanoOBOQualityV0'].get('explanation')

                if 'GptMiniOBOQualityV0' in metrics:
                    record['mini_epistemic_score'] = metrics['GptMiniOBOQualityV0'].get('quality_score')
                    record['mini_explanation'] = metrics['GptMiniOBOQualityV0'].get('explanation')

                if 'GptFullOBOQualityV0' in metrics:
                    record['full_epistemic_score'] = metrics['GptFullOBOQualityV0'].get('quality_score')
                    record['full_explanation'] = metrics['GptFullOBOQualityV0'].get('explanation')
            else:
                # Use epistemic quality metrics for v0
                if 'GptNanoOBOEpistemicQualityV0' in metrics:
                    record['nano_epistemic_score'] = metrics['GptNanoOBOEpistemicQualityV0'].get('epistemic_quality_score')
                    record['nano_explanation'] = metrics['GptNanoOBOEpistemicQualityV0'].get('explanation')

                if 'GptMiniOBOEpistemicQualityV0' in metrics:
                    record['mini_epistemic_score'] = metrics['GptMiniOBOEpistemicQualityV0'].get('epistemic_quality_score')
                    record['mini_explanation'] = metrics['GptMiniOBOEpistemicQualityV0'].get('explanation')

                if 'GptFullOBOEpistemicQualityV0' in metrics:
                    record['full_epistemic_score'] = metrics['GptFullOBOEpistemicQualityV0'].get('epistemic_quality_score')
                    record['full_explanation'] = metrics['GptFullOBOEpistemicQualityV0'].get('explanation')
            
            # Also extract actual karma if available (might be stored in the JSON)
            if 'base_score' in metrics:
                record['actual_karma'] = metrics['base_score']
            
            # Add post metadata if available
            if 'title' in metrics:
                record['title'] = metrics['title']
            if 'author' in metrics:
                record['author'] = metrics['author']
                
            data.append(record)
            
        except Exception as e:
            st.warning(f"Error reading {json_file.name}: {e}")
            continue
    
    df = pd.DataFrame(data)
    
    # Load additional metadata from database if available
    try:
        from sqlalchemy import create_engine
        from dotenv import load_dotenv
        load_dotenv()
        DATABASE_URL = os.getenv("DATABASE_URL")
        
        if DATABASE_URL:
            engine = create_engine(DATABASE_URL)
            
            # Get post metadata
            metadata_query = f"""
            SELECT 
                post_id,
                title,
                author_display_name as author,
                base_score as actual_karma,
                comment_count,
                word_count,
                posted_at
            FROM fellowship_mvp
            WHERE post_id IN ({','.join([f"'{pid}'" for pid in df['post_id'].tolist()])})
            """
            
            metadata_df = pd.read_sql_query(metadata_query, engine)
            engine.dispose()
            
            # Merge with our data
            df = df.merge(metadata_df, on='post_id', how='left', suffixes=('', '_db'))
            
            # Use database values if our JSON didn't have them
            if 'actual_karma_db' in df.columns:
                df['actual_karma'] = df['actual_karma'].fillna(df['actual_karma_db'])
                df = df.drop('actual_karma_db', axis=1)
            if 'title_db' in df.columns:
                df['title'] = df['title'].fillna(df['title_db'])
                df = df.drop('title_db', axis=1)
            if 'author_db' in df.columns:
                df['author'] = df['author'].fillna(df['author_db'])
                df = df.drop('author_db', axis=1)
                
    except Exception as e:
        st.info(f"Could not load metadata from database: {e}")
    
    return df

def load_v3_metrics_from_json() -> pd.DataFrame:
    """Load V3 overall epistemic quality metrics from local JSON files."""
    
    if not V3_DATA_DIR.exists():
        st.error(f"V3 data directory {V3_DATA_DIR} does not exist!")
        return pd.DataFrame()
    
    json_files = list(V3_DATA_DIR.glob("*.json"))
    
    if not json_files:
        st.warning(f"No JSON files found in {V3_DATA_DIR}")
        return pd.DataFrame()
    
    data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                metrics = json.load(f)
                
            # Extract post_id from filename
            post_id = json_file.stem
            
            # Create a flattened record
            record = {'post_id': post_id}
            
            # Extract V3 overall epistemic quality score
            if 'OverallEpistemicQualityV3' in metrics:
                record['v3_overall_epistemic_score'] = metrics['OverallEpistemicQualityV3'].get('overall_epistemic_quality_score')
            
            # Also extract post metadata if available
            if 'base_score' in metrics:
                record['actual_karma'] = metrics['base_score']
            if 'title' in metrics:
                record['title'] = metrics['title']
            if 'author' in metrics:
                record['author'] = metrics['author']
            if 'posted_at' in metrics:
                record['posted_at'] = metrics['posted_at']
                
            data.append(record)
            
        except Exception as e:
            st.warning(f"Error reading {json_file.name}: {e}")
            continue
    
    return pd.DataFrame(data)

def load_combined_v0_v3_data(version: str = "v0-adj") -> pd.DataFrame:
    """Load and combine V0 and V3 metrics for comparison."""
    v0_df = load_v0_metrics_from_json(version)
    v3_df = load_v3_metrics_from_json()
    
    if v0_df.empty or v3_df.empty:
        return pd.DataFrame()
    
    # Merge on post_id
    combined_df = v0_df.merge(v3_df, on='post_id', how='inner', suffixes=('_v0', '_v3'))
    
    # Use V3 metadata if V0 doesn't have it, otherwise use V0
    if 'actual_karma_v3' in combined_df.columns:
        combined_df['actual_karma'] = combined_df['actual_karma_v0'].fillna(combined_df['actual_karma_v3'])
        combined_df = combined_df.drop(['actual_karma_v0', 'actual_karma_v3'], axis=1)
    
    if 'title_v3' in combined_df.columns:
        combined_df['title'] = combined_df['title_v0'].fillna(combined_df['title_v3'])
        combined_df = combined_df.drop(['title_v0', 'title_v3'], axis=1)
        
    if 'author_v3' in combined_df.columns:
        combined_df['author'] = combined_df['author_v0'].fillna(combined_df['author_v3'])
        combined_df = combined_df.drop(['author_v0', 'author_v3'], axis=1)
        
    if 'posted_at_v3' in combined_df.columns:
        combined_df['posted_at'] = combined_df['posted_at_v0'].fillna(combined_df['posted_at_v3'])
        combined_df = combined_df.drop(['posted_at_v0', 'posted_at_v3'], axis=1)
    
    return combined_df

def get_human_readable_name(metric_name: str, version: str = "v0-adj") -> str:
    """Convert metric column name to human readable format."""
    if version == "v0-adj":
        mapping = {
            'nano_epistemic_score': 'GPT-5-nano overall quality score',
            'mini_epistemic_score': 'GPT-5-mini overall quality score',
            'full_epistemic_score': 'GPT-5 overall quality score',
            'actual_karma': 'Actual Karma',
            'comment_count': 'Comments',
            'word_count': 'Word Count'
        }
    else:
        mapping = {
            'nano_epistemic_score': 'GPT-5-nano holistic epistemic quality score',
            'mini_epistemic_score': 'GPT-5-mini holistic epistemic quality score',
            'full_epistemic_score': 'GPT-5 holistic epistemic quality score',
            'actual_karma': 'Actual Karma',
            'comment_count': 'Comments',
            'word_count': 'Word Count'
        }
    return mapping.get(metric_name, metric_name.replace('_', ' ').title())

# Define metrics to analyze
V0_METRIC_COLUMNS = [
    'nano_epistemic_score',
    'mini_epistemic_score', 
    'full_epistemic_score'
]

def bootstrap_spearman_ci(x, y, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for Spearman correlation."""
    correlations = []
    n = len(x)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x.iloc[indices] if hasattr(x, 'iloc') else x[indices]
        y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        
        corr, _ = stats.spearmanr(x_boot, y_boot)
        if np.isscalar(corr) and not np.isnan(corr):
            correlations.append(corr)
    
    if len(correlations) > 0:
        alpha = 1 - confidence
        lower = np.percentile(correlations, 100 * alpha / 2)
        upper = np.percentile(correlations, 100 * (1 - alpha / 2))
        return lower, upper
    else:
        return np.nan, np.nan

def calculate_correlations(df: pd.DataFrame, target_col: str = 'actual_karma', version: str = "v0-adj") -> pd.DataFrame:
    """Calculate correlations between V0 metrics and target with bootstrap CIs."""
    
    correlations = []
    
    for col in V0_METRIC_COLUMNS:
        if col in df.columns and target_col in df.columns:
            valid_data = df[[col, target_col]].dropna()
            
            if len(valid_data) > 1:
                correlation, p_value = stats.spearmanr(valid_data[col], valid_data[target_col])
                
                if np.isscalar(correlation) and not np.isnan(correlation):
                    # Calculate bootstrap confidence interval
                    ci_lower, ci_upper = bootstrap_spearman_ci(
                        valid_data[col], valid_data[target_col]
                    )
                    
                    correlations.append({
                        'Metric': get_human_readable_name(col, version),
                        'Correlation': correlation,
                        'N': len(valid_data),
                        '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "N/A",
                        'metric_col': col
                    })
    
    corr_df = pd.DataFrame(correlations)
    if not corr_df.empty:
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    
    return corr_df

def correlation_analysis_tab(df: pd.DataFrame, version: str = "v0-adj"):
    """Correlation analysis tab for V0 metrics."""
    st.header("📊 V0 Metrics Correlation Analysis")
    
    # Check if we have actual karma data
    if 'actual_karma' not in df.columns or df['actual_karma'].isna().all():
        st.error("No actual karma data available for correlation analysis")
        return
    
    # Date range filtering
    if 'posted_at' in df.columns:
        # Convert posted_at to datetime if it's not already
        df['posted_at'] = pd.to_datetime(df['posted_at'])
        
        # Get min and max dates from the data
        min_date = df['posted_at'].min().date() if df['posted_at'].notna().any() else date(2024, 1, 1)
        max_date = df['posted_at'].max().date() if df['posted_at'].notna().any() else date.today()
        
        # Ensure we have at least the range from Jan 2024 to current date
        min_date = min(min_date, date(2024, 1, 1))
        max_date = max(max_date, date.today())
        
        st.subheader("📅 Date Range Filter")
        date_range = st.slider(
            "Select date range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="MM/YYYY",
            key=f"v0_date_range_{version}"
        )
        
        # Filter dataframe by date range
        filtered_df = df[
            (df['posted_at'].dt.date >= date_range[0]) & 
            (df['posted_at'].dt.date <= date_range[1])
        ].copy()
        
        quality_type = "overall quality scores" if version == "v0-adj" else "epistemic quality scores"
        st.info(f"Analyzing {len(filtered_df)} posts (filtered from {len(df)} total) with V0 {quality_type}")
    else:
        filtered_df = df.copy()
        quality_type = "overall quality scores" if version == "v0-adj" else "epistemic quality scores"
        st.info(f"Analyzing {len(df)} posts with V0 {quality_type} (no date filtering available)")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nano_posts = filtered_df['nano_epistemic_score'].notna().sum() if 'nano_epistemic_score' in filtered_df.columns else 0
        st.metric("Posts with Nano scores", nano_posts)
    
    with col2:
        mini_posts = filtered_df['mini_epistemic_score'].notna().sum() if 'mini_epistemic_score' in filtered_df.columns else 0
        st.metric("Posts with Mini scores", mini_posts)
    
    with col3:
        full_posts = filtered_df['full_epistemic_score'].notna().sum() if 'full_epistemic_score' in filtered_df.columns else 0
        st.metric("Posts with Full scores", full_posts)
    
    # Toggle for correlation target
    correlation_target = st.radio(
        "Show correlations with:",
        ["Actual Karma", "Comment Count"],
        horizontal=True,
        disabled='comment_count' not in filtered_df.columns,
        key=f"v0_corr_target_{version}"
    )
    
    target_col = 'actual_karma' if correlation_target == "Actual Karma" else 'comment_count'
    
    # Calculate correlations
    corr_df = calculate_correlations(filtered_df, target_col, version)
    
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
            title=f"Spearman Correlation with {correlation_target}"
        )
        fig.update_layout(height=max(300, len(corr_df) * 60))
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation table
        st.subheader("Correlation Details")
        display_df = corr_df[['Metric', 'Correlation', '95% CI', 'N']].copy()
        display_df['Correlation'] = display_df['Correlation'].round(3)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Inter-model correlation matrix
    st.subheader("Inter-Model Correlations")
    st.caption("How well do different model sizes agree?")
    
    metric_cols = [col for col in V0_METRIC_COLUMNS if col in filtered_df.columns]
    
    if len(metric_cols) >= 2:
        # Calculate correlation matrix
        corr_matrix = filtered_df[metric_cols].corr(method='spearman')
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix.values,
            x=[get_human_readable_name(col, version) for col in corr_matrix.columns],
            y=[get_human_readable_name(col, version) for col in corr_matrix.index],
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1,
            aspect="auto"
        )
        
        # Add correlation values as text
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                )
        
        fig.update_layout(
            height=400,
            xaxis_title="",
            yaxis_title=""
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots comparing models
    if len(metric_cols) >= 2:
        st.subheader("Model Comparisons")
        
        col1, col2 = st.columns(2)
        
        # Compare Nano vs Mini
        if 'nano_epistemic_score' in filtered_df.columns and 'mini_epistemic_score' in filtered_df.columns:
            with col1:
                fig = px.scatter(
                    filtered_df,
                    x='nano_epistemic_score',
                    y='mini_epistemic_score',
                    title="Nano vs Mini Scores",
                    hover_data=['title'] if 'title' in filtered_df.columns else None,
                    labels={
                        'nano_epistemic_score': 'GPT-Nano Score',
                        'mini_epistemic_score': 'GPT-Mini Score'
                    }
                )
                fig.add_shape(
                    type="line",
                    x0=1, y0=1, x1=10, y1=10,
                    line=dict(color="gray", dash="dash")
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Compare Mini vs Full
        if 'mini_epistemic_score' in filtered_df.columns and 'full_epistemic_score' in filtered_df.columns:
            with col2:
                fig = px.scatter(
                    filtered_df,
                    x='mini_epistemic_score',
                    y='full_epistemic_score',
                    title="Mini vs Full Scores",
                    hover_data=['title'] if 'title' in filtered_df.columns else None,
                    labels={
                        'mini_epistemic_score': 'GPT-Mini Score',
                        'full_epistemic_score': 'GPT-Full Score'
                    }
                )
                fig.add_shape(
                    type="line",
                    x0=1, y0=1, x1=10, y1=10,
                    line=dict(color="gray", dash="dash")
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Distribution comparison
    st.subheader("Score Distributions by Model")
    
    # Create distribution plots
    fig = go.Figure()
    
    for col in metric_cols:
        scores = filtered_df[col].dropna()
        if len(scores) > 0:
            fig.add_trace(go.Histogram(
                x=scores,
                name=get_human_readable_name(col, version),
                opacity=0.7,
                nbinsx=10
            ))
    
    score_type = "Quality Score" if version == "v0-adj" else "Epistemic Quality Score"
    fig.update_layout(
        barmode='overlay',
        xaxis_title=score_type,
        yaxis_title="Count",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical comparison
    st.subheader("Statistical Summary")
    
    summary_data = []
    for col in metric_cols:
        if col in filtered_df.columns:
            scores = filtered_df[col].dropna()
            if len(scores) > 0:
                summary_data.append({
                    'Model': get_human_readable_name(col, version),
                    'Mean': scores.mean(),
                    'Median': scores.median(),
                    'Std Dev': scores.std(),
                    'Min': scores.min(),
                    'Max': scores.max(),
                    'Count': len(scores)
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(2)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

def v0_vs_v3_comparison_tab(version: str = "v0-adj"):
    """V0 vs V3 comparison tab showing mini epistemic vs v3 overall epistemic."""
    st.header("🔬 V0 vs V3 Comparison")
    quality_type = "overall quality" if version == "v0-adj" else "holistic epistemic quality"
    st.caption(f"Comparing GPT-5-mini {quality_type} (V0) with V3 overall epistemic quality")

    # Load combined data
    combined_df = load_combined_v0_v3_data(version)
    
    if combined_df.empty:
        st.error("No overlapping posts found between V0 and V3 datasets!")
        st.info("Make sure both datasets contain posts with matching post_ids")
        return
    
    # Filter to only posts that have both metrics
    comparison_df = combined_df[
        combined_df['mini_epistemic_score'].notna() & 
        combined_df['v3_overall_epistemic_score'].notna()
    ].copy()
    
    if comparison_df.empty:
        st.error("No posts found with both mini epistemic scores and V3 overall epistemic scores!")
        return
    
    # Date range filtering (same as V0 analysis)
    if 'posted_at' in comparison_df.columns:
        # Convert posted_at to datetime if it's not already
        comparison_df['posted_at'] = pd.to_datetime(comparison_df['posted_at'])
        
        # Get min and max dates from the data
        min_date = comparison_df['posted_at'].min().date() if comparison_df['posted_at'].notna().any() else date(2024, 1, 1)
        max_date = comparison_df['posted_at'].max().date() if comparison_df['posted_at'].notna().any() else date.today()
        
        # Ensure we have at least the range from Jan 2024 to current date
        min_date = min(min_date, date(2024, 1, 1))
        max_date = max(max_date, date.today())
        
        st.subheader("📅 Date Range Filter")
        date_range = st.slider(
            "Select date range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="MM/YYYY",
            key=f"v0_v3_date_range_{version}"
        )
        
        # Filter dataframe by date range
        filtered_df = comparison_df[
            (comparison_df['posted_at'].dt.date >= date_range[0]) & 
            (comparison_df['posted_at'].dt.date <= date_range[1])
        ].copy()
        
        st.info(f"Comparing {len(filtered_df)} posts (filtered from {len(comparison_df)} total) with both V0 and V3 scores")
    else:
        filtered_df = comparison_df.copy()
        st.info(f"Comparing {len(comparison_df)} posts with both V0 and V3 scores (no date filtering available)")
    
    if filtered_df.empty:
        st.warning("No posts in selected date range!")
        return
    
    # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        mean_v0 = filtered_df['mini_epistemic_score'].mean()
        st.metric("V0 Mini Mean Score", f"{mean_v0:.2f}")
    
    with col2:
        mean_v3 = filtered_df['v3_overall_epistemic_score'].mean()
        st.metric("V3 Overall Mean Score", f"{mean_v3:.2f}")
    
    # Toggle for correlation target
    correlation_target = st.radio(
        "Show correlations with:",
        ["Actual Karma", "Comment Count"],
        horizontal=True,
        disabled='comment_count' not in filtered_df.columns,
        key=f"v0_v3_corr_target_{version}"
    )
    
    target_col = 'actual_karma' if correlation_target == "Actual Karma" else 'comment_count'
    
    # Check if we have the target column
    if target_col not in filtered_df.columns or filtered_df[target_col].isna().all():
        st.error(f"No {correlation_target.lower()} data available for correlation analysis")
        return
    
    # Calculate correlations for both metrics
    correlations = []
    
    v0_label = "Naive prompt for overall quality (gpt-5-mini)" if version == "v0-adj" else "Naive prompt for epistemic quality (gpt-5-mini)"
    for metric_name, col_name, display_name in [
        ('mini_epistemic_score', 'mini_epistemic_score', v0_label),
        ('v3_overall_epistemic_score', 'v3_overall_epistemic_score', 'Our epistemic quality metric with rubric and scaffolding (gpt-5-mini)')
    ]:
        valid_data = filtered_df[[col_name, target_col]].dropna()
        
        if len(valid_data) > 1:
            correlation, p_value = stats.spearmanr(valid_data[col_name], valid_data[target_col])
            
            if np.isscalar(correlation) and not np.isnan(correlation):
                # Calculate bootstrap confidence interval
                ci_lower, ci_upper = bootstrap_spearman_ci(
                    valid_data[col_name], valid_data[target_col]
                )
                
                correlations.append({
                    'Metric': display_name,
                    'Correlation': correlation,
                    'N': len(valid_data),
                    '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "N/A",
                    'metric_col': col_name
                })
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        
        # Bar chart of correlations
        fig = px.bar(
            corr_df,
            x='Correlation',
            y='Metric',
            orientation='h',
            color='Correlation',
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            title=f"Spearman Correlation with {correlation_target}"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation table
        st.subheader("Correlation Details")
        display_df = corr_df[['Metric', 'Correlation', '95% CI', 'N']].copy()
        display_df['Correlation'] = display_df['Correlation'].round(3)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Direct comparison scatter plot
    st.subheader("V0 vs V3 Direct Comparison")
    fig = px.scatter(
        filtered_df,
        x='mini_epistemic_score',
        y='v3_overall_epistemic_score',
        title="V0 Mini Epistemic vs V3 Overall Epistemic",
        hover_data=['title'] if 'title' in filtered_df.columns else None,
        labels={
            'mini_epistemic_score': 'Out of the box GPT-5-mini overall quality score' if version == "v0-adj" else 'Out of the box GPT-5-mini epistemic quality score',
            'v3_overall_epistemic_score': 'GPT5-mini epistemic quality score with rubric and context'
        }
    )
    
    # Add diagonal line for reference
    min_score = min(filtered_df['mini_epistemic_score'].min(), filtered_df['v3_overall_epistemic_score'].min())
    max_score = max(filtered_df['mini_epistemic_score'].max(), filtered_df['v3_overall_epistemic_score'].max())
    fig.add_shape(
        type="line",
        x0=min_score, y0=min_score, x1=max_score, y1=max_score,
        line=dict(color="gray", dash="dash")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Direct correlation between the two metrics
    v0_v3_corr, _ = stats.spearmanr(
        filtered_df['mini_epistemic_score'].dropna(),
        filtered_df['v3_overall_epistemic_score'].dropna()
    )
    
    st.metric(
        "Direct Correlation (V0 Mini vs V3 Overall)",
        f"{v0_v3_corr:.3f}" if not np.isnan(v0_v3_corr) else "N/A"
    )

def all_correlations_tab(version: str = "v0-adj"):
    """Tab showing all V0 and V3 correlations with karma together."""
    st.header("📊 All Quality Correlations")
    quality_type = "overall quality" if version == "v0-adj" else "epistemic quality"
    st.caption(f"Comparing all V0 naive {quality_type} prompts and V3 overall epistemic quality correlations with actual karma")

    # Load both datasets
    v0_df = load_v0_metrics_from_json(version)
    v3_df = load_v3_metrics_from_json()
    
    if v0_df.empty and v3_df.empty:
        st.error("No metrics data found!")
        return
    
    # Merge on post_id to get all metrics together
    if not v0_df.empty and not v3_df.empty:
        combined_df = v0_df.merge(v3_df[['post_id', 'v3_overall_epistemic_score']], 
                                  on='post_id', how='outer')
    elif not v0_df.empty:
        combined_df = v0_df
    else:
        combined_df = v3_df
    
    # Load metadata from database if needed
    if 'actual_karma' not in combined_df.columns or combined_df['actual_karma'].isna().all():
        try:
            from sqlalchemy import create_engine
            from dotenv import load_dotenv
            load_dotenv()
            DATABASE_URL = os.getenv("DATABASE_URL")
            
            if DATABASE_URL:
                engine = create_engine(DATABASE_URL)
                
                # Get post metadata
                post_ids = combined_df['post_id'].tolist()
                if post_ids:
                    metadata_query = f"""
                    SELECT 
                        post_id,
                        base_score as actual_karma,
                        comment_count,
                        posted_at
                    FROM fellowship_mvp
                    WHERE post_id IN ({','.join([f"'{pid}'" for pid in post_ids])})
                    """
                    
                    metadata_df = pd.read_sql_query(metadata_query, engine)
                    engine.dispose()
                    
                    # Merge metadata
                    combined_df = combined_df.merge(metadata_df, on='post_id', how='left', suffixes=('', '_db'))
                    
                    if 'actual_karma_db' in combined_df.columns:
                        combined_df['actual_karma'] = combined_df['actual_karma'].fillna(combined_df['actual_karma_db'])
                        combined_df = combined_df.drop('actual_karma_db', axis=1)
                        
        except Exception as e:
            st.info(f"Could not load metadata from database: {e}")
    
    # Check if we have karma data
    if 'actual_karma' not in combined_df.columns or combined_df['actual_karma'].isna().all():
        st.error("No actual karma data available for correlation analysis")
        return
    
    # Date range filtering
    if 'posted_at' in combined_df.columns:
        combined_df['posted_at'] = pd.to_datetime(combined_df['posted_at'])
        
        min_date = combined_df['posted_at'].min().date() if combined_df['posted_at'].notna().any() else date(2024, 1, 1)
        max_date = combined_df['posted_at'].max().date() if combined_df['posted_at'].notna().any() else date.today()
        
        min_date = min(min_date, date(2024, 1, 1))
        max_date = max(max_date, date.today())
        
        st.subheader("📅 Date Range Filter")
        date_range = st.slider(
            "Select date range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="MM/YYYY",
            key=f"all_corr_date_range_{version}"
        )
        
        # Filter dataframe by date range
        filtered_df = combined_df[
            (combined_df['posted_at'].dt.date >= date_range[0]) & 
            (combined_df['posted_at'].dt.date <= date_range[1])
        ].copy()
        
        st.info(f"Analyzing {len(filtered_df)} posts (filtered from {len(combined_df)} total)")
    else:
        filtered_df = combined_df.copy()
        st.info(f"Analyzing {len(combined_df)} posts")
    
    if filtered_df.empty:
        st.warning("No posts in selected date range!")
        return
    
    # Add toggle for including V3 metric
    include_v3 = st.checkbox("Include V3 metric", value=True,
                             help="Toggle to include/exclude the V3 epistemic quality metric with scaffolding",
                             key=f"include_v3_{version}")
    
    # Calculate correlations for all metrics
    correlations = []
    
    # V0 metrics
    prompt_type = "overall quality" if version == "v0-adj" else "epistemic quality"
    v0_metrics = [
        ('nano_epistemic_score', f'Naive {prompt_type} prompt (gpt-5-nano)'),
        ('mini_epistemic_score', f'Naive {prompt_type} prompt (gpt-5-mini)'),
        ('full_epistemic_score', f'Naive {prompt_type} prompt (gpt-5)')
    ]
    
    for col, display_name in v0_metrics:
        if col in filtered_df.columns:
            valid_data = filtered_df[[col, 'actual_karma']].dropna()
            
            if len(valid_data) > 1:
                correlation, p_value = stats.spearmanr(valid_data[col], valid_data['actual_karma'])
                
                if np.isscalar(correlation) and not np.isnan(correlation):
                    ci_lower, ci_upper = bootstrap_spearman_ci(
                        valid_data[col], valid_data['actual_karma']
                    )
                    
                    correlations.append({
                        'Metric': display_name,
                        'Type': 'V0 (Naive)',
                        'Correlation': correlation,
                        'N': len(valid_data),
                        '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "N/A",
                        'P-value': p_value
                    })
    
    # V3 overall epistemic quality (only if toggle is enabled)
    if include_v3 and 'v3_overall_epistemic_score' in filtered_df.columns:
        valid_data = filtered_df[['v3_overall_epistemic_score', 'actual_karma']].dropna()
        
        if len(valid_data) > 1:
            correlation, p_value = stats.spearmanr(valid_data['v3_overall_epistemic_score'], valid_data['actual_karma'])
            
            if np.isscalar(correlation) and not np.isnan(correlation):
                ci_lower, ci_upper = bootstrap_spearman_ci(
                    valid_data['v3_overall_epistemic_score'], valid_data['actual_karma']
                )
                
                correlations.append({
                    'Metric': 'Epistemic quality metric with our scaffolding (gpt-5-mini)',
                    'Type': 'V3 (Advanced)',
                    'Correlation': correlation,
                    'N': len(valid_data),
                    '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "N/A",
                    'P-value': p_value
                })
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        
        # Sort by absolute correlation
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        # Bar chart of all correlations
        fig = px.bar(
            corr_df,
            x='Correlation',
            y='Metric',
            orientation='h',
            color='Type',
            color_discrete_map={'V0 (Naive)': '#636EFA', 'V3 (Advanced)': '#EF553B'},
            title="Spearman Correlation with Actual Karma",
            hover_data=['N', '95% CI']
        )
        
        # Add a vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            height=max(400, len(corr_df) * 80),
            showlegend=False,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            yaxis=dict(
                tickfont=dict(size=14)  # Increase y-axis label font size
            ),
            xaxis=dict(
                tickfont=dict(size=12)  # Keep x-axis font size reasonable
            ),
            font=dict(size=13)  # Overall font size increase
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed correlation table
        st.subheader("Detailed Correlation Results")
        display_df = corr_df[['Metric', 'Type', 'Correlation', '95% CI', 'N']].copy()
        display_df['Correlation'] = display_df['Correlation'].round(3)
        
        # Highlight the best correlation
        # def highlight_best(row):
        #     if abs(row['Correlation']) == display_df['Correlation'].abs().max():
        #         return ['background-color: #ffe6e6'] * len(row)
        #     return [''] * len(row)
        
        # styled_df = display_df.style.apply(highlight_best, axis=1)
        st.dataframe(display_df[['Metric','Correlation', '95% CI', 'N']], use_container_width=True, hide_index=True)
        
        # Analysis text
        st.subheader("📈 Analysis")
        
        best_overall = corr_df.loc[corr_df['Correlation'].abs().idxmax()]
        st.write(f"**Best performing metric:** {best_overall['Metric']} with correlation {best_overall['Correlation']:.3f}")
        
        if include_v3 and 'V3 (Advanced)' in corr_df['Type'].values and 'V0 (Naive)' in corr_df['Type'].values:
            v3_corr = corr_df[corr_df['Type'] == 'V3 (Advanced)']['Correlation'].values[0]
            best_v0 = corr_df[corr_df['Type'] == 'V0 (Naive)']['Correlation'].abs().max()
            
            if abs(v3_corr) > best_v0:
                st.success(f"✅ V3 metric with rubric and scaffolding outperforms all naive V0 prompts by {abs(v3_corr) - best_v0:.3f}")
            else:
                st.warning(f"⚠️ Best V0 naive prompt performs comparably to V3 (difference: {abs(v3_corr) - best_v0:.3f})")
        elif not include_v3:
            # If V3 is not included, just show best V0
            v0_corrs = corr_df[corr_df['Type'] == 'V0 (Naive)']['Correlation'].values
            if len(v0_corrs) > 0:
                st.info(f"Showing V0 metrics only. Best V0 correlation: {max(v0_corrs, key=abs):.3f}")
    else:
        st.warning("No valid correlations could be calculated with the available data")

def main():
    st.title("🔬 V0 Metrics Explorer")

    # Version selector
    version = st.selectbox(
        "Select Version:",
        ["v0-adj", "v0"],
        help="v0-adj: overall quality | v0: holistic epistemic quality"
    )

    data_dir = V0_ADJ_DATA_DIR if version == "v0-adj" else V0_DATA_DIR
    quality_type = "overall quality scores" if version == "v0-adj" else "epistemic quality scores"
    st.caption(f"Analyzing {quality_type} from {data_dir}")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["All Correlations", "V0 Analysis", "V0 vs V3 Comparison"])

    with tab1:
        all_correlations_tab(version)

    with tab2:
        # Load V0 data - clear any caching by using version as key
        df = load_v0_metrics_from_json(version)

        if df.empty:
            st.error("No V0 metrics data found!")
            st.info(f"Please ensure JSON files are present in: {data_dir.absolute()}")
        else:
            # Force refresh by using version in key
            st.write(f"**Current version:** {version}")
            correlation_analysis_tab(df, version)

    with tab3:
        v0_vs_v3_comparison_tab(version)

if __name__ == "__main__":
    main()