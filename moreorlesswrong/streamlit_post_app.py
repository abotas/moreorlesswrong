"""Streamlit app for visualizing post metrics."""

import streamlit as st
import json
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
from typing import List
from scipy import stats
import numpy as np

from db import get_representative_posts
from models import Post
from post_metric_registry import get_human_readable_name

# Get version ID from command line args if provided
if len(sys.argv) > 1:
    DEFAULT_VERSION_ID = sys.argv[1]
else:
    DEFAULT_VERSION_ID = "post_v1"

st.set_page_config(page_title="EA Forum Post Metrics", layout="wide")


def discover_available_metrics(version_id: str) -> List[str]:
    """Discover which metrics are available in the data for a given version."""
    metrics_dir = Path(f"data/post_metrics/{version_id}")
    
    if not metrics_dir.exists():
        return []
    
    available_metrics = set()
    
    # Sample first file to see what metrics are available
    for metrics_file in metrics_dir.glob("*.json"):
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            
        available_metrics.update(metrics_data.keys())
        
        # Just check first file for efficiency (assuming all files have same metrics)
        break
    
    return sorted(list(available_metrics))


def load_post_metrics(version_id: str, metric_names: List[str]):
    """Load post metrics data for a given version."""
    metrics_dir = Path(f"data/post_metrics/{version_id}")
    
    if not metrics_dir.exists():
        return None
    
    all_data = []
    
    for metrics_file in metrics_dir.glob("*.json"):
        post_id = metrics_file.stem
        
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        row = {"post_id": post_id}
        
        for metric_name in metric_names:
            if metric_name in metrics_data:
                metric_data = metrics_data[metric_name]
                # Add all fields from the metric
                for key, value in metric_data.items():
                    if key != "post_id":  # Skip duplicate
                        row[f"{metric_name}_{key}"] = value
        
        all_data.append(row)
    
    return pd.DataFrame(all_data)


def get_all_submetrics(df: pd.DataFrame) -> List[str]:
    """Get all numeric submetrics from the dataframe."""
    submetrics = []
    
    # Add base_score if available
    if 'base_score' in df.columns and not df['base_score'].isna().all():
        submetrics.append('base_score')
    
    # Add all numeric metric columns
    for col in df.columns:
        if col not in ['post_id', 'title', 'author'] and df[col].dtype in ['int64', 'float64']:
            submetrics.append(col)
    
    return sorted(submetrics)


def calculate_correlations_with_base_score(df: pd.DataFrame) -> List[tuple]:
    """Calculate correlations with base_score and return top 5 positive correlations."""
    if 'base_score' not in df.columns or df['base_score'].isna().all():
        return []
    
    correlations = []
    
    for col in df.columns:
        if col != 'base_score' and df[col].dtype in ['int64', 'float64']:
            # Remove NaN values for correlation calculation
            valid_data = df[[col, 'base_score']].dropna()
            
            if len(valid_data) > 1:
                correlation, p_value = stats.pearsonr(valid_data[col], valid_data['base_score'])
                if not np.isnan(correlation):
                    correlations.append({
                        'metric': col,
                        'correlation': correlation,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
    
    # Sort by correlation descending (highest positive correlations first)
    correlations.sort(key=lambda x: x['correlation'], reverse=True)
    
    # Return top 5 positive correlations
    return [c for c in correlations if c['correlation'] > 0][:5]


def aggregate_by_author(df: pd.DataFrame, submetrics: List[str]) -> pd.DataFrame:
    """Aggregate metrics by author (mean scores per author)."""
    if 'author' not in df.columns:
        return pd.DataFrame()
    
    # Group by author and calculate mean for numeric columns
    numeric_cols = [col for col in submetrics if col in df.columns]
    author_data = df.groupby('author')[numeric_cols].mean().reset_index()
    
    # Add post count per author
    post_counts = df.groupby('author').size().reset_index(name='post_count')
    author_data = author_data.merge(post_counts, on='author')
    
    return author_data


def main():
    st.title("EA Forum Post Metrics Analysis")
    
    # Sidebar inputs
    st.sidebar.header("Configuration")
    
    version_id = st.sidebar.text_input(
        "Version ID",
        value=DEFAULT_VERSION_ID,
        help="The version ID of the pipeline run"
    )
    
    # Dynamically discover available metrics from the data
    available_metrics = discover_available_metrics(version_id)
    
    if not available_metrics:
        st.warning(f"No metrics found for version '{version_id}'. Please check the version ID or run the post_metric_pipeline first.")
        st.code("uv run python moreorlesswrong/post_metric_pipeline.py --threads 4 --model gpt-5-mini")
        return
    
    # Load data
    df = load_post_metrics(version_id, available_metrics)
    
    if df is None or df.empty:
        st.warning("No data available for the selected metrics.")
        return
    
    # Get posts and their metadata
    posts = get_representative_posts(1000)
    post_scores = {p.post_id: p.base_score for p in posts}
    post_titles = {p.post_id: p.title[:60] + "..." if len(p.title) > 60 else p.title for p in posts}
    post_authors = {p.post_id: p.author_display_name for p in posts}
    
    df["base_score"] = df["post_id"].map(post_scores)
    df["title"] = df["post_id"].map(post_titles)
    df["author"] = df["post_id"].map(post_authors)
    
    # Filter out posts without base_score
    df = df.dropna(subset=['base_score'])
    
    # Get all submetrics
    all_submetrics = get_all_submetrics(df)
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Posts", len(df))
    with col2:
        st.metric("Metrics Available", len(available_metrics))
    with col3:
        if 'base_score' in df.columns and not df['base_score'].isna().all():
            st.metric("Average Base Score", f"{df['base_score'].mean():.1f}")
        else:
            st.metric("Average Base Score", "N/A")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Correlations", "Leaderboards", "Metric Histograms"])
    
    with tab1:
        st.header("Correlations with Base Score")
        
        if 'base_score' in df:
            # Calculate all correlations
            all_correlations = []
            
            for col in all_submetrics:
                if col != 'base_score' and col in df.columns:
                    valid_data = df[[col, 'base_score']].dropna()
                    
                    if len(valid_data) > 1:
                        correlation, p_value = stats.pearsonr(valid_data[col], valid_data['base_score'])
                        if not np.isnan(correlation):
                            all_correlations.append({
                                'Metric': get_human_readable_name(col),
                                'Correlation': correlation,
                                'P-value': p_value,
                                'Significant': '✓' if p_value < 0.05 else ''
                            })
            
            if all_correlations:
                corr_df = pd.DataFrame(all_correlations)
                corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                
                # Bar chart of correlations
                fig = px.bar(
                    corr_df,
                    x='Correlation',
                    y='Metric',
                    orientation='h',
                    color='Correlation',
                    color_continuous_scale='RdBu',
                    range_color=[-1, 1],
                    title="Correlation with Base Score"
                )
                fig.update_layout(height=max(400, len(corr_df) * 25))
                st.plotly_chart(fig, use_container_width=True)
                
                # Table with significance
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            
            # Scatter plots for top correlations
            st.subheader("Scatter Plots (Top Correlations)")
            
            top_correlations = calculate_correlations_with_base_score(df)
            
            for i, corr_item in enumerate(top_correlations[:3]):
                metric_col = corr_item['metric']
                
                if metric_col in df.columns:
                    human_name = get_human_readable_name(metric_col)
                    fig = px.scatter(
                        df,
                        x=metric_col,
                        y='base_score',
                        title=f"{human_name} vs Base Score (r={corr_item['correlation']:.3f})",
                        hover_data=['title', 'author'],
                        trendline="ols"
                    )
                    fig.update_xaxes(title=human_name)
                    fig.update_yaxes(title="Base Score")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Base score not available for correlation analysis.")
    
    with tab2:
        st.header("Leaderboards")
        
        # View toggles
        col1, col2, col3 = st.columns(3)
        with col1:
            view_type = st.selectbox("View by:", ["Authors", "Posts"], key="view_type")
        with col2:
            sort_order = st.selectbox("Show:", ["Top", "Bottom"], key="sort_order")
        with col3:
            if view_type == "Authors":
                min_posts = st.number_input("Min posts per author:", min_value=1, value=3, key="min_posts")
            else:
                min_posts = 3
        
        # Get top 5 correlations
        top_correlations = calculate_correlations_with_base_score(df)
        
        # Add post karma leaderboard first
        st.subheader("Post Karma Leaderboard by Author")
        st.write("*Authors ranked by average post karma (EA Forum upvotes) across their posts*")
        
        if view_type == "Authors":
            # Aggregate by author for base score
            author_data = aggregate_by_author(df, ['base_score'])
            
            if not author_data.empty:
                # Filter by minimum post count
                author_data = author_data[author_data['post_count'] >= min_posts]
                
                if not author_data.empty:
                    st.write(f"*{len(author_data)} authors with ≥{min_posts} posts*")
                    if sort_order == "Top":
                        base_score_authors = author_data.nlargest(5, 'base_score')[['author', 'base_score', 'post_count']]
                    else:
                        base_score_authors = author_data.nsmallest(5, 'base_score')[['author', 'base_score', 'post_count']]
                    
                    # Rename columns to be more readable
                    display_df = base_score_authors.copy()
                    display_df.columns = ['Author', 'Average Post Karma', 'Post Count']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No authors found with at least {min_posts} posts.")
        
        if top_correlations:
            st.subheader(f"Top 5 Metrics Most Correlated with Post Karma")
            st.write("*Post karma = EA Forum upvotes for the post*")
            st.write("*r = correlation coefficient (-1 to +1, where +1 = perfect positive correlation)*")
            
            for i, corr_item in enumerate(top_correlations):
                metric_name = corr_item['metric']
                correlation = corr_item['correlation']
                human_name = get_human_readable_name(metric_name)
                
                st.write(f"**{i+1}. {human_name}** (r={correlation:.3f})")
                
                if view_type == "Posts":
                    # Show top/bottom posts for this metric
                    if sort_order == "Top":
                        top_posts = df.nlargest(5, metric_name)[['title', 'author', metric_name, 'base_score']]
                    else:
                        top_posts = df.nsmallest(5, metric_name)[['title', 'author', metric_name, 'base_score']]
                    
                    st.dataframe(top_posts, use_container_width=True, hide_index=True)
                
                else:  # Authors
                    # Aggregate by author
                    author_data = aggregate_by_author(df, [metric_name, 'base_score'])
                    
                    if not author_data.empty:
                        # Filter by minimum post count
                        author_data = author_data[author_data['post_count'] >= min_posts]
                        
                        if not author_data.empty:
                            st.write(f"*{len(author_data)} authors with ≥{min_posts} posts*")
                            if sort_order == "Top":
                                top_authors = author_data.nlargest(5, metric_name)[['author', metric_name, 'post_count']]
                            else:
                                top_authors = author_data.nsmallest(5, metric_name)[['author', metric_name, 'post_count']]
                            
                            # Rename columns to be more readable
                            display_df = top_authors.copy()
                            display_df.columns = ['Author', human_name, 'Post Count']
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                        else:
                            st.info(f"No authors found with at least {min_posts} posts.")
                
                st.write("")  # Add spacing
        
        else:
            st.info("No positive correlations found with base score.")
    
    with tab3:
        st.header("Metric Histograms")
        
        # Create human readable options for dropdown
        metric_options = {get_human_readable_name(metric): metric for metric in all_submetrics}
        
        # Dropdown to select metric
        selected_human_name = st.selectbox(
            "Select metric to visualize:",
            list(metric_options.keys()),
            key="histogram_metric"
        )
        
        selected_metric = metric_options[selected_human_name]
        
        if selected_metric and selected_metric in df.columns:
            # Create histogram
            fig = px.histogram(
                df,
                x=selected_metric,
                nbins=20,
                title=f"Distribution of {selected_human_name}",
                labels={selected_metric: selected_human_name}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show basic statistics
            col_stats = df[selected_metric].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{col_stats['mean']:.2f}")
            with col2:
                st.metric("Std Dev", f"{col_stats['std']:.2f}")
            with col3:
                st.metric("Min", f"{col_stats['min']:.0f}")
            with col4:
                st.metric("Max", f"{col_stats['max']:.0f}")


if __name__ == "__main__":
    main()