"""Streamlit app for visualizing alej_v1 metrics from database."""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
from scipy import stats
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

st.set_page_config(page_title="EA Forum Metrics Analysis (DB)", layout="wide")

def get_connection():
    """Create a new database connection."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# Cache data loading
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_metrics_data(cluster_n: int = 5) -> pd.DataFrame:
    """Load posts with alej_v1_metrics and cluster information from database."""
    # Use SQLAlchemy engine for pandas compatibility
    engine = create_engine(DATABASE_URL)
    
    query = f"""
    SELECT 
        post_id,
        title,
        author_display_name as author,
        base_score,
        ea_cluster_{cluster_n} as cluster_id,
        ea_cluster_{cluster_n}_name as cluster_name,
        -- Extract all metrics from JSONB
        (alej_v1_metrics->>'value_ea')::int as value_ea,
        (alej_v1_metrics->>'value_humanity')::int as value_humanity,
        (alej_v1_metrics->>'robustness_score')::int as robustness_score,
        (alej_v1_metrics->>'author_fame_ea')::int as author_fame_ea,
        (alej_v1_metrics->>'author_fame_humanity')::int as author_fame_humanity,
        (alej_v1_metrics->>'clarity_score')::int as clarity_score,
        (alej_v1_metrics->>'novelty_ea')::int as novelty_ea,
        (alej_v1_metrics->>'novelty_humanity')::int as novelty_humanity,
        (alej_v1_metrics->>'reasoning_quality')::int as reasoning_quality,
        (alej_v1_metrics->>'evidence_quality')::int as evidence_quality,
        (alej_v1_metrics->>'overall_support')::int as overall_support,
        (alej_v1_metrics->>'emperical_claim_validation_score')::int as empirical_validation
    FROM fellowship_mvp
    WHERE alej_v1_metrics IS NOT NULL
    """
    
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    
    return df

def get_human_readable_name(metric_name: str) -> str:
    """Convert metric column name to human readable format."""
    mapping = {
        'value_ea': 'Value to EA',
        'value_humanity': 'Value to Humanity',
        'robustness_score': 'Robustness',
        'author_fame_ea': 'Author Fame (EA)',
        'author_fame_humanity': 'Author Fame (Humanity)',
        'clarity_score': 'Clarity',
        'novelty_ea': 'Novelty (EA)',
        'novelty_humanity': 'Novelty (Humanity)',
        'reasoning_quality': 'Reasoning Quality',
        'evidence_quality': 'Evidence Quality',
        'overall_support': 'Overall Support',
        'empirical_validation': 'Empirical Validation',
        'base_score': 'Post Karma'
    }
    return mapping.get(metric_name, metric_name.replace('_', ' ').title())

def get_available_cluster_ns() -> List[int]:
    """Get available cluster sizes from database."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Check which ea_cluster_N columns exist
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'fellowship_mvp' 
        AND column_name LIKE 'ea_cluster_%_name'
    """)
    
    columns = cur.fetchall()
    conn.close()
    
    # Extract N values
    cluster_ns = []
    for col in columns:
        col_name = col['column_name']
        # Extract number from ea_cluster_N_name
        parts = col_name.split('_')
        if len(parts) >= 3 and parts[2].isdigit():
            cluster_ns.append(int(parts[2]))
    
    return sorted(cluster_ns)

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between all metrics and base_score."""
    metric_columns = [
        'value_ea', 'value_humanity', 'robustness_score',
        'author_fame_ea', 'author_fame_humanity', 'clarity_score',
        'novelty_ea', 'novelty_humanity', 'reasoning_quality',
        'evidence_quality', 'overall_support', 'empirical_validation'
    ]
    
    correlations = []
    
    for col in metric_columns:
        if col in df.columns:
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

def aggregate_by_cluster(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Aggregate metrics by cluster."""
    # Group by cluster and calculate mean for numeric columns
    numeric_cols = [col for col in metrics if col in df.columns]
    
    if not numeric_cols or 'cluster_name' not in df.columns:
        return pd.DataFrame()
    
    # Remove rows with null cluster names
    df_with_cluster = df.dropna(subset=['cluster_name'])
    
    cluster_data = df_with_cluster.groupby('cluster_name')[numeric_cols].mean()
    
    # Add post count per cluster
    post_counts = df_with_cluster.groupby('cluster_name').size().reset_index(name='post_count')
    cluster_data = cluster_data.reset_index().merge(post_counts, on='cluster_name')
    
    return cluster_data

def main():
    st.title("EA Forum Metrics Analysis (Database)")
    st.caption("Analyzing alej_v1 metrics stored in the database")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Get available cluster sizes
    available_clusters = get_available_cluster_ns()
    
    if not available_clusters:
        st.error("No cluster columns found in database")
        return
    
    # Cluster size selector
    default_cluster = 12 if 12 in available_clusters else available_clusters[0]
    cluster_n = st.sidebar.selectbox(
        "Cluster Size (N)",
        available_clusters,
        index=available_clusters.index(default_cluster),
        help="Number of clusters to use for grouping"
    )
    
    # Load data
    df = load_metrics_data(cluster_n)
    
    if df.empty:
        st.warning("No posts with alej_v1_metrics found in database")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Posts", len(df))
    with col2:
        st.metric("Unique Authors", df['author'].nunique())
    with col3:
        st.metric("Clusters", df['cluster_name'].nunique())
    with col4:
        st.metric("Avg Post Karma", f"{df['base_score'].mean():.1f}")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Correlations", 
        "👥 Author Leaderboards", 
        "🎯 Cluster Leaderboards",
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
                    hover_data=['title', 'author', 'cluster_name'],
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
            top_metrics = ['value_ea', 'clarity_score', 'reasoning_quality']
        
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
        st.header(f"Cluster Leaderboards (N={cluster_n})")
        
        # Show ALL metrics by cluster as bar graphs
        all_metrics = [
            'base_score', 'value_ea', 'value_humanity', 'robustness_score',
            'author_fame_ea', 'author_fame_humanity', 'clarity_score',
            'novelty_ea', 'novelty_humanity', 'reasoning_quality',
            'evidence_quality', 'overall_support', 'empirical_validation'
        ]
        
        for metric in all_metrics:
            if metric in df.columns:
                cluster_data = aggregate_by_cluster(df, [metric])
                
                if not cluster_data.empty:
                    # Remove null values for proper sorting
                    cluster_data_clean = cluster_data.dropna(subset=[metric])
                    
                    if not cluster_data_clean.empty:
                        # Sort by metric value descending (highest first)
                        display_data = cluster_data_clean.sort_values(metric, ascending=False)
                        
                        # Get title with LLM-graded prefix for all metrics except base_score
                        if metric == 'base_score':
                            title = f"Average {get_human_readable_name(metric)} by Cluster"
                            y_label = get_human_readable_name(metric)
                        else:
                            title = f"Average LLM-graded {get_human_readable_name(metric)} by Cluster"
                            y_label = f"LLM-graded {get_human_readable_name(metric)}"
                        
                        # Create bar chart with appropriate color scale
                        if metric == 'base_score':
                            # Use dynamic range for post karma
                            fig = px.bar(
                                display_data,
                                x='cluster_name',
                                y=metric,
                                title=title,
                                labels={
                                    'cluster_name': 'Cluster',
                                    metric: y_label
                                },
                                text=metric,
                                color=metric,
                                color_continuous_scale='viridis'
                            )
                        else:
                            # Use normalized 0-10 scale for LLM-graded metrics
                            fig = px.bar(
                                display_data,
                                x='cluster_name',
                                y=metric,
                                title=title,
                                labels={
                                    'cluster_name': 'Cluster',
                                    metric: y_label
                                },
                                text=metric,
                                color=metric,
                                color_continuous_scale='viridis',
                                range_color=[0, 10]
                            )
                        
                        # Format text based on metric type
                        if metric == 'base_score':
                            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                        else:
                            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        
                        # Set y-axis range
                        if metric == 'base_score':
                            # Dynamic y-axis for post karma
                            fig.update_layout(
                                showlegend=False,
                                xaxis_tickangle=-45,
                                height=500
                            )
                        else:
                            # Fixed 0-10 y-axis for LLM-graded metrics
                            fig.update_layout(
                                showlegend=False,
                                xaxis_tickangle=-45,
                                height=500,
                                yaxis=dict(range=[0, 10])
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Metric Distributions")
        
        # Metric selector
        metric_columns = [
            'value_ea', 'value_humanity', 'robustness_score',
            'clarity_score', 'novelty_ea', 'novelty_humanity',
            'reasoning_quality', 'evidence_quality', 'overall_support'
        ]
        
        selected_metric = st.selectbox(
            "Select metric",
            [m for m in metric_columns if m in df.columns],
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
            
            # Distribution by cluster
            st.subheader("Distribution by Cluster")
            
            fig = px.box(
                df,
                x='cluster_name',
                y=selected_metric,
                title=f"{get_human_readable_name(selected_metric)} by Cluster",
                labels={
                    'cluster_name': 'Cluster',
                    selected_metric: get_human_readable_name(selected_metric)
                }
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