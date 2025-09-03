"""Streamlit app for visualizing alej_v2 metrics from database."""

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
    """Load posts with alej_v2_metrics and cluster information from database."""
    # Use SQLAlchemy engine for pandas compatibility
    engine = create_engine(DATABASE_URL)
    
    # Handle special case for 2-cluster (ea_classification)
    if cluster_n == 2:
        cluster_col = "ea_classification"
        cluster_name_col = "ea_classification"
    else:
        cluster_col = f"ea_cluster_{cluster_n}"
        cluster_name_col = f"ea_cluster_{cluster_n}_name"
    
    query = f"""
    SELECT 
        post_id,
        title,
        author_display_name as author,
        base_score,
        posted_at,
        {cluster_col} as cluster_id,
        {cluster_name_col} as cluster_name,
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
    ORDER BY posted_at
    """
    
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    
    # Convert posted_at to datetime
    df['posted_at'] = pd.to_datetime(df['posted_at'])
    
    return df

def get_human_readable_name(metric_name: str) -> str:
    """Convert v2 metric column name to human readable format."""
    mapping = {
        'value_score': 'Value',
        'cooperativeness_score': 'Cooperativeness',
        'clarity_score': 'Clarity',
        'precision_score': 'Precision',
        'ea_fame_score': 'Author EA Fame',
        'external_validation_score': 'External Validation',
        'robustness_score': 'Robustness',
        'reasoning_quality_score': 'Reasoning Quality',
        'title_clickability_score': 'Title Clickability',
        'controversy_temperature_score': 'Controversy Temperature',
        'empirical_evidence_quality_score': 'Empirical Evidence Quality',
        'base_score': 'Post Karma'
    }
    return mapping.get(metric_name, metric_name.replace('_', ' ').title())

# Removed get_available_cluster_ns function - using hardcoded cluster options

# Define metric columns at module level for reuse
METRIC_COLUMNS = [
    'value_score', 'cooperativeness_score', 'clarity_score', 'precision_score',
    'ea_fame_score', 'external_validation_score', 'robustness_score',
    'reasoning_quality_score', 'title_clickability_score', 
    'controversy_temperature_score', 'empirical_evidence_quality_score'
]

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between all metrics and base_score."""
    
    correlations = []
    
    for col in METRIC_COLUMNS:
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
    st.caption("Analyzing alej_v2 metrics stored in the database")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Cluster size selector - hardcoded options
    available_clusters = [2, 5, 12, 30, 60]
    cluster_n = st.sidebar.selectbox(
        "Cluster Size (N)",
        available_clusters,
        index=2,  # Default to 12 clusters
        help="Number of clusters to use for grouping (2=EA classification, others=topic clusters)"
    )
    
    # Load data
    df = load_metrics_data(cluster_n)
    
    if df.empty:
        st.warning("No posts with alej_v2_metrics found in database")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Correlations", 
        "👥 Author Leaderboards", 
        "🎯 Cluster Leaderboards",
        "📈 Distributions",
        "⏱️ Timeseries"
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
            display_df = corr_df[['Metric', 'Correlation', 'P-value']].copy()
            display_df['Correlation'] = display_df['Correlation'].round(3)
            display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.4f} ✓" if x < 0.05 else f"{x:.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Metric-to-Metric Correlation Matrix (moved up)
            st.subheader("Metric-to-Metric Correlations")
            st.caption("Correlations between all pairs of metrics (excluding post karma)")
            
            # Get only the metric columns (exclude base_score, metadata)
            metric_only_cols = [col for col in METRIC_COLUMNS if col in df.columns]
            
            if len(metric_only_cols) >= 2:
                # Group metrics by category
                epistemic_virtues = [
                    'value_score', 'cooperativeness_score', 'clarity_score', 
                    'precision_score', 'external_validation_score', 'robustness_score', 
                    'reasoning_quality_score', 'empirical_evidence_quality_score'
                ]
                author_metrics = ['ea_fame_score']
                engagement_metrics = ['title_clickability_score', 'controversy_temperature_score']
                
                # Create ordered list of metrics (only those present in data)
                ordered_metrics = []
                for group in [epistemic_virtues, author_metrics, engagement_metrics]:
                    for metric in group:
                        if metric in metric_only_cols:
                            ordered_metrics.append(metric)
                
                # Calculate correlation matrix and reorder
                import numpy as np
                corr_matrix = df[metric_only_cols].corr()
                corr_matrix_ordered = corr_matrix.loc[ordered_metrics, ordered_metrics]
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix_ordered.values,
                    x=[get_human_readable_name(col) for col in corr_matrix_ordered.columns],
                    y=[get_human_readable_name(col) for col in corr_matrix_ordered.index],
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1,
                    title="Correlation Matrix: Metrics vs Metrics",
                    aspect="auto"
                )
                
                # Add correlation values as text
                for i in range(len(corr_matrix_ordered.index)):
                    for j in range(len(corr_matrix_ordered.columns)):
                        fig.add_annotation(
                            x=j, y=i,
                            text=f"{corr_matrix_ordered.iloc[i, j]:.2f}",
                            showarrow=False,
                            font=dict(color="white" if abs(corr_matrix_ordered.iloc[i, j]) > 0.5 else "black")
                        )
                
                fig.update_layout(
                    height=max(400, len(metric_only_cols) * 40),
                    xaxis_title="",
                    yaxis_title=""
                )
                fig.update_xaxes(side="bottom")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 metrics to show correlation matrix")
            
            # Top 3 scatter plots (moved to bottom)
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
            top_metrics = ['value_score', 'clarity_score', 'reasoning_quality_score']
        
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
        
        # Show all available metrics
        for metric in METRIC_COLUMNS:
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
            'base_score', 'value_score', 'cooperativeness_score', 'clarity_score', 'precision_score',
            'ea_fame_score', 'external_validation_score', 'robustness_score',
            'reasoning_quality_score', 'title_clickability_score', 
            'controversy_temperature_score', 'empirical_evidence_quality_score'
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
        selected_metric = st.selectbox(
            "Select metric",
            [m for m in METRIC_COLUMNS if m in df.columns],
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
    
    with tab5:
        st.header("Timeseries Analysis")
        st.caption("Rolling averages with standard deviation bands")
        
        # Configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            rolling_days = st.number_input(
                "Rolling window (days)", 
                min_value=7, 
                max_value=365, 
                value=30,
                help="Number of days for rolling average calculation"
            )
        with col2:
            available_metrics = ['base_score'] + [m for m in METRIC_COLUMNS if m in df.columns]
            selected_metrics = st.multiselect(
                "Select metrics to display",
                available_metrics,
                default=available_metrics,  # Show all metrics by default
                format_func=get_human_readable_name
            )
        with col3:
            # Cluster display option
            show_by_clusters = st.checkbox("Show by clusters", value=False)
            n_clusters = None
            if show_by_clusters:
                # Get actual number of unique clusters in current data
                actual_cluster_count = len(df['cluster_name'].unique())
                
                # Create options list - only show numbers up to actual cluster count
                cluster_options = list(range(2, min(actual_cluster_count + 1, 11)))  # Cap at 10 to keep reasonable
                
                if cluster_options:
                    n_clusters = st.selectbox(
                        "Number of clusters",
                        cluster_options,
                        index=min(3, len(cluster_options) - 1) if len(cluster_options) > 3 else 0,  # Default to 5 if available
                        help="Number of top clusters to display"
                    )
                else:
                    st.warning("Not enough clusters available for display")
                    n_clusters = None
        
        if selected_metrics and len(df) > 0:
            # Add post count timeseries first
            st.subheader("📈 Graded Posts Per Day Over Time")
            
            # Create post count plot
            import plotly.graph_objects as go
            rolling_window = f'{rolling_days}D'
            
            if show_by_clusters and n_clusters:
                # Get top N clusters by post count
                top_clusters = df['cluster_name'].value_counts().head(n_clusters).index.tolist()
                
                fig = go.Figure()
                colors = px.colors.qualitative.Set1[:n_clusters]
                
                for i, cluster in enumerate(top_clusters):
                    cluster_df = df[df['cluster_name'] == cluster]
                    daily_counts = cluster_df.set_index('posted_at').resample('D').size()
                    daily_counts = daily_counts.reindex(pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq='D')).fillna(0)
                    
                    # Calculate rolling average (no std dev bands for clusters)
                    rolling_count_mean = daily_counts.rolling(window=rolling_window, min_periods=1).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=rolling_count_mean.index,
                        y=rolling_count_mean.values,
                        mode='lines',
                        name=f'{cluster} ({len(cluster_df)} posts)',
                        line=dict(color=colors[i], width=2)
                    ))
                
                fig.update_layout(
                    title=f"Graded Posts Per Day Over Time - Top {n_clusters} Clusters",
                    xaxis_title="Date",
                    yaxis_title="Number of Graded Posts",
                    hovermode='x unified'
                )
            else:
                # Show overall post counts with std dev bands
                daily_counts = df.set_index('posted_at').resample('D').size()
                daily_counts = daily_counts.reindex(pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq='D')).fillna(0)
                
                rolling_count_mean = daily_counts.rolling(window=rolling_window, min_periods=1).mean()
                rolling_count_std = daily_counts.rolling(window=rolling_window, min_periods=1).std()
                
                fig = go.Figure()
                
                # Add daily counts as bars (lighter)
                fig.add_trace(go.Bar(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    name='Daily Graded Posts',
                    opacity=0.3,
                    marker_color='lightblue'
                ))
                
                # Add rolling average line
                fig.add_trace(go.Scatter(
                    x=rolling_count_mean.index,
                    y=rolling_count_mean.values,
                    mode='lines',
                    name=f'{rolling_days}-day Rolling Average',
                    line=dict(color='blue', width=2)
                ))
                
                # Add standard deviation bands
                upper_band = rolling_count_mean + rolling_count_std
                lower_band = rolling_count_mean - rolling_count_std
                
                fig.add_trace(go.Scatter(
                    x=rolling_count_mean.index,
                    y=upper_band.values,
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=rolling_count_mean.index,
                    y=lower_band.values,
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='±1 Std Dev',
                    fillcolor='rgba(0,100,80,0.2)'
                ))
                
                fig.update_layout(
                    title="Graded Posts Per Day Over Time",
                    xaxis_title="Date",
                    yaxis_title="Number of Graded Posts",
                    hovermode='x unified'
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Now show metric timeseries
            for metric in selected_metrics:
                st.subheader(f"📈 {get_human_readable_name(metric)} Over Time")
                
                fig = go.Figure()
                
                if show_by_clusters and n_clusters:
                    # Show each cluster as separate line (no std dev bands)
                    colors = px.colors.qualitative.Set1[:n_clusters]
                    
                    for i, cluster in enumerate(top_clusters):
                        cluster_df = df[df['cluster_name'] == cluster]
                        ts_df = cluster_df[['posted_at', metric]].dropna()
                        
                        if len(ts_df) > 0:
                            ts_df.set_index('posted_at', inplace=True)
                            ts_df.sort_index(inplace=True)
                            
                            # Calculate rolling average only
                            rolling_mean = ts_df[metric].rolling(window=rolling_window, min_periods=1).mean()
                            
                            fig.add_trace(go.Scatter(
                                x=rolling_mean.index,
                                y=rolling_mean.values,
                                mode='lines',
                                name=f'{cluster}',
                                line=dict(color=colors[i], width=2)
                            ))
                    
                    fig.update_layout(
                        title=f"{get_human_readable_name(metric)} Over Time - By Cluster",
                        xaxis_title="Date",
                        yaxis_title=get_human_readable_name(metric),
                        hovermode='x unified'
                    )
                else:
                    # Show overall with std dev bands
                    ts_df = df[['posted_at', metric]].dropna()
                    
                    if len(ts_df) > 0:
                        ts_df.set_index('posted_at', inplace=True)
                        ts_df.sort_index(inplace=True)
                        
                        # Calculate rolling statistics
                        rolling_mean = ts_df[metric].rolling(window=rolling_window, min_periods=1).mean()
                        rolling_std = ts_df[metric].rolling(window=rolling_window, min_periods=1).std()
                        
                        # Add individual points (optional)
                        fig.add_trace(go.Scatter(
                            x=ts_df.index,
                            y=ts_df[metric],
                            mode='markers',
                            name='Individual Posts',
                            marker=dict(size=3, opacity=0.3),
                            showlegend=False
                        ))
                        
                        # Add rolling average line
                        fig.add_trace(go.Scatter(
                            x=rolling_mean.index,
                            y=rolling_mean.values,
                            mode='lines',
                            name=f'{rolling_days}-day Rolling Average',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Add standard deviation bands
                        upper_band = rolling_mean + rolling_std
                        lower_band = rolling_mean - rolling_std
                        
                        # Upper band
                        fig.add_trace(go.Scatter(
                            x=rolling_mean.index,
                            y=upper_band.values,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Lower band (fill between)
                        fig.add_trace(go.Scatter(
                            x=rolling_mean.index,
                            y=lower_band.values,
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(0,100,80,0.2)',
                            fill='tonexty',
                            name='±1 Std Dev',
                            hoverinfo='skip'
                        ))
                        
                        fig.update_layout(
                            title=f'{get_human_readable_name(metric)} - {rolling_days} Day Rolling Average',
                            xaxis_title='Date',
                            yaxis_title=get_human_readable_name(metric),
                            hovermode='x unified',
                            height=400
                        )
                        
                        # Set y-axis range for LLM metrics (1-10 scale)
                        if metric != 'base_score':
                            fig.update_yaxes(range=[0, 10])
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one metric to display.")

if __name__ == "__main__":
    main()