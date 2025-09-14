"""Streamlit app for exploring V3 metrics with V2 title clickability."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from scipy import stats
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

st.set_page_config(page_title="EA Forum V3 Metrics Explorer", layout="wide")

def get_connection():
    """Create a new database connection."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# Cache data loading
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_all_metrics_data() -> pd.DataFrame:
    """Load posts with V3 metrics and V2 title clickability from database."""
    engine = create_engine(DATABASE_URL)
    
    query = """
    SELECT 
        post_id,
        title,
        page_url,
        author_display_name as author,
        base_score,
        posted_at,
        word_count,
        comment_count,
        ea_cluster_12 as cluster_id,
        ea_cluster_12_name as cluster_name,
        -- Extract V3 metrics from JSONB
        (alej_v3_metrics->>'value_score')::float as value_score,
        (alej_v3_metrics->>'reasoning_quality_score')::float as reasoning_quality_score,
        (alej_v3_metrics->>'cooperativeness_score')::float as cooperativeness_score,
        (alej_v3_metrics->>'precision_score')::float as precision_score,
        (alej_v3_metrics->>'empirical_evidence_quality_score')::float as empirical_evidence_quality_score,
        (alej_v3_metrics->>'memetic_potential_score')::float as memetic_potential_score,
        (alej_v3_metrics->>'ea_fame_score')::float as ea_fame_score,
        (alej_v3_metrics->>'controversy_temperature_score')::float as controversy_temperature_score,
        (alej_v3_metrics->>'overall_epistemic_quality_score')::float as overall_epistemic_quality_score,
        (alej_v3_metrics->>'predicted_karma_score')::float as predicted_karma_score,
        -- Extract title clickability from V2 metrics
        (alej_v2_metrics->>'title_clickability_score')::float as title_clickability_score
    FROM fellowship_mvp
    WHERE alej_v3_metrics IS NOT NULL
    ORDER BY posted_at DESC
    """
    
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    
    # Convert posted_at to datetime
    df['posted_at'] = pd.to_datetime(df['posted_at'])
    
    # Add year and month columns for filtering
    df['year'] = df['posted_at'].dt.year
    df['month'] = df['posted_at'].dt.month
    df['year_month'] = df['posted_at'].dt.to_period('M').astype(str)
    
    return df

def get_human_readable_name(metric_name: str) -> str:
    """Convert metric column name to human readable format."""
    mapping = {
        'value_score': 'Value',
        'reasoning_quality_score': 'Reasoning Quality',
        'cooperativeness_score': 'Cooperativeness',
        'precision_score': 'Precision',
        'empirical_evidence_quality_score': 'Empirical Evidence',
        'memetic_potential_score': 'Memetic Potential',
        'ea_fame_score': 'Author Fame',
        'controversy_temperature_score': 'Controversy',
        'overall_epistemic_quality_score': 'Overall Epistemic Quality',
        'predicted_karma_score': 'Predicted Karma',
        'title_clickability_score': 'Title Clickability',
        'base_score': 'Actual Karma',
        'word_count': 'Word Count',
        'comment_count': 'Comments'
    }
    return mapping.get(metric_name, metric_name.replace('_', ' ').title())

# Define available metrics for sorting (ordered by priority)
METRIC_COLUMNS = [
    'base_score',  # Actual karma first
    'overall_epistemic_quality_score',
    'predicted_karma_score',  # Add predicted karma back
    'value_score',
    'reasoning_quality_score',
    'cooperativeness_score',
    'precision_score',
    'empirical_evidence_quality_score',
    'memetic_potential_score',
    'ea_fame_score',
    'controversy_temperature_score',
    'title_clickability_score'
    # word_count and comment_count removed
]

def author_explore_tab(df: pd.DataFrame):
    """Author exploration tab - show histograms with author's average."""
    # Get list of unique authors
    authors = sorted(df['author'].dropna().unique())
    
    # Author search/selection with default to Joe_Carlsmith
    default_author = "Joe_Carlsmith" if "Joe_Carlsmith" in authors else None
    default_index = authors.index(default_author) if default_author else 0
    
    selected_author = st.selectbox(
        "Select Author",
        options=authors,
        help="Choose an author to see their performance relative to all posts",
        index=default_index,
        placeholder="Type to search authors..."
    )
    
    if selected_author:
        # Filter for selected author
        author_df = df[df['author'] == selected_author].copy()
        
        if author_df.empty:
            st.warning(f"No posts found for author: {selected_author}")
            return
        
        st.subheader(f"{selected_author} vs All Posts")
        st.caption(f"Red line shows {selected_author}'s average for each metric • {len(author_df)} graded posts")
        
        # Create histograms for each metric
        for metric in METRIC_COLUMNS:
            if metric in df.columns and not df[metric].isna().all():
                # Calculate averages
                author_avg = author_df[metric].mean()
                overall_avg = df[metric].mean()
                
                if pd.isna(author_avg):
                    continue  # Skip if author has no data for this metric
                
                # Create histogram with tighter bins for more matplotlib-like appearance
                fig = go.Figure()
                if metric in ['base_score', 'predicted_karma_score', 'word_count', 'comment_count']:
                    nbins= 30
                else:
                    nbins= 10
                # Add histogram of all posts with touching bars
                fig.add_trace(go.Histogram(
                    x=df[metric].dropna(),
                    nbinsx=nbins,
                    name="All Posts",
                    opacity=0.7
                ))
                
                # Add vertical line for overall average (dotted)
                fig.add_vline(
                    x=overall_avg,
                    line_dash="dot",
                    line_color="white",
                    line_width=2,
                    annotation_text=f"Overall Avg: {overall_avg:.2f}",
                    annotation_position="bottom"
                )
                
                fig.add_vline(
                    x=author_avg,
                    line_dash="solid",
                    line_color="red" if author_avg < overall_avg else "green",
                    line_width=3,
                    annotation_text=f"{selected_author} Avg: {author_avg:.2f}",
                    # annotation_position="top left"
                )
                
                fig.update_layout(
                    title=dict(
                        text=f"{get_human_readable_name(metric)}",
                        font=dict(size=16)
                    ),
                    xaxis_title=get_human_readable_name(metric),
                    yaxis_title="Posts",
                    showlegend=False,
                    height=250,
                    margin=dict(l=50, r=50, t=60, b=40),
                    bargap=0.05,  # Small gap between bars for cleaner look
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=0.5
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=0.5
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)

def cluster_explore_tab(df: pd.DataFrame):
    """Cluster exploration tab - show histograms with cluster's average."""
    # Get list of unique clusters
    clusters = sorted(df['cluster_name'].dropna().unique())
    
    # Cluster selection with default to Technical AI Safety & Forecasting
    default_cluster = "Technical AI Safety & Forecasting" if "Technical AI Safety & Forecasting" in clusters else None
    default_index = clusters.index(default_cluster) if default_cluster else 0
    
    selected_cluster = st.selectbox(
        "Select Cluster",
        options=clusters,
        help="Choose a cluster to see its performance relative to all posts",
        index=default_index,
        placeholder="Type to search clusters..."
    )
    
    if selected_cluster:
        # Filter for selected cluster
        cluster_df = df[df['cluster_name'] == selected_cluster].copy()
        
        if cluster_df.empty:
            st.warning(f"No posts found for cluster: {selected_cluster}")
            return
        
        st.subheader(f"{selected_cluster} vs All Posts")
        st.caption(f"Red line shows {selected_cluster}'s average for each metric • {len(cluster_df)} graded posts")
        
        # Create histograms for each metric
        for metric in METRIC_COLUMNS:
            if metric in df.columns and not df[metric].isna().all():
                # Calculate averages
                cluster_avg = cluster_df[metric].mean()
                overall_avg = df[metric].mean()
                
                if pd.isna(cluster_avg):
                    continue  # Skip if cluster has no data for this metric
                
                # Create histogram with tighter bins for more matplotlib-like appearance
                fig = go.Figure()
                if metric in ['base_score', 'predicted_karma_score', 'word_count', 'comment_count']:
                    nbins= 30
                else:
                    nbins= 10
                # Add histogram of all posts with touching bars
                fig.add_trace(go.Histogram(
                    x=df[metric].dropna(),
                    nbinsx=nbins,
                    name="All Posts",
                    opacity=0.7
                ))
                
                # Add vertical line for overall average (dotted)
                fig.add_vline(
                    x=overall_avg,
                    line_dash="dot",
                    line_color="white",
                    line_width=2,
                    annotation_text=f"Overall Avg: {overall_avg:.2f}",
                    annotation_position="bottom"
                )
                
                fig.add_vline(
                    x=cluster_avg,
                    line_dash="solid",
                    line_color="red" if cluster_avg < overall_avg else "green",
                    line_width=3,
                    annotation_text=f"{selected_cluster} Avg: {cluster_avg:.2f}",
                )
                
                fig.update_layout(
                    title=dict(
                        text=f"{get_human_readable_name(metric)}",
                        font=dict(size=16)
                    ),
                    xaxis_title=get_human_readable_name(metric),
                    yaxis_title="Posts",
                    showlegend=False,
                    height=250,
                    margin=dict(l=50, r=50, t=60, b=40),
                    bargap=0.05,  # Small gap between bars for cleaner look
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=0.5
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=0.5
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)

def bootstrap_spearman_ci(x, y, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for Spearman correlation."""
    correlations = []
    n = len(x)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x.iloc[indices]
        y_boot = y.iloc[indices]
        
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

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between all metrics and base_score with bootstrap CIs."""
    from scipy import stats
    
    correlations = []
    
    for col in METRIC_COLUMNS:
        if col in df.columns:
            valid_data = df[[col, 'base_score']].dropna()
            
            if len(valid_data) > 1:
                correlation, p_value = stats.spearmanr(valid_data[col], valid_data['base_score'])
                # Handle case where correlation might be an array
                if np.isscalar(correlation) and not np.isnan(correlation):
                    # Calculate bootstrap confidence interval
                    ci_lower, ci_upper = bootstrap_spearman_ci(
                        valid_data[col], valid_data['base_score']
                    )
                    
                    correlations.append({
                        'Metric': get_human_readable_name(col),
                        'Correlation': correlation,
                        'N': len(valid_data),
                        '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "N/A",
                        'metric_col': col  # Keep original column name
                    })
    
    corr_df = pd.DataFrame(correlations)
    if not corr_df.empty:
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    
    return corr_df

def correlation_analysis_tab(df: pd.DataFrame):
    """Correlation analysis tab - show correlations with karma and intra-metric correlations."""
    st.header("📊 Correlation Analysis")
    
    # Toggle for correlation target
    correlation_target = st.radio(
        "Show correlations with:",
        ["Post Karma", "Number of Comments"],
        horizontal=True
    )
    
    if correlation_target == "Post Karma":
        # Calculate correlations with karma
        corr_df = calculate_correlations(df)
        
        if not corr_df.empty:
            # Bar chart of correlations with karma
            fig = px.bar(
                corr_df,
                x='Correlation',
                y='Metric',
                orientation='h',
                color='Correlation',
                color_continuous_scale='RdBu',
                range_color=[-1, 1],
                title="Spearman Correlation with Post Karma"
            )
            fig.update_layout(height=max(400, len(corr_df) * 30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation table
            display_df = corr_df[['Metric', 'Correlation', '95% CI', 'N']].copy()
            display_df['Correlation'] = display_df['Correlation'].round(3)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    else:  # Number of Comments
        # Calculate correlations with comments
        comment_correlations = []
        
        for col in METRIC_COLUMNS:
            if col in df.columns and col != 'base_score':  # Exclude actual karma from comment correlations
                valid_data = df[[col, 'comment_count']].dropna()
                
                if len(valid_data) > 1:
                    correlation, p_value = stats.spearmanr(valid_data[col], valid_data['comment_count'])
                    # Handle case where correlation might be an array
                    if np.isscalar(correlation) and not np.isnan(correlation):
                        # Calculate bootstrap confidence interval
                        ci_lower, ci_upper = bootstrap_spearman_ci(
                            valid_data[col], valid_data['comment_count']
                        )
                        
                        comment_correlations.append({
                            'Metric': get_human_readable_name(col),
                            'Correlation': correlation,
                            'N': len(valid_data),
                            '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "N/A",
                            'metric_col': col  # Keep original column name
                        })
        
        comment_corr_df = pd.DataFrame(comment_correlations)
        if not comment_corr_df.empty:
            comment_corr_df = comment_corr_df.sort_values('Correlation', key=abs, ascending=False)
            
            # Bar chart of correlations with comments
            fig = px.bar(
                comment_corr_df,
                x='Correlation',
                y='Metric',
                orientation='h',
                color='Correlation',
                color_continuous_scale='RdBu',
                range_color=[-1, 1],
                title="Spearman Correlation with Number of Comments"
            )
            fig.update_layout(height=max(400, len(comment_corr_df) * 30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation table
            display_df = comment_corr_df[['Metric', 'Correlation', '95% CI', 'N']].copy()
            display_df['Correlation'] = display_df['Correlation'].round(3)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Metric-to-Metric Correlation Matrix (outside of columns)
    st.subheader("Metric-to-Metric Spearman Correlations")
    
    # Get only the metric columns (exclude base_score, metadata)
    metric_only_cols = [col for col in METRIC_COLUMNS if col in df.columns]
    
    if len(metric_only_cols) >= 2:
        # Group metrics by category in requested order
        synthesis_metrics = ['predicted_karma_score', 'overall_epistemic_quality_score']
        epistemic_virtues = [
            'value_score', 'reasoning_quality_score', 'cooperativeness_score', 
            'precision_score', 'empirical_evidence_quality_score'
        ]
        engagement_metrics = ['memetic_potential_score']
        author_metrics = ['ea_fame_score']
        social_metrics = ['controversy_temperature_score']
        external_metrics = ['title_clickability_score']
        
        # Create ordered list of metrics (only those present in data)
        ordered_metrics = []
        for group in [synthesis_metrics, epistemic_virtues, engagement_metrics, author_metrics, social_metrics, external_metrics]:
            for metric in group:
                if metric in metric_only_cols:
                    ordered_metrics.append(metric)
        
        # Calculate Spearman correlation matrix and reorder
        corr_matrix = df[metric_only_cols].corr(method='spearman')
        corr_matrix_ordered = corr_matrix.loc[ordered_metrics, ordered_metrics]
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix_ordered.values,
            x=[get_human_readable_name(col) for col in corr_matrix_ordered.columns],
            y=[get_human_readable_name(col) for col in corr_matrix_ordered.index],
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1,
            # title="Spearman Correlation Matrix: Metrics vs Metrics",
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

def author_leaderboard_tab(df: pd.DataFrame):
    """Author leaderboard tab - show top authors by selected metric."""
    st.header("🏆 Author Leaderboard")
    
    # Get list of available metrics for selection (exclude base_score from dropdown)
    available_metrics = [col for col in METRIC_COLUMNS if col in df.columns and col != 'base_score']
    metric_names = {col: get_human_readable_name(col) for col in available_metrics}
    
    # Create two columns for controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Metric selection dropdown (default to overall epistemic quality)
        default_metric = 'overall_epistemic_quality_score' if 'overall_epistemic_quality_score' in available_metrics else available_metrics[0]
        selected_metric = st.selectbox(
            "Select metric to rank by:",
            available_metrics,
            format_func=lambda x: metric_names[x],
            index=available_metrics.index(default_metric)
        )
    
    with col2:
        # Minimum posts threshold
        min_posts = st.number_input(
            "Minimum posts required:",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
    
    # Calculate author statistics
    author_stats = df.groupby('author').agg({
        selected_metric: 'mean',
        'post_id': 'count'
    }).rename(columns={'post_id': 'post_count'})
    
    # Filter by minimum posts
    author_stats_filtered = author_stats[author_stats['post_count'] >= min_posts].copy()
    
    # Display number of qualifying authors and average post score
    if len(author_stats_filtered) > 0:
        avg_post_score = df[df['author'].isin(author_stats_filtered.index)][selected_metric].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**{len(author_stats_filtered)}** authors meet the threshold of {min_posts}+ graded posts")
        with col2:
            st.metric(f"Average Post Score ({metric_names[selected_metric]})", f"{avg_post_score:.2f}")
        
        # Sort by selected metric
        author_stats_filtered = author_stats_filtered.sort_values(selected_metric, ascending=False)
        
        # Add rank column
        author_stats_filtered['rank'] = range(1, len(author_stats_filtered) + 1)
        
        # Reset index to make author a column
        author_stats_filtered = author_stats_filtered.reset_index()
        
        # Rename columns for display
        display_df = author_stats_filtered[['rank', 'author', selected_metric, 'post_count']].copy()
        display_df.columns = ['Rank', 'Author', f'Avg {metric_names[selected_metric]}', 'Posts']
        
        # Round the metric value
        display_df[f'Avg {metric_names[selected_metric]}'] = display_df[f'Avg {metric_names[selected_metric]}'].round(2)
        
        # Display the leaderboard with scrolling enabled (always show all)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=600
        )
        
    else:
        st.warning(f"No authors have {min_posts} or more posts with {metric_names[selected_metric]} scores")

def hidden_gems_tab(df: pd.DataFrame):
    """Hidden gems tab - find posts with high epistemic quality but low karma."""
    
    # Check if we have the required data
    if 'overall_epistemic_quality_score' not in df.columns or df['overall_epistemic_quality_score'].isna().all():
        st.error("No epistemic quality data available in the dataset.")
        return
    
    # Filter to posts with both epistemic quality and karma data
    gems_df = df[['post_id', 'title', 'page_url', 'author', 'base_score', 'predicted_karma_score', 
                  'overall_epistemic_quality_score']].dropna(subset=['overall_epistemic_quality_score', 'base_score'])
    
    if gems_df.empty:
        st.warning("No posts with both epistemic quality and karma scores found.")
        return
    
    
    # Hidden Gems Section
    st.subheader("💎 Hidden Gems (?)")
    st.caption("Posts with high epistemic quality but low actual karma")
    
    # Controls for hidden gems
    col1, col2 = st.columns(2)
    
    with col1:
        epistemic_percentile = st.number_input(
            "Min Epistemic Quality Percentile",
            min_value=0, max_value=100, value=90, step=1,
            help="Posts must be in top X% for epistemic quality"
        )
    
    with col2:
        karma_percentile = st.number_input(
            "Max Karma Percentile", 
            min_value=0, max_value=100, value=50, step=1,
            help="Posts must be in bottom X% for actual karma"
        )
    
    # Calculate thresholds
    epistemic_threshold = gems_df['overall_epistemic_quality_score'].quantile(epistemic_percentile / 100)
    karma_threshold = gems_df['base_score'].quantile(karma_percentile / 100)
    
    # Filter for hidden gems
    hidden_gems_mask = (
        (gems_df['overall_epistemic_quality_score'] >= epistemic_threshold) &
        (gems_df['base_score'] <= karma_threshold)
    )
    
    hidden_gems = gems_df[hidden_gems_mask].copy()
    
    # Sort by epistemic quality descending
    hidden_gems = hidden_gems.sort_values('overall_epistemic_quality_score', ascending=False)
    
    st.info(f"Found **{len(hidden_gems)}** hidden gems (top {100-epistemic_percentile}% epistemic quality, bottom {karma_percentile}% karma)")
    
    if len(hidden_gems) > 0:
        # Display hidden gems with clickable links
        gems_display = hidden_gems[['title', 'page_url', 'author', 'overall_epistemic_quality_score', 
                                   'base_score', 'predicted_karma_score']].copy()
        
        # Rename columns for display  
        gems_display.columns = ['Title', 'URL', 'Author', 'Epistemic Quality', 'Actual Karma', 'Predicted Karma']
        gems_display['Epistemic Quality'] = gems_display['Epistemic Quality'].round(2)
        gems_display['Predicted Karma'] = gems_display['Predicted Karma'].round(1)
        
        st.dataframe(gems_display, use_container_width=True, hide_index=True, height=400, column_config={
            "URL": st.column_config.LinkColumn("URL", help="Click to view the post", display_text="🔗 View Post")
        })
        
        # Summary stats for hidden gems
        st.write("**Hidden Gems Statistics**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_epistemic = hidden_gems['overall_epistemic_quality_score'].mean()
            st.metric("Avg Epistemic Quality", f"{avg_epistemic:.2f}")
        
        with col2:
            avg_actual_karma = hidden_gems['base_score'].mean()
            st.metric("Avg Actual Karma", f"{avg_actual_karma:.1f}")
        
        with col3:
            avg_predicted_karma = hidden_gems['predicted_karma_score'].mean()
            st.metric("Avg Predicted Karma", f"{avg_predicted_karma:.1f}")
    else:
        st.warning("No hidden gems found with the current criteria. Try adjusting the thresholds.")

def main():
    st.title("🎯 EA Forum Metrics Explorer")
    
    # Load data
    df = load_all_metrics_data()
    
    if df.empty:
        st.warning("No posts with metrics found in database")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 Author Explorer",
        "🏷️ Cluster Explorer",
        "📊 Correlations",
        "🏆 Author Leaderboard",
        "💎 Hidden Gems"
    ])
    
    with tab1:
        author_explore_tab(df)
    
    with tab2:
        cluster_explore_tab(df)
    
    with tab3:
        correlation_analysis_tab(df)
    
    with tab4:
        author_leaderboard_tab(df)
    
    with tab5:
        hidden_gems_tab(df)

if __name__ == "__main__":
    main()