"""Streamlit app for visualizing claim metrics."""

import streamlit as st
import json
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
from typing import List

from db import get_representative_posts
from models import Post
from metric_models import get_metric_score_fields

# Get version ID from command line args if provided
if len(sys.argv) > 1:
    DEFAULT_VERSION_ID = sys.argv[1]
else:
    DEFAULT_VERSION_ID = "v1"

st.set_page_config(page_title="EA Forum Claim Metrics", layout="wide")


def discover_available_metrics(version_id: str) -> List[str]:
    """Discover which metrics are available in the data for a given version."""
    metrics_dir = Path(f"data/metrics/{version_id}")
    
    if not metrics_dir.exists():
        return []
    
    available_metrics = set()
    
    # Sample first file to see what metrics are available
    for metrics_file in metrics_dir.glob("*.json"):
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            
        # Check all claims for available metrics
        for claim_id, claim_metrics in metrics_data.items():
            available_metrics.update(claim_metrics.keys())
        
        # Just check first file for efficiency (assuming all files have same metrics)
        break
    
    return sorted(list(available_metrics))


def load_metrics(version_id: str, metric_names: List[str]):
    """Load metrics data for a given version."""
    metrics_dir = Path(f"data/metrics/{version_id}")
    
    if not metrics_dir.exists():
        return None
    
    all_data = []
    
    for metrics_file in metrics_dir.glob("*.json"):
        post_id = metrics_file.stem
        
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        for claim_id, claim_metrics in metrics_data.items():
            row = {
                "post_id": post_id,
                "claim_id": claim_id
            }
            
            for metric_name in metric_names:
                if metric_name in claim_metrics:
                    metric_data = claim_metrics[metric_name]
                    # Add all fields from the metric
                    for key, value in metric_data.items():
                        if key not in ["post_id", "claim_id"]:  # Skip duplicates
                            row[f"{metric_name}_{key}"] = value
            
            all_data.append(row)
    
    return pd.DataFrame(all_data)


def main():
    st.title("EA Forum Claim Metrics Analysis")
    
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
        st.warning(f"No metrics found for version '{version_id}'. Please check the version ID or run the pipeline first.")
        return
    
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics",
        options=available_metrics,
        default=available_metrics
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric")
        return
    
    # Load data
    df = load_metrics(version_id, selected_metrics)
    
    if df is None or df.empty:
        st.error(f"No data found for version '{version_id}'")
        return
    
    # Get posts and their base scores - use enough posts to cover all metrics data
    posts = get_representative_posts(20)
    post_scores = {p.post_id: p.base_score for p in posts}
    df["base_score"] = df["post_id"].map(post_scores)
    
    st.success(f"Loaded {len(df)} claim metrics from {df['post_id'].nunique()} posts")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Scatter Plots", "Correlations", "Raw Data"])
    
    with tab1:
        st.header("Metric Distributions")
        
        # Create distribution plots for each metric
        for metric in selected_metrics:
            st.subheader(f"{metric} Distribution")
            
            # Check which fields exist for this metric
            metric_fields = [col for col in df.columns if col.startswith(f"{metric}_")]
            numeric_fields = [f for f in metric_fields if not f.endswith("_explanation")]
            
            if numeric_fields:
                cols = st.columns(len(numeric_fields))
                for i, field in enumerate(numeric_fields):
                    with cols[i]:
                        fig = px.histogram(
                            df,
                            x=field,
                            title=field.replace(f"{metric}_", "").replace("_", " ").title(),
                            nbins=10,
                            labels={field: "Score (1-10)"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Metrics vs Base Score")
        
        # Create scatter plots
        for metric in selected_metrics:
            st.subheader(f"{metric} vs Base Score")
            
            metric_fields = [col for col in df.columns if col.startswith(f"{metric}_")]
            numeric_fields = [f for f in metric_fields if not f.endswith("_explanation")]
            
            if numeric_fields and "base_score" in df.columns:
                cols = st.columns(len(numeric_fields))
                for i, field in enumerate(numeric_fields):
                    with cols[i]:
                        fig = px.scatter(
                            df,
                            x="base_score",
                            y=field,
                            title=field.replace(f"{metric}_", "").replace("_", " ").title(),
                            labels={
                                "base_score": "Post Base Score",
                                field: "Metric Score (1-10)"
                            },
                            hover_data=["post_id", "claim_id"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Correlations with Base Score")
        
        # Get all score fields from the metric models
        metric_columns = []
        for metric_name in selected_metrics:
            score_fields = get_metric_score_fields(metric_name)
            # Add the metric prefix to match column names in DataFrame
            for field in score_fields:
                col_name = f"{metric_name}_{field}"
                if col_name in df.columns:
                    metric_columns.append(col_name)
        
        if metric_columns and "base_score" in df.columns:
            # Calculate correlations
            correlations = []
            for col in metric_columns:
                if col in df.columns:
                    corr = df[col].corr(df["base_score"])
                    # Extract metric name and format field name same as distributions tab
                    metric_name = col.split("_")[0]
                    field_name = col.replace(f"{metric_name}_", "").replace("_", " ").title()
                    display_name = f"{metric_name} - {field_name}"
                    correlations.append({
                        "Metric": display_name,
                        "Correlation": corr
                    })
            
            corr_df = pd.DataFrame(correlations)
            corr_df = corr_df.sort_values("Correlation", ascending=False)
            
            # Display correlation bar chart
            fig = px.bar(
                corr_df,
                x="Correlation",
                y="Metric",
                orientation='h',
                title="Correlation with Base Score",
                color="Correlation",
                color_continuous_scale=["red", "yellow", "green"],
                range_color=[-1, 1]
            )
            fig.update_layout(height=400 + len(corr_df) * 20)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display correlation table
            st.subheader("Correlation Values")
            st.dataframe(
                corr_df.style.background_gradient(subset=['Correlation'], cmap='RdYlGn', vmin=-1, vmax=1),
                use_container_width=True
            )
        else:
            st.info("No numeric metrics available for correlation analysis")
    
    with tab4:
        st.header("Raw Data")
        
        # Show raw dataframe
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"metrics_{version_id}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()