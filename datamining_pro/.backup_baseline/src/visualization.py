"""
Visualization Module for WEBPATHMINE.
Provides dark navy & restrained cyan themed Plotly visualizations.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NAVY_THEME = {
    'layout': {
        'paper_bgcolor': 'rgba(10, 17, 40, 0.7)',
        'plot_bgcolor': 'rgba(0, 31, 84, 0.3)',
        'font': {'color': '#e0e6ed', 'family': 'Inter, Segoe UI, sans-serif'},
        'margin': {'l': 40, 'r': 40, 't': 45, 'b': 40},
        'xaxis': {
            'gridcolor': 'rgba(255, 255, 255, 0.08)',
            'zerolinecolor': 'rgba(255, 255, 255, 0.15)',
            'color': '#e0e6ed'
        },
        'yaxis': {
            'gridcolor': 'rgba(255, 255, 255, 0.08)',
            'zerolinecolor': 'rgba(255, 255, 255, 0.15)',
            'color': '#e0e6ed'
        }
    }
}

CYAN_PRIMARY = '#00e8c6'
CYAN_BRIGHT = '#00f0ff'
NAVY_ACCENT = '#1282a2'
CORAL_ACCENT = '#ff6b6b'

def apply_navy_theme(fig: go.Figure) -> go.Figure:
    """Applies clean dark navy styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=NAVY_THEME['layout']['paper_bgcolor'],
        plot_bgcolor=NAVY_THEME['layout']['plot_bgcolor'],
        font=NAVY_THEME['layout']['font'],
        margin=NAVY_THEME['layout']['margin'],
        xaxis=NAVY_THEME['layout']['xaxis'],
        yaxis=NAVY_THEME['layout']['yaxis']
    )
    return fig

def plot_navigation_sankey() -> go.Figure:
    """Creates a high-level e-commerce browsing journey Sankey diagram for Overview."""
    labels = [
        "Home / Landing", "Catalog Browse", "Search Query",
        "Product Detail", "Reviews / FAQ", "Cart",
        "Checkout Started", "Purchase Completed", "Session Exit"
    ]
    # Source indices to target indices
    sources = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6]
    targets = [1, 2, 8, 3, 8, 3, 8, 5, 4, 3, 6, 8, 7, 8]
    values =  [500, 300, 200, 420, 80, 250, 50, 320, 100, 80, 240, 80, 180, 60]
    
    node_colors = [
        "#00e8c6", "#1282a2", "#1282a2",
        "#00f0ff", "#034078", "#f77f00",
        "#fcbf49", "#2ec4b6", "#e63946"
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="#0a1128", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(0, 232, 198, 0.22)"
        )
    )])
    fig.update_layout(
        title="<b>E-Commerce Navigation Flow Architecture</b>",
        font=dict(color="#e0e6ed", size=12)
    )
    return apply_navy_theme(fig)

def plot_missing_values(missing_df: pd.DataFrame) -> go.Figure:
    """Plots missing value percentages per column."""
    filtered = missing_df[missing_df['Missing %'] > 0].sort_values(by='Missing %', ascending=True)
    if filtered.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No Missing Values Detected in Dataset (100% Complete)",
            showarrow=False,
            font=dict(size=16, color=CYAN_PRIMARY)
        )
        return apply_navy_theme(fig)
        
    fig = px.bar(
        filtered,
        x='Missing %',
        y='Column',
        orientation='h',
        color='Missing %',
        color_continuous_scale=[[0, NAVY_ACCENT], [1, CORAL_ACCENT]],
        title="<b>Missing Values by Column (%)</b>"
    )
    fig.update_coloraxes(showscale=False)
    return apply_navy_theme(fig)

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
    """Interactive correlation matrix heatmap for numerical variables."""
    if len(numeric_cols) < 2:
        return go.Figure()
        
    corr = df[numeric_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='Tealrose',
        zmin=-1,
        zmax=1,
        title="<b>Feature Correlation Matrix</b>"
    )
    return apply_navy_theme(fig)

def plot_rules_scatter(rules_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of Support vs Confidence with Lift color scale."""
    if rules_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Association Rules to Plot", showarrow=False, font=dict(color=CYAN_PRIMARY))
        return apply_navy_theme(fig)
        
    fig = px.scatter(
        rules_df,
        x='support',
        y='confidence',
        color='lift',
        size='lift',
        hover_data=['rule_repr', 'lift', 'support', 'confidence'],
        color_continuous_scale='Viridis',
        title="<b>Association Rules: Support vs Confidence (Color/Size = Lift)</b>",
        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
    )
    return apply_navy_theme(fig)

def plot_top_rules_bar(rules_df: pd.DataFrame, top_n: int = 10, metric: str = 'lift') -> go.Figure:
    """Horizontal bar chart of top association rules."""
    if rules_df.empty:
        return go.Figure()
        
    top_rules = rules_df.head(top_n).iloc[::-1]
    fig = px.bar(
        top_rules,
        x=metric,
        y='rule_repr',
        orientation='h',
        color=metric,
        color_continuous_scale='Bluyl',
        title=f"<b>Top {min(top_n, len(rules_df))} Rules by {metric.capitalize()}</b>",
        labels={metric: metric.capitalize(), 'rule_repr': 'Association Rule'}
    )
    return apply_navy_theme(fig)

def plot_transition_heatmap(prob_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of page transition probabilities."""
    if prob_matrix.empty:
        return go.Figure()
        
    fig = px.imshow(
        prob_matrix,
        text_auto=".2f",
        color_continuous_scale='Blues',
        zmin=0,
        zmax=1,
        title="<b>First-Order Page Navigation Transition Matrix (P(Next | Current))</b>"
    )
    return apply_navy_theme(fig)

def plot_elbow_and_silhouette(eval_dict: Dict[str, Any]) -> go.Figure:
    """Plots Inertia (Elbow) and Silhouette Score on dual y-axes."""
    k_vals = eval_dict['k_values']
    inertias = eval_dict['inertias']
    sil_scores = eval_dict['silhouette_scores']
    
    fig = go.Figure()
    
    # Inertia line (left axis)
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=inertias,
        name="Inertia (Sum of Squared Distances)",
        mode='lines+markers',
        line=dict(color=CYAN_PRIMARY, width=3),
        marker=dict(size=8)
    ))
    
    # Silhouette line (right axis)
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=sil_scores,
        name="Silhouette Score",
        mode='lines+markers',
        line=dict(color='#ff9f1c', width=3, dash='dash'),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="<b>Optimal Cluster Diagnostics: Elbow & Silhouette Analysis</b>",
        xaxis=dict(title="Number of Clusters (k)"),
        yaxis=dict(
            title=dict(text="Inertia", font=dict(color=CYAN_PRIMARY)),
            tickfont=dict(color=CYAN_PRIMARY)
        ),
        yaxis2=dict(
            title=dict(text="Silhouette Score", font=dict(color='#ff9f1c')),
            tickfont=dict(color='#ff9f1c'),
            overlaying='y',
            side='right',
            range=[-0.1, 1.0]
        ),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.4)')
    )
    return apply_navy_theme(fig)

def plot_clusters_pca(pca_coords: np.ndarray, labels: np.ndarray, var_ratios: List[float]) -> go.Figure:
    """2D PCA projection of behavioral clusters."""
    df_plot = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'Cluster': [f"Cluster {l}" for l in labels]
    })
    
    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Cluster',
        color_discrete_sequence=['#00e8c6', '#00b4d8', '#ffb703', '#e63946', '#9d4edd', '#06d6a0'],
        title="<b>Behavioral Clusters in 2D Principal Component Space</b>",
        labels={
            'PC1': f"Principal Component 1 ({var_ratios[0]}% var)",
            'PC2': f"Principal Component 2 ({var_ratios[1]}% var)"
        }
    )
    return apply_navy_theme(fig)

def plot_cluster_sizes(size_df: pd.DataFrame) -> go.Figure:
    """Donut chart of cluster population proportions."""
    fig = px.pie(
        size_df,
        names='Cluster',
        values='Session Count',
        hole=0.45,
        color_discrete_sequence=['#00e8c6', '#00b4d8', '#ffb703', '#e63946', '#9d4edd'],
        title="<b>Cluster Size Distribution</b>"
    )
    return apply_navy_theme(fig)

def plot_cluster_radar(profile_df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Radar chart comparing normalized cluster feature profiles."""
    fig = go.Figure()
    
    # Min-max normalize for radar comparison
    min_vals = profile_df.min()
    max_vals = profile_df.max()
    denom = (max_vals - min_vals).replace(0, 1)
    norm_df = (profile_df - min_vals) / denom
    
    colors = ['#00e8c6', '#00b4d8', '#ffb703', '#e63946', '#9d4edd']
    
    for idx, cluster_name in enumerate(norm_df.index):
        vals = norm_df.loc[cluster_name].tolist()
        vals.append(vals[0])  # Close the radar loop
        feats = features + [features[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=feats,
            fill='toself',
            name=cluster_name,
            line=dict(color=colors[idx % len(colors)], width=2),
            opacity=0.65
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.15)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.15)')
        ),
        title="<b>Cluster Feature Profile Comparison (Normalized Relative Scales)</b>"
    )
    return apply_navy_theme(fig)

def plot_confusion_matrix(cm_list: List[List[int]]) -> go.Figure:
    """Confusion matrix heatmap with absolute counts and labels."""
    z = np.array(cm_list)
    x_labels = ["Pred: No Purchase", "Pred: Purchase"]
    y_labels = ["True: No Purchase", "True: Purchase"]
    
    total = np.sum(z)
    annotations = []
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = z[i][j]
            pct = f"({val / total * 100:.1f}%)" if total > 0 else ""
            annotations.append(dict(
                x=x_labels[j],
                y=y_labels[i],
                text=f"<b>{val}</b><br>{pct}",
                showarrow=False,
                font=dict(color='#ffffff', size=14)
            ))
            
    fig = px.imshow(
        z,
        x=x_labels,
        y=y_labels,
        color_continuous_scale='Teal',
        title="<b>Confusion Matrix (Evaluation Set)</b>"
    )
    fig.update_layout(annotations=annotations)
    fig.update_coloraxes(showscale=False)
    return apply_navy_theme(fig)

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart for Decision Tree feature importances."""
    top_df = importance_df.head(top_n).iloc[::-1]
    
    fig = px.bar(
        top_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=[[0, '#001f54'], [1, CYAN_PRIMARY]],
        title=f"<b>Top {min(top_n, len(importance_df))} Feature Importances (Gini/Entropy Gain)</b>"
    )
    return apply_navy_theme(fig)

def inject_custom_css() -> str:
    """Returns custom CSS for Streamlit with dark navy and cyan accents."""
    css = """
    <style>
        /* Import clean modern fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        code, pre, [class*="stCode"] {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Top header accent bar */
        header[data-testid="stHeader"] {
            background: rgba(10, 17, 40, 0.85);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(0, 232, 198, 0.15);
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #050b1a !important;
            border-right: 1px solid rgba(0, 232, 198, 0.12);
        }

        /* Modern card container */
        .navy-card {
            background: linear-gradient(135deg, rgba(0, 31, 84, 0.45) 0%, rgba(10, 17, 40, 0.65) 100%);
            border: 1px solid rgba(0, 232, 198, 0.18);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .navy-card:hover {
            border-color: rgba(0, 232, 198, 0.4);
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(0, 31, 84, 0.5) 0%, rgba(5, 11, 26, 0.7) 100%);
            border: 1px solid rgba(0, 232, 198, 0.18);
            border-radius: 10px;
            padding: 0.85rem 1rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        div[data-testid="stMetric"]:hover {
            border-color: #00e8c6;
        }
        div[data-testid="stMetric"] label {
            color: #8da4c4 !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #00e8c6 !important;
            font-weight: 700 !important;
            font-size: 1.8rem !important;
        }

        /* Hero banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(0, 31, 84, 0.8) 0%, rgba(10, 17, 40, 0.95) 100%);
            border: 1px solid rgba(0, 232, 198, 0.25);
            border-left: 4px solid #00e8c6;
            border-radius: 12px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .hero-title span {
            color: #00e8c6;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #8da4c4;
            margin-top: 0.4rem;
            margin-bottom: 0;
            font-weight: 400;
        }

        /* Badge pill */
        .cyan-pill {
            display: inline-block;
            background: rgba(0, 232, 198, 0.12);
            color: #00e8c6;
            border: 1px solid rgba(0, 232, 198, 0.3);
            border-radius: 20px;
            padding: 0.2rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #034078 0%, #001f54 100%);
            color: #00e8c6;
            border: 1px solid rgba(0, 232, 198, 0.4);
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.25rem;
            transition: all 0.2s ease;
        }
        div.stButton > button:hover {
            background: #00e8c6;
            color: #0a1128;
            border-color: #00e8c6;
            box-shadow: 0 0 15px rgba(0, 232, 198, 0.4);
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid rgba(0, 232, 198, 0.15);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            color: #8da4c4;
            font-weight: 500;
            padding: 0.6rem 1.2rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 232, 198, 0.1) !important;
            color: #00e8c6 !important;
            border-bottom: 2px solid #00e8c6 !important;
        }

        /* Tables & Dataframes */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(0, 232, 198, 0.15);
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """
    return css

def render_hero(title: str, subtitle: str, badge: str = "WEBPATHMINE") -> str:
    """Renders standardized dark navy hero banner HTML."""
    return f"""
    <div class="hero-banner">
        <span class="cyan-pill">{badge}</span>
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """

def plot_session_timeline(
    events_df: pd.DataFrame,
    session_col: str = 'session_id',
    timestamp_col: Optional[str] = 'timestamp',
    page_col: str = 'page_name',
    event_col: Optional[str] = 'event_type',
    max_sessions: int = 8
) -> go.Figure:
    """
    Renders an interactive chronological timeline of user events across top sessions.
    """
    if events_df.empty or session_col not in events_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No event data to plot timeline", showarrow=False, font=dict(color=CYAN_PRIMARY))
        return apply_navy_theme(fig)
        
    # Take top sessions
    top_sessions = events_df[session_col].unique()[:max_sessions]
    df_sub = events_df[events_df[session_col].isin(top_sessions)].copy()
    
    # Parse or synthesize timestamps
    has_valid_ts = False
    if timestamp_col and timestamp_col in df_sub.columns:
        df_sub['dt'] = pd.to_datetime(df_sub[timestamp_col], errors='coerce')
        if df_sub['dt'].notnull().any():
            has_valid_ts = True
            df_sub = df_sub.sort_values(by=[session_col, 'dt'])
            
    if not has_valid_ts:
        # Create relative step indices
        df_sub['step'] = df_sub.groupby(session_col).cumcount() + 1
        x_axis = 'step'
        x_title = "Chronological Event Step in Session"
    else:
        x_axis = 'dt'
        x_title = "Event Timestamp"
        
    color_col = event_col if event_col and event_col in df_sub.columns else page_col
    
    fig = px.scatter(
        df_sub,
        x=x_axis,
        y=session_col,
        color=color_col,
        hover_data=[page_col, session_col] + ([timestamp_col] if timestamp_col in df_sub.columns else []),
        title=f"<b>Chronological User Activity Timeline (Top {len(top_sessions)} Sessions)</b>",
        color_discrete_sequence=['#00e8c6', '#00b4d8', '#ffb703', '#e63946', '#9d4edd', '#06d6a0', '#f77f00']
    )
    
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='#0a1128')))
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Session ID",
        legend_title="Event / Page"
    )
    return apply_navy_theme(fig)

def plot_prediction_distribution(summary: Dict[str, Any]) -> go.Figure:
    """Renders a donut chart of predicted conversion distribution."""
    labels = ['No Purchase', 'Purchase']
    values = [
        summary.get('predicted_no_purchases', 0),
        summary.get('predicted_purchases', 0)
    ]
    
    fig = px.pie(
        names=labels,
        values=values,
        hole=0.45,
        color_discrete_sequence=['#1282a2', CYAN_PRIMARY],
        title="<b>Predicted Purchase Intent Distribution</b>"
    )
    return apply_navy_theme(fig)

def render_purchase_verdict_card(
    prediction: Any,
    probability: float,
    session_id: str,
    cluster_name: Optional[str] = None,
    key_drivers: Optional[Dict[str, Any]] = None,
    recommendation: Optional[str] = None
) -> str:
    """
    Renders an eye-catching, glowing dark-navy verdict card displaying the final
    conversion outcome (Purchase vs No Purchase), confidence gauge, and decision drivers.
    Returns clean HTML without leading line-indentation to prevent markdown code-block escaping.
    """
    is_purchase = bool(
        prediction == 1 or
        prediction is True or
        str(prediction).lower() in ['1', 'true', 'purchase']
    )
    prob_pct = max(0.0, min(100.0, float(probability) * 100))
    
    if is_purchase:
        status_title = "PREDICTED CONVERSION: PURCHASE"
        status_badge = "LIKELY BUYER"
        status_color = "#00e8c6"  # cyan
        status_bg = "rgba(0, 232, 198, 0.12)"
        border_color = "rgba(0, 232, 198, 0.55)"
        glow_shadow = "0 0 25px rgba(0, 232, 198, 0.25)"
        icon = "🛍️"
        headline = "High Purchase Intent Detected — Shopper exhibits strong purchasing triggers"
        gauge_gradient = "linear-gradient(90deg, #1282a2 0%, #00e8c6 100%)"
        default_rec = "User exhibits high checkout propensity. Deliver frictionless 1-click checkout, express payment options, and trust reassurance badges."
    else:
        status_title = "PREDICTED CONVERSION: NO PURCHASE"
        status_badge = "BROWSING ONLY"
        status_color = "#ff6b6b"  # coral
        status_bg = "rgba(255, 107, 107, 0.12)"
        border_color = "rgba(255, 107, 107, 0.5)"
        glow_shadow = "0 0 20px rgba(255, 107, 107, 0.2)"
        icon = "👀"
        headline = "Low Purchase Intent / High Drop-off Risk — Browsing session without immediate buying signals"
        gauge_gradient = "linear-gradient(90deg, #4a5568 0%, #ff6b6b 100%)"
        default_rec = "Session shows elevated exit/bounce characteristics. Deploy targeted exit-intent incentives, free-shipping thresholds, or personalized trending product feeds."

    rec_text = recommendation if recommendation else default_rec
    cluster_pill = (
        f'<span style="background: rgba(18, 130, 162, 0.25); color: #00f0ff; '
        f'border: 1px solid rgba(0, 240, 255, 0.35); border-radius: 14px; '
        f'padding: 3px 10px; font-size: 0.78rem; font-weight: 600; margin-left: 8px;">'
        f'{cluster_name}</span>'
    ) if cluster_name else ''

    # Build key driver chips
    chips_html = ""
    if key_drivers:
        chip_items = []
        for k, v in key_drivers.items():
            if v is None:
                continue
            if isinstance(v, float):
                val_str = f"{v:.2f}"
            else:
                val_str = str(v)
            chip_items.append(
                f'<div style="background: rgba(10, 17, 40, 0.75); border: 1px solid rgba(255, 255, 255, 0.1); '
                f'border-radius: 8px; padding: 6px 12px; min-width: 120px;">'
                f'<div style="color: #8da4c4; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;">{k}</div>'
                f'<div style="color: #ffffff; font-size: 0.95rem; font-weight: 700; margin-top: 2px;">{val_str}</div>'
                f'</div>'
            )
        if chip_items:
            chips_html = f'<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px;">{"".join(chip_items)}</div>'

    card_html = (
        f'<div style="background: linear-gradient(135deg, rgba(0, 31, 84, 0.85) 0%, rgba(10, 17, 40, 0.98) 100%); '
        f'border: 2px solid {border_color}; border-radius: 14px; padding: 1.5rem 1.75rem; margin-bottom: 1.5rem; '
        f'box-shadow: {glow_shadow}; font-family: Inter, -apple-system, Segoe UI, sans-serif;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">'
        f'<div style="display: flex; align-items: center;">'
        f'<span style="background: rgba(255, 255, 255, 0.08); color: #e0e6ed; border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 14px; padding: 3px 12px; font-size: 0.8rem; font-weight: 600;">'
        f'Session: <strong style="color: #ffffff;">{session_id}</strong></span>'
        f'{cluster_pill}</div>'
        f'<div><span style="background: {status_bg}; color: {status_color}; border: 1px solid {border_color}; border-radius: 14px; padding: 4px 12px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em;">{status_badge}</span></div>'
        f'</div>'
        f'<div style="display: flex; align-items: flex-start; gap: 14px; margin-top: 8px;">'
        f'<div style="font-size: 2.4rem; line-height: 1;">{icon}</div>'
        f'<div style="flex-grow: 1;">'
        f'<div style="color: {status_color}; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.01em; margin: 0;">{status_title}</div>'
        f'<div style="color: #cbd5e1; font-size: 0.92rem; margin-top: 4px;">{headline}</div>'
        f'</div>'
        f'<div style="text-align: right;">'
        f'<div style="color: #8da4c4; font-size: 0.75rem; text-transform: uppercase; font-weight: 600;">Purchase Likelihood</div>'
        f'<div style="color: {status_color}; font-size: 1.85rem; font-weight: 800; line-height: 1.1;">{prob_pct:.1f}%</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top: 14px;">'
        f'<div style="width: 100%; height: 10px; background: rgba(255, 255, 255, 0.08); border-radius: 6px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.08);">'
        f'<div style="width: {prob_pct}%; height: 100%; background: {gauge_gradient}; border-radius: 6px;"></div>'
        f'</div>'
        f'</div>'
        f'{chips_html}'
        f'<div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255, 255, 255, 0.08); color: #94a3b8; font-size: 0.82rem; line-height: 1.5;">'
        f'<strong style="color: {status_color};">⚡ Actionable Recommendation:</strong> {rec_text}</div>'
        f'</div>'
    )
    return card_html


def generate_analysis_report(
    metadata: Dict[str, Any],
    validation: Dict[str, Any],
    rules_summary: Optional[Dict[str, Any]] = None,
    clustering_summary: Optional[Dict[str, Any]] = None,
    prediction_summary: Optional[Dict[str, Any]] = None
) -> str:
    """Generates comprehensive markdown text analysis report."""
    report = []
    report.append("# WEBPATHMINE: Automated E-Commerce Navigation & Intent Analysis Report\n")
    report.append(f"- **Generated At:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- **Dataset Source:** {metadata.get('file_name', 'Uploaded Dataset')}")
    report.append(f"- **Total Events Ingested:** {validation.get('total_events', 0):,}")
    report.append(f"- **Total Unique Sessions:** {validation.get('total_sessions', 0):,}")
    report.append(f"- **Valid Timestamp Rate:** {validation.get('valid_timestamp_pct', 0.0):.1f}%\n")
    
    report.append("## 1. Data Quality & Preprocessing")
    if validation.get('errors'):
        report.append("- **Errors Encountered:** " + "; ".join(validation['errors']))
    else:
        report.append("- **Schema Status:** Validated successfully without fatal schema errors.")
    if validation.get('warnings'):
        report.append("- **Notices:** " + "; ".join(validation['warnings']))
    report.append("\n")
    
    if rules_summary:
        report.append("## 2. FP-Growth Navigation Pattern Mining")
        report.append(f"- **Frequent Itemsets Discovered:** {rules_summary.get('itemset_count', 0):,}")
        report.append(f"- **Association Rules Generated:** {rules_summary.get('rule_count', 0):,}")
        report.append(f"- **Execution Duration:** {rules_summary.get('execution_time_ms', 0):.2f} ms")
        if rules_summary.get('top_rule'):
            report.append(f"- **Top Association Rule:** `{rules_summary['top_rule']}`")
        report.append("\n")
        
    if clustering_summary:
        report.append("## 3. Behavioral Segmentation (K-Means)")
        report.append(f"- **Clustering Model:** Pre-trained K-Means ({clustering_summary.get('n_clusters', 3)} clusters)")
        report.append(f"- **Standardized Features:** {', '.join(clustering_summary.get('features', []))}")
        if clustering_summary.get('silhouette') is not None:
            report.append(f"- **Silhouette Score:** {clustering_summary['silhouette']:.4f}")
        report.append("\n")
        
    if prediction_summary:
        report.append("## 4. Purchase Intent Prediction (Decision Tree)")
        report.append(f"- **Total Sessions Evaluated:** {prediction_summary.get('total_sessions', 0):,}")
        report.append(f"- **Predicted Purchases:** {prediction_summary.get('predicted_purchases', 0):,} ({prediction_summary.get('predicted_conversion_rate', 0.0):.1f}%)")
        report.append(f"- **Predicted No Purchases:** {prediction_summary.get('predicted_no_purchases', 0):,}")
        
        eval_dict = prediction_summary.get('evaluation')
        if eval_dict:
            report.append(f"- **Ground-Truth Evaluation Accuracy:** {eval_dict['accuracy'] * 100:.1f}%")
            report.append(f"- **Precision:** {eval_dict['precision'] * 100:.1f}% | **Recall:** {eval_dict['recall'] * 100:.1f}% | **F1-Score:** {eval_dict['f1_score']:.3f}")
        else:
            report.append("- **Ground-Truth Labels:** Unlabelled session input; predictions generated via statistical inference.")
        report.append("\n")
        
    report.append("## 5. Strategic Conversion Rate Optimization (CRO) Recommendations")
    report.append("1. **High-Intent Intervention:** Deploy exit-intent promotions or real-time assistance on sessions that trigger Cart or Checkout navigation.")
    report.append("2. **Friction Reduction:** Streamline landing page paths for high-bounce referral categories.")
    report.append("3. **Retargeting Strategy:** Follow up with shoppers categorized in high product-viewing segments who have not completed checkout within 24 hours.\n")
    
    return "\n".join(report)


