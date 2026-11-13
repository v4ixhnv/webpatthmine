"""
Visualization Module for WebPathMine.
Provides professional white-and-orange themed Plotly visualizations and UI components.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Design System Color Tokens (Professional White & Orange Theme)
ORANGE_PRIMARY = '#F97316'   # Tailwind Orange 500
ORANGE_DARK = '#EA580C'      # Tailwind Orange 600
ORANGE_LIGHT = '#FFEDD5'     # Tailwind Orange 100
ORANGE_SUBTLE = '#FFF7ED'    # Tailwind Orange 50
TEXT_MAIN = '#1F2937'        # Dark Charcoal
TEXT_MUTED = '#6B7280'       # Secondary Grey
BORDER_LIGHT = '#E5E7EB'     # Light Neutral Grey
BG_WHITE = '#FFFFFF'
BG_PANEL = '#F9FAFB'         # Pale Warm Neutral
SUCCESS_GREEN = '#16A34A'    # Green for positive conversions
ALERT_CORAL = '#DC2626'      # Red for drop-off / non-purchase

LIGHT_THEME = {
    'layout': {
        'paper_bgcolor': BG_WHITE,
        'plot_bgcolor': BG_PANEL,
        'font': {'color': TEXT_MAIN, 'family': 'Inter, Segoe UI, sans-serif'},
        'margin': {'l': 40, 'r': 40, 't': 45, 'b': 40},
        'xaxis': {
            'gridcolor': BORDER_LIGHT,
            'zerolinecolor': '#D1D5DB',
            'color': TEXT_MAIN
        },
        'yaxis': {
            'gridcolor': BORDER_LIGHT,
            'zerolinecolor': '#D1D5DB',
            'color': TEXT_MAIN
        }
    }
}

# Aliases for backwards compatibility with existing test scripts
NAVY_THEME = LIGHT_THEME
CYAN_PRIMARY = ORANGE_PRIMARY
CYAN_BRIGHT = ORANGE_PRIMARY
NAVY_ACCENT = ORANGE_DARK
CORAL_ACCENT = ALERT_CORAL

def apply_light_theme(fig: go.Figure) -> go.Figure:
    """Applies clean white-and-orange styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=LIGHT_THEME['layout']['paper_bgcolor'],
        plot_bgcolor=LIGHT_THEME['layout']['plot_bgcolor'],
        font=LIGHT_THEME['layout']['font'],
        margin=LIGHT_THEME['layout']['margin'],
        xaxis=LIGHT_THEME['layout']['xaxis'],
        yaxis=LIGHT_THEME['layout']['yaxis']
    )
    return fig

# Backward-compatible alias
apply_navy_theme = apply_light_theme

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
        ORANGE_PRIMARY, "#3B82F6", "#6366F1",
        "#10B981", "#8B5CF6", ORANGE_DARK,
        "#F59E0B", SUCCESS_GREEN, "#9CA3AF"
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color=BORDER_LIGHT, width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(249, 115, 22, 0.25)"
        )
    )])
    fig.update_layout(
        title="<b>E-Commerce Navigation Flow Architecture</b>",
        font=dict(color=TEXT_MAIN, size=12)
    )
    return apply_light_theme(fig)

def plot_missing_values(missing_df: pd.DataFrame) -> go.Figure:
    """Plots missing value percentages per column."""
    filtered = missing_df[missing_df['Missing %'] > 0].sort_values(by='Missing %', ascending=True)
    if filtered.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No Missing Values Detected in Dataset (100% Complete)",
            showarrow=False,
            font=dict(size=15, color=SUCCESS_GREEN)
        )
        return apply_light_theme(fig)
        
    fig = px.bar(
        filtered,
        x='Missing %',
        y='Column',
        orientation='h',
        color='Missing %',
        color_continuous_scale=[[0, ORANGE_LIGHT], [1, ORANGE_DARK]],
        title="<b>Missing Values by Column (%)</b>"
    )
    fig.update_coloraxes(showscale=False)
    return apply_light_theme(fig)

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
    """Interactive correlation matrix heatmap for numerical variables."""
    if len(numeric_cols) < 2:
        return go.Figure()
        
    corr = df[numeric_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title="<b>Feature Correlation Matrix</b>"
    )
    return apply_light_theme(fig)

def plot_rules_scatter(rules_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of Support vs Confidence with Lift color scale."""
    if rules_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Association Rules to Plot", showarrow=False, font=dict(color=TEXT_MUTED))
        return apply_light_theme(fig)
        
    fig = px.scatter(
        rules_df,
        x='support',
        y='confidence',
        color='lift',
        size='lift',
        hover_data=['rule_repr', 'lift', 'support', 'confidence'],
        color_continuous_scale='Oranges',
        title="<b>Association Rules: Support vs Confidence (Color/Size = Lift)</b>",
        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
    )
    return apply_light_theme(fig)

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
        color_continuous_scale='Oranges',
        title=f"<b>Top {min(top_n, len(rules_df))} Rules by {metric.capitalize()}</b>",
        labels={metric: metric.capitalize(), 'rule_repr': 'Association Rule'}
    )
    return apply_light_theme(fig)

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
    return apply_light_theme(fig)

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
        line=dict(color=TEXT_MAIN, width=2.5),
        marker=dict(size=7)
    ))
    
    # Silhouette line (right axis)
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=sil_scores,
        name="Silhouette Score",
        mode='lines+markers',
        line=dict(color=ORANGE_PRIMARY, width=2.5, dash='dash'),
        marker=dict(size=7),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="<b>Cluster Diagnostics: Elbow & Silhouette Analysis</b>",
        xaxis=dict(title="Number of Clusters (k)"),
        yaxis=dict(
            title=dict(text="Inertia", font=dict(color=TEXT_MAIN)),
            tickfont=dict(color=TEXT_MAIN)
        ),
        yaxis2=dict(
            title=dict(text="Silhouette Score", font=dict(color=ORANGE_PRIMARY)),
            tickfont=dict(color=ORANGE_PRIMARY),
            overlaying='y',
            side='right',
            range=[-0.1, 1.0]
        ),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)', bordercolor=BORDER_LIGHT, borderwidth=1)
    )
    return apply_light_theme(fig)

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
        color_discrete_sequence=['#F97316', '#2563EB', '#10B981', '#8B5CF6', '#EC4899', '#F59E0B'],
        title="<b>Behavioral Clusters in 2D Principal Component Space</b>",
        labels={
            'PC1': f"Principal Component 1 ({var_ratios[0]}% var)",
            'PC2': f"Principal Component 2 ({var_ratios[1]}% var)"
        }
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='#FFFFFF')))
    return apply_light_theme(fig)

def plot_cluster_sizes(size_df: pd.DataFrame) -> go.Figure:
    """Donut chart of cluster population proportions."""
    fig = px.pie(
        size_df,
        names='Cluster',
        values='Session Count',
        hole=0.45,
        color_discrete_sequence=['#F97316', '#2563EB', '#10B981', '#8B5CF6', '#F59E0B'],
        title="<b>Cluster Size Distribution</b>"
    )
    return apply_light_theme(fig)

def plot_cluster_radar(profile_df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Radar chart comparing normalized cluster feature profiles."""
    fig = go.Figure()
    
    min_vals = profile_df.min()
    max_vals = profile_df.max()
    denom = (max_vals - min_vals).replace(0, 1)
    norm_df = (profile_df - min_vals) / denom
    
    colors = ['#F97316', '#2563EB', '#10B981', '#8B5CF6', '#F59E0B']
    
    for idx, cluster_name in enumerate(norm_df.index):
        vals = norm_df.loc[cluster_name].tolist()
        vals.append(vals[0])
        feats = features + [features[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=feats,
            fill='toself',
            name=cluster_name,
            line=dict(color=colors[idx % len(colors)], width=2),
            opacity=0.5
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=BORDER_LIGHT),
            angularaxis=dict(gridcolor=BORDER_LIGHT)
        ),
        title="<b>Cluster Feature Profile Comparison (Normalized Relative Scales)</b>",
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor=BORDER_LIGHT, borderwidth=1)
    )
    return apply_light_theme(fig)

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
            txt_color = '#FFFFFF' if val > (total * 0.3) else TEXT_MAIN
            annotations.append(dict(
                x=x_labels[j],
                y=y_labels[i],
                text=f"<b>{val:,}</b><br>{pct}",
                showarrow=False,
                font=dict(color=txt_color, size=13)
            ))
            
    fig = px.imshow(
        z,
        x=x_labels,
        y=y_labels,
        color_continuous_scale='Oranges',
        title="<b>Confusion Matrix (Evaluation Set)</b>"
    )
    fig.update_layout(annotations=annotations)
    fig.update_coloraxes(showscale=False)
    return apply_light_theme(fig)

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart for Decision Tree feature importances."""
    top_df = importance_df.head(top_n).iloc[::-1]
    
    fig = px.bar(
        top_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=[[0, ORANGE_LIGHT], [1, ORANGE_PRIMARY]],
        title=f"<b>Top {min(top_n, len(importance_df))} Feature Importances (Gini/Entropy Gain)</b>"
    )
    fig.update_coloraxes(showscale=False)
    return apply_light_theme(fig)

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
        fig.add_annotation(text="No event data to plot timeline", showarrow=False, font=dict(color=TEXT_MUTED))
        return apply_light_theme(fig)
        
    top_sessions = events_df[session_col].unique()[:max_sessions]
    df_sub = events_df[events_df[session_col].isin(top_sessions)].copy()
    
    has_valid_ts = False
    if timestamp_col and timestamp_col in df_sub.columns:
        df_sub['dt'] = pd.to_datetime(df_sub[timestamp_col], errors='coerce')
        if df_sub['dt'].notnull().any():
            has_valid_ts = True
            df_sub = df_sub.sort_values(by=[session_col, 'dt'])
            
    if not has_valid_ts:
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
        color_discrete_sequence=['#F97316', '#2563EB', '#10B981', '#8B5CF6', '#F59E0B', '#06B6D4', '#EC4899']
    )
    
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color='#FFFFFF')))
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Session ID",
        legend_title="Event / Page"
    )
    return apply_light_theme(fig)

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
        color_discrete_sequence=['#9CA3AF', ORANGE_PRIMARY],
        title="<b>Predicted Purchase Intent Distribution</b>"
    )
    return apply_light_theme(fig)

def inject_custom_css() -> str:
    """Returns custom CSS for Streamlit with clean white background and professional orange accents."""
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1F2937;
        }

        code, pre, [class*="stCode"] {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Top header accent bar */
        header[data-testid="stHeader"] {
            background: #FFFFFF;
            border-bottom: 1px solid #E5E7EB;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #F9FAFB !important;
            border-right: 1px solid #E5E7EB;
        }

        /* Clean White Card container */
        .navy-card, .light-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            transition: border-color 0.15s ease;
        }
        .navy-card:hover, .light-card:hover {
            border-color: #F97316;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }
        div[data-testid="stMetric"]:hover {
            border-color: #F97316;
        }
        div[data-testid="stMetric"] label {
            color: #6B7280 !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F97316 !important;
            font-weight: 800 !important;
            font-size: 1.65rem !important;
        }

        /* Minimal Header */
        .workspace-header {
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #E5E7EB;
        }
        .workspace-badge {
            display: inline-block;
            background: #FFF7ED;
            color: #EA580C;
            border: 1px solid #FDBA74;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .workspace-title {
            font-size: 1.85rem;
            font-weight: 800;
            color: #1F2937;
            margin: 0 0 4px 0;
            letter-spacing: -0.02em;
        }
        .workspace-subtitle {
            font-size: 0.95rem;
            color: #6B7280;
            margin: 0;
            line-height: 1.5;
            font-weight: 400;
        }

        /* Standard Buttons */
        div.stButton > button {
            background: #F97316;
            color: #FFFFFF;
            border: 1px solid #EA580C;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1.15rem;
            transition: all 0.15s ease;
        }
        div.stButton > button:hover {
            background: #EA580C;
            color: #FFFFFF;
            border-color: #C2410C;
            box-shadow: 0 2px 6px rgba(249, 115, 22, 0.25);
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #E5E7EB;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            color: #6B7280;
            font-weight: 600;
            padding: 0.55rem 1.1rem;
            border: none;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFF7ED !important;
            color: #EA580C !important;
            border-bottom: 2px solid #F97316 !important;
        }

        /* Tables & Dataframes */
        div[data-testid="stDataFrame"] {
            border: 1px solid #E5E7EB;
            border-radius: 6px;
            overflow: hidden;
        }
    </style>
    """
    return css

def render_hero(title: str, subtitle: str, badge: str = "WEBPATHMINE") -> str:
    """Renders standardized low-profile white-and-orange workspace header HTML."""
    clean_title = title.replace('<span>', '').replace('</span>', '')
    return f"""
    <div class="workspace-header">
        <span class="workspace-badge">{badge}</span>
        <h1 class="workspace-title">{clean_title}</h1>
        <p class="workspace-subtitle">{subtitle}</p>
    </div>
    """

def render_purchase_verdict_card(
    prediction: Any,
    probability: float,
    session_id: str,
    cluster_name: Optional[str] = None,
    key_drivers: Optional[Dict[str, Any]] = None,
    recommendation: Optional[str] = None
) -> str:
    """
    Renders a clean, professional white-and-orange verdict card displaying the conversion outcome,
    confidence gauge, and decision drivers. Zero decorative emojis.
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
        status_color = SUCCESS_GREEN
        status_bg = "#F0FDF4"
        border_color = "#86EFAC"
        headline = "High Purchase Intent Detected — Session exhibits strong purchasing signals"
        gauge_gradient = "linear-gradient(90deg, #FDBA74 0%, #16A34A 100%)"
        default_rec = "Session exhibits high checkout propensity. Provide frictionless 1-click checkout, express payment options, and clear trust reassurance."
    else:
        status_title = "PREDICTED CONVERSION: NO PURCHASE"
        status_badge = "BROWSING ONLY"
        status_color = ALERT_CORAL
        status_bg = "#FEF2F2"
        border_color = "#FECACA"
        headline = "Low Purchase Intent — Browsing session without immediate buying indicators"
        gauge_gradient = "linear-gradient(90deg, #D1D5DB 0%, #DC2626 100%)"
        default_rec = "Session shows elevated exit/bounce characteristics. Deploy targeted exit-intent promotions, free-shipping thresholds, or trending product feeds."

    rec_text = recommendation if recommendation else default_rec
    cluster_pill = (
        f'<span style="background: #F3F4F6; color: #374151; '
        f'border: 1px solid #D1D5DB; border-radius: 4px; '
        f'padding: 2px 8px; font-size: 0.78rem; font-weight: 600; margin-left: 8px;">'
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
                f'<div style="background: #F9FAFB; border: 1px solid #E5E7EB; '
                f'border-radius: 6px; padding: 6px 12px; min-width: 120px;">'
                f'<div style="color: #6B7280; font-size: 0.72rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;">{k}</div>'
                f'<div style="color: #1F2937; font-size: 0.95rem; font-weight: 700; margin-top: 2px;">{val_str}</div>'
                f'</div>'
            )
        if chip_items:
            chips_html = f'<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;">{"".join(chip_items)}</div>'

    card_html = (
        f'<div style="background: #FFFFFF; border: 2px solid {border_color}; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; '
        f'box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05); font-family: Plus Jakarta Sans, -apple-system, Segoe UI, sans-serif;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">'
        f'<div style="display: flex; align-items: center;">'
        f'<span style="background: #F3F4F6; color: #4B5563; border: 1px solid #E5E7EB; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; font-weight: 600;">'
        f'Session: <strong style="color: #1F2937;">{session_id}</strong></span>'
        f'{cluster_pill}</div>'
        f'<div><span style="background: {status_bg}; color: {status_color}; border: 1px solid {border_color}; border-radius: 4px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em;">{status_badge}</span></div>'
        f'</div>'
        f'<div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; margin-top: 6px;">'
        f'<div style="flex-grow: 1;">'
        f'<div style="color: {status_color}; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.01em; margin: 0;">{status_title}</div>'
        f'<div style="color: #4B5563; font-size: 0.9rem; margin-top: 4px;">{headline}</div>'
        f'</div>'
        f'<div style="text-align: right; min-width: 120px;">'
        f'<div style="color: #6B7280; font-size: 0.75rem; text-transform: uppercase; font-weight: 600;">Purchase Likelihood</div>'
        f'<div style="color: {status_color}; font-size: 1.75rem; font-weight: 800; line-height: 1.1;">{prob_pct:.1f}%</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top: 12px;">'
        f'<div style="width: 100%; height: 8px; background: #F3F4F6; border-radius: 4px; overflow: hidden; border: 1px solid #E5E7EB;">'
        f'<div style="width: {prob_pct}%; height: 100%; background: {gauge_gradient}; border-radius: 4px;"></div>'
        f'</div>'
        f'</div>'
        f'{chips_html}'
        f'<div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #F3F4F6; color: #4B5563; font-size: 0.85rem; line-height: 1.5;">'
        f'<strong style="color: {status_color};">Actionable Recommendation:</strong> {rec_text}</div>'
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
    """Generates comprehensive markdown text analysis report without emojis."""
    report = []
    report.append("# WEBPATHMINE: Navigation Pattern Mining and Intent Analysis Report\n")
    report.append(f"- **Generated At:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- **Dataset Source:** {metadata.get('file_name', 'Uploaded Dataset')}")
    report.append(f"- **Total Events Ingested:** {validation.get('total_events', 0):,}")
    report.append(f"- **Total Unique Sessions:** {validation.get('total_sessions', 0):,}")
    report.append(f"- **Valid Timestamp Rate:** {validation.get('valid_timestamp_pct', 0.0):.1f}%\n")
    
    report.append("## 1. Data Quality and Preprocessing")
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
