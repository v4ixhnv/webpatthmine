"""
Page 2: Navigation Pattern Mining
Implements FP-Growth frequent itemset mining, association rules with support/confidence/lift,
execution timing benchmarks, and sequential navigation flow analysis.
"""
import io
import time
import os
import streamlit as st
import pandas as pd

from src.visualization import (
    inject_custom_css,
    render_hero,
    plot_rules_scatter,
    plot_top_rules_bar,
    plot_transition_heatmap
)
from src.feature_engineering import (
    clickstream_to_transactions,
    shoppers_to_transactions,
    extract_navigation_sequences
)
from src.pattern_mining import (
    run_fpgrowth,
    generate_rules,
    filter_rules
)

st.set_page_config(
    page_title="Pattern Mining | WEBPATHMINE",
    page_icon="🕸️",
    layout="wide"
)

st.markdown(inject_custom_css(), unsafe_allow_html=True)

st.markdown(
    render_hero(
        title="PATTERN <span>MINING</span>",
        subtitle="FP-Growth Frequent Itemset Discovery, Association Rules & Navigation Flows",
        badge="MODULE 2: NAVIGATION PATTERN MINING"
    ),
    unsafe_allow_html=True
)

# Educational Callout: Distinguishing Unordered vs Ordered Mining
st.markdown("""
<div class="navy-card">
    <h4 style="color: #00e8c6; margin: 0 0 0.5rem 0;">💡 Concept Distinction: Itemset Mining vs. Sequential Navigation Analysis</h4>
    <p style="margin: 0; color: #cbd5e1; font-size: 0.92rem; line-height: 1.5;">
        <strong>Unordered Itemset Mining (FP-Growth):</strong> Treats each user session as an unordered basket of pages visited or actions taken. 
        It identifies which pages/behaviors co-occur together frequently within a session (e.g., <em>{Product Detail, Reviews} ➔ {Cart}</em>), 
        regardless of the order in which they occurred.<br>
        <strong>Ordered Navigation Sequence Analysis:</strong> Explicitly preserves timestamps and chronological page order to measure 
        transition probabilities from step <em>t</em> to step <em>t+1</em> (e.g., <em>Home ➔ Catalog ➔ Product</em>).
    </p>
</div>
""", unsafe_allow_html=True)

# Dataset Resolution
df_source = None
is_clickstream = False

# Check if clickstream_df exists or active df is clickstream
if st.session_state.get('clickstream_df') is not None:
    clickstream_source = st.session_state.clickstream_df
else:
    p_click = "data/clickstream_sample.csv"
    clickstream_source = pd.read_csv(p_click) if os.path.exists(p_click) else None

active_df = st.session_state.get('df')

# Selection mode
mining_mode = st.radio(
    "Select Transaction Mode for Pattern Mining:",
    options=[
        "Clickstream Page Navigation (Timestamped Page Views)",
        "Shopper Behavioral Features (Page Categories & Intent)"
    ],
    horizontal=True
)

if "Clickstream" in mining_mode:
    if clickstream_source is not None:
        df_source = clickstream_source
        is_clickstream = True
    elif active_df is not None and ('session_id' in active_df.columns or 'page_name' in active_df.columns):
        df_source = active_df
        is_clickstream = True
    else:
        st.warning("Clickstream dataset not loaded. Loading default sample clickstream data...")
        if os.path.exists("data/clickstream_sample.csv"):
            df_source = pd.read_csv("data/clickstream_sample.csv")
            st.session_state.clickstream_df = df_source
            is_clickstream = True
else:
    if active_df is not None and 'ProductRelated' in active_df.columns:
        df_source = active_df
        is_clickstream = False
    else:
        st.info("Loading UCI Online Shoppers dataset for behavioral pattern mining...")
        if os.path.exists("data/online_shoppers_intention.csv"):
            df_source = pd.read_csv("data/online_shoppers_intention.csv")
            st.session_state.df = df_source
            is_clickstream = False

if df_source is None:
    st.error("No valid dataset available for pattern mining. Please load data from the Data Explorer page.")
    st.stop()

# Prepare Transaction Matrix
with st.spinner("Preparing transaction records..."):
    if is_clickstream:
        # User can map columns if necessary
        sess_col = 'session_id' if 'session_id' in df_source.columns else df_source.columns[0]
        item_col = 'page_name' if 'page_name' in df_source.columns else ('page_path' if 'page_path' in df_source.columns else df_source.columns[1])
        time_col = 'timestamp' if 'timestamp' in df_source.columns else None
        
        transactions, one_hot_df, item_names = clickstream_to_transactions(
            df_source,
            session_col=sess_col,
            item_col=item_col,
            timestamp_col=time_col
        )
    else:
        transactions, one_hot_df, item_names = shoppers_to_transactions(df_source)

# Configuration Controls
st.markdown("### ⚙️ FP-Growth & Association Rule Parameters")
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

with ctrl1:
    min_support = st.slider(
        "Min Support (Frequency)",
        min_value=0.01,
        max_value=0.60,
        value=0.08,
        step=0.01,
        help="Proportion of sessions that must contain the itemset."
    )

with ctrl2:
    metric_choice = st.selectbox(
        "Rule Evaluation Metric",
        options=["confidence", "lift", "support"],
        index=0,
        help="Metric used to filter discovered association rules."
    )

with ctrl3:
    default_thresh = 0.3 if metric_choice == 'confidence' else (1.1 if metric_choice == 'lift' else 0.05)
    min_threshold = st.slider(
        f"Min {metric_choice.capitalize()} Threshold",
        min_value=0.01 if metric_choice != 'lift' else 0.5,
        max_value=1.0 if metric_choice != 'lift' else 5.0,
        value=default_thresh,
        step=0.02 if metric_choice != 'lift' else 0.1
    )

with ctrl4:
    max_len = st.selectbox(
        "Max Itemset Length",
        options=[None, 2, 3, 4, 5],
        index=0,
        format_func=lambda x: "No Limit" if x is None else f"Max {x} Items"
    )

# Execution
frequent_itemsets, exec_time = run_fpgrowth(one_hot_df, min_support=min_support, max_len=max_len)
rules_df = generate_rules(frequent_itemsets, metric=metric_choice, min_threshold=min_threshold)

# Telemetry Display
st.markdown("---")
b1, b2, b3, b4 = st.columns(4)
with b1:
    st.metric("Total Sessions (Transactions)", f"{len(transactions):,}")
with b2:
    st.metric("Unique Items Discovered", f"{len(item_names)}")
with b3:
    st.metric("Frequent Itemsets Found", f"{len(frequent_itemsets):,}")
with b4:
    # Execution time badge
    time_badge = f"{exec_time * 1000:.2f} ms" if exec_time < 1.0 else f"{exec_time:.3f} s"
    st.metric("FP-Growth Execution Time", time_badge)

# Results Tabs
tab_itemsets, tab_rules, tab_viz, tab_seq, tab_export = st.tabs([
    "📦 Frequent Itemsets",
    "🔗 Association Rules",
    "📊 Rule Visualizations",
    "🧭 Sequential Flow Analysis",
    "💾 Export Results"
])

with tab_itemsets:
    st.markdown("#### Discovered Frequent Itemsets")
    if frequent_itemsets.empty:
        st.warning("No itemsets meet the current minimum support threshold. Try lowering Min Support.")
    else:
        st.caption(f"Showing {len(frequent_itemsets):,} frequent itemsets discovered in {time_badge}.")
        disp_itemsets = frequent_itemsets[['itemsets_str', 'support', 'length']].rename(
            columns={'itemsets_str': 'Itemset', 'support': 'Support', 'length': 'Item Count'}
        )
        st.dataframe(disp_itemsets, use_container_width=True, height=350)

with tab_rules:
    st.markdown("#### Association Rules")
    if rules_df.empty:
        st.warning("No association rules found meeting the threshold criteria. Try lowering the threshold or support.")
    else:
        st.caption(f"Showing {len(rules_df):,} association rules ranked by {metric_choice.capitalize()}.")
        
        # Rule search filter
        search_filter = st.text_input("Filter rules containing keyword (e.g., 'cart', 'Purchase', 'reviews'):")
        filtered_rules = rules_df.copy()
        if search_filter:
            filtered_rules = filtered_rules[
                filtered_rules['rule_repr'].str.lower().str.contains(search_filter.lower())
            ]
            
        disp_rules = filtered_rules[['rule_repr', 'support', 'confidence', 'lift']].rename(
            columns={'rule_repr': 'Rule (Antecedents -> Consequents)', 'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
        )
        st.dataframe(disp_rules, use_container_width=True, height=380)

with tab_viz:
    st.markdown("#### Association Rule Metrics Visualizations")
    if rules_df.empty:
        st.info("No rules available to visualize.")
    else:
        vcol1, vcol2 = st.columns([1, 1])
        with vcol1:
            scatter_fig = plot_rules_scatter(rules_df)
            st.plotly_chart(scatter_fig, use_container_width=True)
        with vcol2:
            top_bar_fig = plot_top_rules_bar(rules_df, top_n=10, metric=metric_choice)
            st.plotly_chart(top_bar_fig, use_container_width=True)

with tab_seq:
    st.markdown("#### Ordered Navigation Sequence & Transition Probabilities")
    st.caption("First-order Markovian transition probability from step t to step t+1 based on timestamped user journeys.")
    
    if is_clickstream:
        seqs, prob_matrix = extract_navigation_sequences(
            df_source,
            session_col='session_id' if 'session_id' in df_source.columns else df_source.columns[0],
            page_col='page_name' if 'page_name' in df_source.columns else df_source.columns[1],
            timestamp_col='timestamp' if 'timestamp' in df_source.columns else None
        )
        
        if not prob_matrix.empty:
            heat_fig = plot_transition_heatmap(prob_matrix)
            st.plotly_chart(heat_fig, use_container_width=True)
            
            st.markdown("##### Most Common Direct Navigation Paths")
            path_strings = [" -> ".join(s[:6]) + ("..." if len(s) > 6 else "") for s in seqs if len(s) >= 2]
            path_counts = pd.Series(path_strings).value_counts().head(8).reset_index()
            path_counts.columns = ['Navigation Path', 'Frequency (Sessions)']
            st.dataframe(path_counts, use_container_width=True)
        else:
            st.info("Insufficient sequence transitions found in data.")
    else:
        st.info("Sequential flow matrix requires timestamped event logs. Switch to Clickstream mode above to inspect temporal transitions.")

with tab_export:
    st.markdown("#### Export Pattern Mining Results")
    exp1, exp2 = st.columns(2)
    with exp1:
        if not frequent_itemsets.empty:
            fi_csv = frequent_itemsets[['itemsets_str', 'support', 'length']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Frequent Itemsets CSV",
                data=fi_csv,
                file_name="frequent_itemsets.csv",
                mime="text/csv",
                use_container_width=True
            )
    with exp2:
        if not rules_df.empty:
            r_csv = rules_df[['rule_repr', 'antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Association Rules CSV",
                data=r_csv,
                file_name="association_rules.csv",
                mime="text/csv",
                use_container_width=True
            )
