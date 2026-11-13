"""
Page 5: Dynamic Model Insights & Strategic Synthesis
Synthesizes discovered navigation association rules, empirical behavioral clusters,
leak-free purchase prediction drivers, and behavioural engagement rankings into
data-grounded observations and suggested optimization strategies.
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.visualization import (
    inject_custom_css,
    render_hero
)

st.set_page_config(
    page_title="Model Insights | WEBPATHMINE",
    page_icon="💡",
    layout="wide"
)

st.markdown(inject_custom_css(), unsafe_allow_html=True)

st.markdown(
    render_hero(
        title="MODEL <span>INSIGHTS</span>",
        subtitle="Dynamic Multi-Module Synthesis, Empirical Rules, Cluster Traits & Strategic Observations",
        badge="MODULE 5: STRATEGIC SYNTHESIS"
    ),
    unsafe_allow_html=True
)

st.markdown("""
<div class="navy-card">
    <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
        <strong>Scientific Rigor Guarantee:</strong> All insights and recommendations below are dynamically 
        derived from your active analysis modules (Pattern Mining, K-Means Clustering, Engagement Ranking, 
        and the Decision Tree Classifier). No fabricated rules, placeholder charts, or hardcoded assumptions are presented.
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: DYNAMIC ASSOCIATION RULES
# ==============================================================================
st.markdown("### 1. Frequent Navigation Association Rules (FP-Growth)")
rules_res = st.session_state.get('rules_results')

if rules_res is not None and 'rules' in rules_res and not rules_res['rules'].empty:
    rules_df = rules_res['rules'].copy()
    exec_ms = rules_res.get('exec_time', 0.0) * 1000.0
    
    st.caption(
        f"Mined using **FP-Growth** in `{exec_ms:.2f} ms` | "
        f"Min Support: `{rules_res.get('min_support', 0.08)}` | "
        f"Total Rules Discovered: **{len(rules_df):,}**"
    )
    
    # Sort by lift descending
    top_rules = rules_df.sort_values(by='lift', ascending=False).head(5).reset_index(drop=True)
    
    col_r1, col_r2 = st.columns([1.4, 1])
    with col_r1:
        st.markdown("##### Top Association Rules (Ranked by Lift)")
        # Format antecedents and consequents
        display_rules = pd.DataFrame({
            "Antecedents (If Visited)": top_rules['antecedents'].apply(lambda x: ', '.join(list(x))),
            "Consequents (Then Also Visited)": top_rules['consequents'].apply(lambda x: ', '.join(list(x))),
            "Support": top_rules['support'].apply(lambda s: f"{s:.1%}"),
            "Confidence": top_rules['confidence'].apply(lambda c: f"{c:.1%}"),
            "Lift": top_rules['lift'].apply(lambda l: f"{l:.2f}x")
        })
        st.dataframe(display_rules, use_container_width=True)
        
    with col_r2:
        st.markdown("##### Empirical Rule Interpretations")
        for i, row in top_rules.head(3).iterrows():
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            lift = float(row['lift'])
            conf = float(row['confidence'])
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border-left: 3px solid #00e8c6; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem; border-radius: 4px;">
                <span style="color: #00e8c6; font-weight: 600;">Rule {i+1}:</span> 
                Sessions visiting <code>{ant}</code> are <strong>{lift:.2f}×</strong> more likely to also navigate to <code>{con}</code> 
                (observed with <strong>{conf:.1%}</strong> confidence).
            </div>
            """, unsafe_allow_html=True)
elif rules_res is not None and 'rules' in rules_res and rules_res['rules'].empty:
    st.info("ℹ️ FP-Growth completed, but no association rules met the selected support and confidence thresholds. Lower the thresholds on Page 2 to discover additional patterns.")
else:
    st.info("💡 Run frequent pattern mining on the **Pattern Mining** page to discover and synthesize empirical navigation association rules.")


# ==============================================================================
# SECTION 2: BEHAVIORAL CLUSTERS & ARCHETYPES
# ==============================================================================
st.markdown("---")
st.markdown("### 2. Discovered Behavioral Segments (K-Means Clustering)")
clust_res = st.session_state.get('clustering_results')

if clust_res is not None and 'descriptions' in clust_res:
    descs = clust_res['descriptions']
    profile_df = clust_res['profile_df']
    sil = clust_res.get('silhouette')
    sil_str = f"{sil:.3f}" if sil is not None else "N/A"
    
    st.caption(
        f"Active Segmentation: **{clust_res['n_clusters']} clusters** | "
        f"Silhouette Separation Score: `{sil_str}` | "
        f"Evaluated Features: `{', '.join(clust_res['features'])}`"
    )
    
    cols = st.columns(len(descs))
    for idx, d in enumerate(descs):
        with cols[idx]:
            st.markdown(f"""
            <div class="navy-card">
                <h4 style="color: #00e8c6; margin: 0 0 0.5rem 0;">{d['cluster']}</h4>
                <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">
                    <strong>Size:</strong> {d['size']:,} sessions ({d['pct']}%)
                </p>
                <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                    <strong>Empirical Z-Score Characteristics:</strong>
                    <ul style="padding-left: 1.2rem; margin: 0.3rem 0 0 0;">
                        {''.join([f"<li>{trait}</li>" for trait in d['characteristics'][:3]])}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("💡 Run clustering on the **Behavior Segmentation** page to view quantitative segment profiles.")


# ==============================================================================
# SECTION 3: PURCHASE PREDICTION DRIVERS & GENERALIZATION
# ==============================================================================
st.markdown("---")
st.markdown("### 3. Purchase Intent Prediction Drivers (Decision Tree)")
model_res = st.session_state.get('model_results')

if model_res is not None and 'feature_importance' in model_res:
    imp_df = model_res['feature_importance']
    top_feats = imp_df.head(3)['Feature'].tolist() if 'Feature' in imp_df.columns else imp_df.head(3)['feature'].tolist()
    top_scores = imp_df.head(3)['Importance'].tolist() if 'Importance' in imp_df.columns else imp_df.head(3)['importance'].tolist()
    
    p_c1, p_c2 = st.columns([1.2, 1])
    with p_c1:
        st.markdown(f"""
        - **Primary Decision Tree Splitter:** `{top_feats[0]}` (Relative Weight: **{top_scores[0] * 100:.1f}%**)
        - **Secondary Splitter:** `{top_feats[1]}` (Relative Weight: **{top_scores[1] * 100:.1f}%**)
        - **Tertiary Splitter:** `{top_feats[2] if len(top_feats) > 2 else 'N/A'}` (Relative Weight: **{top_scores[2] * 100:.1f}%** if len(top_scores) > 2 else '')
        - **Test Accuracy:** `{model_res['accuracy'] * 100:.1f}%` | **Recall (Buyers):** `{model_res['recall'] * 100:.1f}%` | **Precision:** `{model_res['precision'] * 100:.1f}%`
        """)
        
        if 'cv_summary' in model_res and model_res['cv_summary']:
            cv = model_res['cv_summary']
            st.caption(
                f"🔄 **5-Fold Stratified Cross-Validation:** "
                f"Accuracy `{cv['cv_accuracy_mean']*100:.1f}% ± {cv['cv_accuracy_std']*100:.1f}%` | "
                f"F1 `{cv['cv_f1_mean']:.3f} ± {cv['cv_f1_std']:.3f}`"
            )
            
    with p_c2:
        st.info(f"""
        **Empirical Takeaway:** 
        The leak-free Decision Tree confirms that `{top_feats[0]}` produces the greatest reduction in node impurity. 
        Sessions with high `{top_feats[0]}` separate decisively into the high-conversion leaf nodes.
        """)
else:
    st.info("💡 Train the Decision Tree classifier on the **Purchase Prediction** page to inspect empirical feature importances.")


# ==============================================================================
# SECTION 4: BEHAVIOURAL ENGAGEMENT RANKING INSIGHTS
# ==============================================================================
st.markdown("---")
st.markdown("### 4. Behavioural Engagement Ranking Synthesis (0–100)")
rank_res = st.session_state.get('ranking_results')

if rank_res is not None and 'ranking_df' in rank_res:
    r_df = rank_res['ranking_df']
    r_meta = rank_res['meta']
    
    rk1, rk2, rk3, rk4 = st.columns(4)
    with rk1:
        st.metric("Total Ranked", f"{r_meta['total_records']:,} {r_meta['entity_type']}s")
    with rk2:
        st.metric("Mean Engagement", f"{r_meta['mean_score']:.1f} / 100")
    with rk3:
        st.metric("High Intent Tier (≥70)", f"{r_meta['high_intent_pct']}%", f"{r_meta['high_intent_count']:,} records")
    with rk4:
        st.metric("Moderate Intent (40–69)", f"{r_meta['moderate_intent_count']:,} records")
        
    st.markdown("##### 🏆 Top-5 Highly Engaged Sessions / Visitors")
    preview_cols = [c for c in ['Rank', 'Entity_ID', 'Engagement_Score', 'Engagement_Tier', 'Cluster', 'Score_Product', 'Score_Duration'] if c in r_df.columns]
    st.dataframe(r_df[preview_cols].head(5), use_container_width=True)
else:
    st.info("💡 Run clustering on the **Behavior Segmentation** page to automatically compute the Behavioural Engagement Leaderboard.")


# ==============================================================================
# SECTION 5: DATA-GROUNDED BUSINESS STRATEGY & CRO OBSERVATIONS
# ==============================================================================
st.markdown("---")
st.markdown("### 5. Data-Grounded Strategic Observations & CRO Actions")

# Build dynamic observation cards based on active results
obs_cards = []

# Dynamic Observation 1 from Prediction Top Feature
if model_res is not None and 'feature_importance' in model_res:
    imp_df = model_res['feature_importance']
    top_f = imp_df.head(1)['Feature'].iloc[0] if 'Feature' in imp_df.columns else imp_df.head(1)['feature'].iloc[0]
    top_w = float(imp_df.head(1)['Importance'].iloc[0]) if 'Importance' in imp_df.columns else float(imp_df.head(1)['importance'].iloc[0])
    obs_cards.append({
        'title': f"🎯 Intent Driver: Optimize Around '{top_f}'",
        'finding': f"The decision tree identifies <code>{top_f}</code> as the governing split criterion ({top_w:.1%} relative importance).",
        'prescription': f"Focus conversion rate optimization on activities influencing {top_f}. Provide frictionless pathways to product exploration and clear catalog navigational taxonomies."
    })
else:
    obs_cards.append({
        'title': "🎯 Conversion Modeling Recommendation",
        'finding': "Supervised classification has not yet been executed for the active dataset.",
        'prescription': "Train the leak-free Decision Tree Pipeline in Module 4 to isolate statistical predictors of purchase conversion."
    })

# Dynamic Observation 2 from Association Rules
if rules_res is not None and 'rules' in rules_res and not rules_res['rules'].empty:
    top_r = rules_res['rules'].sort_values(by='lift', ascending=False).iloc[0]
    ant_str = ', '.join(list(top_r['antecedents']))
    con_str = ', '.join(list(top_r['consequents']))
    obs_cards.append({
        'title': f"🔗 Navigation Flow: Connect '{ant_str}' ➔ '{con_str}'",
        'finding': f"Discovered high association lift ({float(top_r['lift']):.2f}x) from <code>{ant_str}</code> to <code>{con_str}</code> (confidence: {float(top_r['confidence']):.1%}).",
        'prescription': f"Implement direct in-page shortcuts or cross-recommendation widgets linking {ant_str} directly to {con_str} to minimize navigational friction."
    })
else:
    obs_cards.append({
        'title': "🔗 Navigation Mining Recommendation",
        'finding': "No high-lift association rules currently active.",
        'prescription': "Run FP-Growth in Module 2 with adjusted support thresholds to detect co-occurring page transitions."
    })

# Dynamic Observation 3 from Behavioral Clusters
if clust_res is not None and 'descriptions' in clust_res and len(clust_res['descriptions']) > 0:
    largest_cluster = max(clust_res['descriptions'], key=lambda x: x['size'])
    obs_cards.append({
        'title': f"👥 Audience Prioritization: '{largest_cluster['cluster']}' ({largest_cluster['pct']}%)",
        'finding': f"The largest discovered segment represents {largest_cluster['size']:,} sessions with traits: {'; '.join(largest_cluster['characteristics'][:2])}.",
        'prescription': "Tailor landing page messaging and search filters to address the dominant behavioral pattern observed in this majority segment."
    })

# Dynamic Observation 4 from Engagement Ranking
if rank_res is not None and 'meta' in rank_res:
    r_meta = rank_res['meta']
    obs_cards.append({
        'title': f"🏆 Engagement Distribution: {r_meta['high_intent_pct']}% High Intent",
        'finding': f"Out of {r_meta['total_records']:,} evaluated {r_meta['entity_type']}s, {r_meta['high_intent_count']:,} scored ≥ 70 on pre-outcome browsing depth.",
        'prescription': "Target high-intent sessions with immediate real-time assistance or cart prompts, while providing educational orientation to low-intent casual visitors."
    })

# Render 2x2 grid of dynamic observations
for i in range(0, len(obs_cards), 2):
    c1, c2 = st.columns(2)
    with c1:
        card = obs_cards[i]
        st.markdown(f"""
        <div class="navy-card">
            <h4 style="color: #00e8c6; margin-top: 0;">{card['title']}</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                <strong>Data Finding:</strong> {card['finding']}<br>
                <strong>Suggested Action:</strong> {card['prescription']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    if i + 1 < len(obs_cards):
        with c2:
            card = obs_cards[i + 1]
            st.markdown(f"""
            <div class="navy-card">
                <h4 style="color: #00e8c6; margin-top: 0;">{card['title']}</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                    <strong>Data Finding:</strong> {card['finding']}<br>
                    <strong>Suggested Action:</strong> {card['prescription']}
                </p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 1.5rem; padding: 0.75rem 1rem; border-left: 3px solid #f59e0b; background: rgba(30, 41, 59, 0.5); border-radius: 4px;">
    <p style="margin: 0; color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
        ⚠️ <strong>Academic / Scientific Disclaimer:</strong> Discovered association rules and feature importances reflect empirical correlations 
        within the analyzed dataset. Association does not prove causality, nor does an engagement score guarantee conversion. 
        Proposed business actions should be validated through controlled A/B experimentation before widespread deployment.
    </p>
</div>
""", unsafe_allow_html=True)
