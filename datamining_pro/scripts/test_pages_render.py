"""
Tests that Behavior Segmentation and Purchase Prediction page scripts
execute without raising any exceptions when session state is populated.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clustering import prepare_clustering_data, fit_kmeans, compute_elbow_and_silhouette, generate_cluster_profiles
from src.prediction import train_and_evaluate_model, predict_single_session
from src.visualization import plot_elbow_and_silhouette, plot_confusion_matrix, plot_feature_importance

def test_pages_logic():
    print("Testing Behavior Segmentation logic...")
    df = pd.read_csv("data/online_shoppers_intention.csv")
    cluster_features = ['Administrative_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    scaled_data, scaler, valid_subset = prepare_clustering_data(df, cluster_features)
    km_model, labels, sil = fit_kmeans(scaled_data, n_clusters=3)
    diag = compute_elbow_and_silhouette(scaled_data, k_range=range(2, 6))
    
    # Verify Plotly figure builds with no Plotly 6 property errors
    diag_fig = plot_elbow_and_silhouette(diag)
    assert diag_fig is not None
    print("plot_elbow_and_silhouette generated successfully with no titlefont error!")
    
    print("Testing Purchase Prediction form logic...")
    pipeline, results = train_and_evaluate_model(
        df,
        target_col='Revenue',
        criterion='gini',
        max_depth=5,
        test_size=0.2
    )
    cm_fig = plot_confusion_matrix(results['confusion_matrix'])
    imp_fig = plot_feature_importance(results['feature_importance'])
    assert cm_fig is not None
    assert imp_fig is not None
    
    # Test column typing logic from Purchase Prediction single session form
    feat_cols = results['feature_columns']
    input_data = {}
    for col in feat_cols:
        col_series = df[col]
        if pd.api.types.is_bool_dtype(col_series):
            input_data[col] = False
        elif pd.api.types.is_numeric_dtype(col_series):
            input_data[col] = float(col_series.median())
        else:
            unique_vals = [str(x) for x in col_series.dropna().unique().tolist()]
            input_data[col] = unique_vals[0]
            
    pred_res = predict_single_session(pipeline, input_data)
    assert pred_res['outcome_label'] in ['Purchase', 'No Purchase']
    print(f"Single session form simulated inference succeeded: {pred_res['outcome_label']}")
    
    print("Testing Behavioural Ranking logic...")
    from src.feature_engineering import compute_behavioural_engagement_score
    ranking_df, rank_meta = compute_behavioural_engagement_score(df)
    assert not ranking_df.empty
    assert rank_meta['mean_score'] >= 0.0
    assert (ranking_df['Engagement_Score'] >= 0.0).all() and (ranking_df['Engagement_Score'] <= 100.0).all()
    print(f"Ranking verified: {len(ranking_df)} sessions ranked, mean score {rank_meta['mean_score']:.1f}")
    
    print("All page components verified cleanly!")

if __name__ == '__main__':
    test_pages_logic()
