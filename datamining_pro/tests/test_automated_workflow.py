"""
Unit and Integration Tests for the Automated Analysis Workflow in WEBPATHMINE.
Tests CSV/JSON ingestion, schema validation, event-to-session feature engineering,
trained K-Means cluster assignment, Decision Tree batch prediction (with and without ground-truth),
and leakage prevention.
"""
import unittest
import os
import sys
import json
import tempfile
import pandas as pd
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import (
    load_dataset,
    validate_activity_dataset
)
from src.feature_engineering import (
    aggregate_events_to_sessions,
    clickstream_to_transactions
)
from src.pattern_mining import (
    run_fpgrowth,
    generate_rules
)
from src.clustering import (
    get_or_train_reference_clustering,
    assign_clusters,
    DEFAULT_BEHAVIORAL_FEATURES
)
from src.prediction import (
    get_or_train_reference_pipeline,
    batch_predict_sessions
)
from src.visualization import (
    plot_session_timeline,
    generate_analysis_report
)

class TestAutomatedWorkflow(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create sample events log
        cls.events_data = [
            # Session 1: Buyer journey
            {"session_id": "S1", "timestamp": "2024-03-01 10:00:00", "page_name": "home", "event_type": "page_view", "dwell_time_seconds": 30},
            {"session_id": "S1", "timestamp": "2024-03-01 10:01:00", "page_name": "catalog_electronics", "event_type": "page_view", "dwell_time_seconds": 60},
            {"session_id": "S1", "timestamp": "2024-03-01 10:02:00", "page_name": "cart", "event_type": "add_to_cart", "dwell_time_seconds": 45},
            {"session_id": "S1", "timestamp": "2024-03-01 10:03:00", "page_name": "checkout_shipping", "event_type": "page_view", "dwell_time_seconds": 40},
            {"session_id": "S1", "timestamp": "2024-03-01 10:04:00", "page_name": "order_confirmation", "event_type": "purchase", "dwell_time_seconds": 20},
            # Session 2: Cart abandoner
            {"session_id": "S2", "timestamp": "2024-03-01 11:00:00", "page_name": "home", "event_type": "page_view", "dwell_time_seconds": 25},
            {"session_id": "S2", "timestamp": "2024-03-01 11:01:00", "page_name": "cart", "event_type": "add_to_cart", "dwell_time_seconds": 50},
            # Session 3: Single bounce
            {"session_id": "S3", "timestamp": "2024-03-01 12:00:00", "page_name": "home", "event_type": "page_view", "dwell_time_seconds": 10}
        ]
        cls.events_df = pd.DataFrame(cls.events_data)
        
    def test_json_and_csv_loading(self):
        # Test loading from JSON file
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump(self.events_data, f)
            temp_json = f.name
            
        try:
            loaded_json_df = load_dataset(temp_json)
            self.assertEqual(len(loaded_json_df), len(self.events_data))
            self.assertIn("session_id", loaded_json_df.columns)
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)
                
    def test_validate_activity_dataset(self):
        # Valid dataset
        val_res = validate_activity_dataset(self.events_df)
        self.assertTrue(val_res['is_valid'])
        self.assertEqual(val_res['total_sessions'], 3)
        self.assertEqual(val_res['session_col'], 'session_id')
        self.assertGreater(val_res['valid_timestamp_pct'], 90.0)
        
        # Invalid dataset (missing session column)
        bad_df = pd.DataFrame({'random_col': [1, 2, 3]})
        bad_val = validate_activity_dataset(bad_df)
        self.assertFalse(bad_val['is_valid'])
        self.assertTrue(any('Session Identifier' in e for e in bad_val['errors']))
        
    def test_aggregate_events_to_sessions(self):
        session_df, cleaned_events = aggregate_events_to_sessions(
            self.events_df,
            session_col='session_id',
            timestamp_col='timestamp',
            page_col='page_name',
            event_col='event_type',
            dwell_col='dwell_time_seconds'
        )
        # Check sessions count
        self.assertEqual(len(session_df), 3)
        # Check raw events preserved separately
        self.assertEqual(len(cleaned_events), len(self.events_data))
        
        # Verify Session 1 (Buyer)
        s1 = session_df[session_df['session_id'] == 'S1'].iloc[0]
        self.assertTrue(s1['Revenue'])
        self.assertGreater(s1['Cart_Interactions'], 0)
        self.assertGreater(s1['Checkout_Activity'], 0)
        self.assertGreater(s1['Total_Duration'], 100)
        
        # Verify Session 3 (Bounce)
        s3 = session_df[session_df['session_id'] == 'S3'].iloc[0]
        self.assertFalse(s3['Revenue'])
        self.assertEqual(s3['Cart_Interactions'], 0)
        self.assertGreaterEqual(s3['BounceRates'], 0.15)
        
    def test_pretrained_clustering_assignment(self):
        session_df, _ = aggregate_events_to_sessions(self.events_df)
        
        # Load or initialize pre-trained clustering model
        model, scaler, feat_cols = get_or_train_reference_clustering(n_clusters=3)
        self.assertIsNotNone(model)
        
        # Assign multiple sessions
        labels, sil, pca_coords, var_ratios = assign_clusters(model, scaler, session_df, feat_cols)
        self.assertEqual(len(labels), 3)
        self.assertEqual(pca_coords.shape[0], 3)
        
        # Test assignment on single uploaded session (must not crash, must not retrain)
        single_sess = session_df.iloc[[0]].copy()
        s_labels, s_sil, s_pca, _ = assign_clusters(model, scaler, single_sess, feat_cols)
        self.assertEqual(len(s_labels), 1)
        self.assertIsNone(s_sil)  # Silhouette is None for single session
        
    def test_pretrained_decision_tree_batch_prediction(self):
        session_df, _ = aggregate_events_to_sessions(self.events_df)
        
        # Get reference pipeline
        pipeline, _ = get_or_train_reference_pipeline(criterion='gini', max_depth=5)
        self.assertIsNotNone(pipeline)
        
        # Batch predict with ground truth Revenue present
        pred_df, summary = batch_predict_sessions(pipeline, session_df, target_col='Revenue')
        self.assertEqual(len(pred_df), 3)
        self.assertIn('Predicted_Conversion', pred_df.columns)
        self.assertIn('Purchase_Probability', pred_df.columns)
        self.assertTrue(summary['has_ground_truth'])
        self.assertIsNotNone(summary['evaluation'])
        
        # Batch predict with UNLABELLED sessions (prevent metric fabrication)
        unlabelled_df = session_df.drop(columns=['Revenue']).copy()
        pred_unlabelled, summary_unlabelled = batch_predict_sessions(pipeline, unlabelled_df, target_col=None)
        self.assertFalse(summary_unlabelled['has_ground_truth'])
        self.assertIsNone(summary_unlabelled['evaluation'])
        self.assertEqual(summary_unlabelled['total_sessions'], 3)
        
    def test_timeline_and_report_generation(self):
        fig = plot_session_timeline(self.events_df)
        self.assertIsNotNone(fig)
        
        report = generate_analysis_report(
            metadata={'file_name': 'test.json'},
            validation={'total_events': 8, 'total_sessions': 3, 'valid_timestamp_pct': 100.0, 'errors': [], 'warnings': []},
            prediction_summary={'total_sessions': 3, 'predicted_purchases': 1, 'predicted_no_purchases': 2, 'predicted_conversion_rate': 33.3, 'evaluation': None}
        )
        self.assertIn("WEBPATHMINE", report)
        self.assertIn("Predicted Purchases", report)

    def test_render_purchase_verdict_card(self):
        from src.visualization import render_purchase_verdict_card
        
        # Test Purchase outcome card
        card_buyer = render_purchase_verdict_card(
            prediction=1,
            probability=0.92,
            session_id='VIP_01',
            cluster_name='Cluster 1',
            key_drivers={'Page Values': 45.2, 'Product Duration': '320s'}
        )
        self.assertIn("PREDICTED CONVERSION: PURCHASE", card_buyer)
        self.assertIn("LIKELY BUYER", card_buyer)
        self.assertIn("92.0%", card_buyer)
        self.assertIn("VIP_01", card_buyer)
        self.assertIn("Cluster 1", card_buyer)
        self.assertIn("45.2", card_buyer)

        # Test No Purchase outcome card
        card_nonbuyer = render_purchase_verdict_card(
            prediction=0,
            probability=0.08,
            session_id='BOUNCE_02',
            cluster_name='Cluster 0',
            key_drivers={'Bounce Rate': '20.0%'}
        )
        self.assertIn("PREDICTED CONVERSION: NO PURCHASE", card_nonbuyer)
        self.assertIn("BROWSING ONLY", card_nonbuyer)
        self.assertIn("8.0%", card_nonbuyer)
        self.assertIn("BOUNCE_02", card_nonbuyer)

    def test_prepare_unified_session_data(self):
        from src.feature_engineering import prepare_unified_session_data

        # 1. Clickstream event DataFrame
        res_click = prepare_unified_session_data(self.events_df)
        self.assertEqual(res_click['dataset_type'], 'clickstream')
        self.assertEqual(res_click['total_sessions'], 3)
        self.assertFalse(res_click['is_single_session'])
        self.assertTrue(res_click['has_event_stream'])

        # 2. Single session event DataFrame
        single_df = self.events_df[self.events_df['session_id'] == 'S1']
        res_single = prepare_unified_session_data(single_df)
        self.assertTrue(res_single['is_single_session'])
        self.assertEqual(res_single['total_sessions'], 1)

        # 3. Tabular session DataFrame (UCI format)
        tab_df = pd.DataFrame([{
            'Administrative': 2, 'Administrative_Duration': 45.0,
            'ProductRelated': 15, 'ProductRelated_Duration': 350.0,
            'BounceRates': 0.0, 'ExitRates': 0.02, 'PageValues': 25.0,
            'Revenue': True
        }])
        res_tab = prepare_unified_session_data(tab_df)
        self.assertEqual(res_tab['total_sessions'], 1)
        self.assertTrue(res_tab['is_single_session'])
        self.assertIn('ProductRelated', res_tab['session_features_df'].columns)

    def test_workflow_session_state_unification_contract(self):
        """Verify that unified session data seamlessly feeds clustering, prediction, and ranking."""
        from src.feature_engineering import prepare_unified_session_data, compute_behavioural_engagement_score
        res = prepare_unified_session_data(self.events_df)
        session_df = res['session_features_df']
        
        # Verify columns required by downstream pages exist
        self.assertIn('session_id', session_df.columns)
        self.assertIn('ProductRelated', session_df.columns)
        self.assertIn('ProductRelated_Duration', session_df.columns)
        self.assertIn('BounceRates', session_df.columns)
        self.assertIn('ExitRates', session_df.columns)
        self.assertIn('Revenue', session_df.columns)
        
        # Verify ranking operates on this unified output
        ranked_df, meta = compute_behavioural_engagement_score(session_df)
        self.assertEqual(len(ranked_df), len(session_df))
        self.assertIn('Engagement_Score', ranked_df.columns)
        self.assertIn('Rank', ranked_df.columns)
        self.assertTrue((ranked_df['Engagement_Score'] >= 0.0).all() and (ranked_df['Engagement_Score'] <= 100.0).all())

if __name__ == '__main__':
    unittest.main()

