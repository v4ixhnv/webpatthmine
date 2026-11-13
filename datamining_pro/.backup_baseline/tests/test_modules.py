"""
Unit Tests for WEBPATHMINE Core Modules.
Covers preprocessing, feature engineering, FP-Growth pattern mining,
K-Means clustering, Decision Tree prediction pipeline, and persistence.
"""
import unittest
import os
import sys
import shutil

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from src.preprocessing import (
    load_dataset,
    inspect_dataset,
    detect_dataset_type,
    clean_dataset
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
from src.clustering import (
    prepare_clustering_data,
    fit_kmeans,
    compute_elbow_and_silhouette,
    generate_cluster_profiles
)
from src.prediction import (
    train_and_evaluate_model,
    predict_single_session,
    save_model,
    load_model
)

class TestWebPathMine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create small synthetic test DataFrames
        np.random.seed(42)
        cls.shoppers_df = pd.DataFrame({
            'Administrative': [0, 2, 4, 1, 0, 3, 1, 0, 5, 2],
            'Administrative_Duration': [0.0, 50.0, 120.0, 20.0, 0.0, 90.0, 15.0, 0.0, 160.0, 45.0],
            'Informational': [0, 0, 1, 0, 0, 2, 0, 0, 1, 0],
            'Informational_Duration': [0.0, 0.0, 30.0, 0.0, 0.0, 60.0, 0.0, 0.0, 40.0, 0.0],
            'ProductRelated': [3, 15, 35, 8, 2, 28, 6, 4, 45, 12],
            'ProductRelated_Duration': [50.0, 300.0, 950.0, 150.0, 30.0, 800.0, 120.0, 80.0, 1200.0, 250.0],
            'BounceRates': [0.15, 0.02, 0.01, 0.05, 0.20, 0.01, 0.08, 0.12, 0.005, 0.03],
            'ExitRates': [0.18, 0.04, 0.02, 0.07, 0.20, 0.02, 0.10, 0.15, 0.01, 0.05],
            'PageValues': [0.0, 15.2, 35.8, 0.0, 0.0, 42.1, 0.0, 0.0, 55.0, 12.0],
            'SpecialDay': [0.0] * 10,
            'Month': ['Feb', 'Mar', 'May', 'May', 'Nov', 'Nov', 'Dec', 'Mar', 'Nov', 'Oct'],
            'OperatingSystems': [1, 2, 2, 3, 1, 2, 2, 1, 2, 3],
            'Browser': [1, 2, 2, 1, 1, 2, 1, 1, 2, 2],
            'Region': [1, 3, 2, 1, 4, 1, 2, 3, 1, 2],
            'TrafficType': [1, 2, 2, 3, 1, 2, 1, 2, 2, 3],
            'VisitorType': ['Returning_Visitor', 'Returning_Visitor', 'New_Visitor', 'Returning_Visitor',
                            'Returning_Visitor', 'New_Visitor', 'Returning_Visitor', 'Returning_Visitor',
                            'New_Visitor', 'Returning_Visitor'],
            'Weekend': [False, True, False, False, True, False, True, False, False, True],
            'Revenue': [False, True, True, False, False, True, False, False, True, False]
        })
        
        cls.clickstream_df = pd.DataFrame({
            'session_id': ['S1', 'S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S3', 'S4', 'S4'],
            'timestamp': [
                '2024-03-01 10:00:00', '2024-03-01 10:02:00', '2024-03-01 10:05:00',
                '2024-03-01 11:00:00', '2024-03-01 11:03:00',
                '2024-03-01 12:00:00', '2024-03-01 12:02:00', '2024-03-01 12:04:00',
                '2024-03-01 13:00:00', '2024-03-01 13:01:00'
            ],
            'page_name': ['home', 'catalog', 'cart', 'home', 'search', 'home', 'catalog', 'cart', 'home', 'cart']
        })
        
        os.makedirs('tests/tmp', exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists('tests/tmp'):
            shutil.rmtree('tests/tmp')

    # PREPROCESSING TESTS
    def test_inspect_dataset(self):
        audit = inspect_dataset(self.shoppers_df)
        self.assertEqual(audit['n_rows'], 10)
        self.assertEqual(audit['n_cols'], 18)
        self.assertEqual(audit['duplicate_count'], 0)
        self.assertIn('Revenue', audit['potential_targets'])
        
    def test_detect_dataset_type(self):
        self.assertEqual(detect_dataset_type(self.shoppers_df), 'uci_online_shoppers')
        self.assertEqual(detect_dataset_type(self.clickstream_df), 'clickstream')
        
    def test_clean_dataset_duplicate_handling(self):
        df_with_dups = pd.concat([self.shoppers_df, self.shoppers_df.iloc[[0]]], ignore_index=True)
        cleaned, report = clean_dataset(df_with_dups, drop_duplicates=True)
        self.assertEqual(len(cleaned), 10)
        self.assertEqual(report['duplicates_removed'], 1)

    def test_clean_dataset_missing_imputation(self):
        df_with_null = self.shoppers_df.copy()
        df_with_null.loc[0, 'Administrative_Duration'] = np.nan
        cleaned, report = clean_dataset(df_with_null, missing_strategy='impute')
        self.assertFalse(cleaned['Administrative_Duration'].isnull().any())
        self.assertTrue(len(report['imputed_columns']) > 0)

    # FEATURE ENGINEERING & PATTERN MINING TESTS
    def test_clickstream_to_transactions(self):
        tx, one_hot, items = clickstream_to_transactions(self.clickstream_df, 'session_id', 'page_name')
        self.assertEqual(len(tx), 4)  # 4 unique sessions
        self.assertIn('home', items)
        self.assertIn('cart', items)
        self.assertEqual(len(one_hot), 4)
        
    def test_fpgrowth_and_rules(self):
        _, one_hot, _ = clickstream_to_transactions(self.clickstream_df, 'session_id', 'page_name')
        itemsets, exec_time = run_fpgrowth(one_hot, min_support=0.3)
        self.assertFalse(itemsets.empty)
        self.assertGreater(exec_time, 0.0)
        
        rules = generate_rules(itemsets, metric='confidence', min_threshold=0.3)
        self.assertIsInstance(rules, pd.DataFrame)
        if not rules.empty:
            filtered = filter_rules(rules, min_confidence=0.5, min_lift=1.0)
            self.assertIsInstance(filtered, pd.DataFrame)

    def test_navigation_sequences(self):
        seqs, prob_matrix = extract_navigation_sequences(self.clickstream_df, 'session_id', 'page_name')
        self.assertEqual(len(seqs), 4)
        self.assertFalse(prob_matrix.empty)
        self.assertIn('home', prob_matrix.index)

    # CLUSTERING TESTS
    def test_clustering_workflow(self):
        features = ['Administrative_Duration', 'ProductRelated_Duration', 'BounceRates', 'PageValues']
        scaled, scaler, subset = prepare_clustering_data(self.shoppers_df, features)
        self.assertEqual(scaled.shape, (10, 4))
        
        kmeans, labels, sil = fit_kmeans(scaled, n_clusters=2, random_state=42)
        self.assertEqual(len(labels), 10)
        self.assertIsNotNone(sil)
        
        diag = compute_elbow_and_silhouette(scaled, k_range=range(2, 5))
        self.assertEqual(len(diag['inertias']), 3)
        
        profile, size_df, desc = generate_cluster_profiles(self.shoppers_df, labels, features)
        self.assertEqual(len(profile), 2)
        self.assertEqual(len(size_df), 2)
        self.assertEqual(len(desc), 2)

    # PREDICTION TESTS
    def test_prediction_pipeline(self):
        pipeline, results = train_and_evaluate_model(
            self.shoppers_df,
            target_col='Revenue',
            criterion='gini',
            max_depth=3,
            test_size=0.3,
            random_state=42
        )
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1_score', results)
        self.assertIn('confusion_matrix', results)
        self.assertFalse(results['feature_importance'].empty)
        
        # Test inference on single session
        sample_input = {
            'Administrative': 2,
            'Administrative_Duration': 60.0,
            'Informational': 0,
            'Informational_Duration': 0.0,
            'ProductRelated': 18,
            'ProductRelated_Duration': 400.0,
            'BounceRates': 0.01,
            'ExitRates': 0.03,
            'PageValues': 25.0,
            'SpecialDay': 0.0,
            'Month': 'Nov',
            'OperatingSystems': 2,
            'Browser': 2,
            'Region': 1,
            'TrafficType': 2,
            'VisitorType': 'Returning_Visitor',
            'Weekend': False
        }
        pred_res = predict_single_session(pipeline, sample_input)
        self.assertIn(pred_res['outcome_label'], ['Purchase', 'No Purchase'])
        self.assertIsNotNone(pred_res['probability_purchase'])
        
        # Test persistence
        test_model_path = 'tests/tmp/test_pipeline.joblib'
        save_model(pipeline, test_model_path)
        self.assertTrue(os.path.exists(test_model_path))
        
        loaded = load_model(test_model_path)
        pred_res2 = predict_single_session(loaded, sample_input)
        self.assertEqual(pred_res['outcome_label'], pred_res2['outcome_label'])

if __name__ == '__main__':
    unittest.main()
