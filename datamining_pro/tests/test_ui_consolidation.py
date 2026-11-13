"""
Unit and Integration Tests for UI Consolidation and Professional Standards.
Verifies the 4-workspace architecture, removal of emojis, white-and-orange theme,
cache invalidation, and preservation of zero-leakage constraints.
"""
import unittest
import os
import re
import sys
import toml
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestUIConsolidation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cls.pages_dir = os.path.join(cls.root_dir, 'pages')
        cls.emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]')

    def test_pages_directory_structure(self):
        """Verify pages/ contains exactly the 3 intended workspace scripts and no legacy emoji pages."""
        page_files = [f for f in os.listdir(self.pages_dir) if f.endswith('.py')]
        expected_pages = [
            '1_Pattern_Mining.py',
            '2_Segmentation_and_Engagement.py',
            '3_Prediction_and_Insights.py'
        ]
        self.assertEqual(sorted(page_files), sorted(expected_pages))
        
        # Verify no file in pages/ contains any emoji
        for fname in page_files:
            self.assertIsNone(
                self.emoji_pattern.search(fname),
                f"Page filename '{fname}' must not contain emojis."
            )

    def test_zero_emojis_in_active_python_code(self):
        """Verify no decorative emojis exist in any active Python file in src, pages, scripts, and app.py."""
        emoji_violations = []
        for check_dir in ['src', 'pages', 'scripts']:
            full_dir = os.path.join(self.root_dir, check_dir)
            if not os.path.exists(full_dir):
                continue
            for fname in os.listdir(full_dir):
                if fname.endswith('.py'):
                    fpath = os.path.join(full_dir, fname)
                    with open(fpath, encoding='utf-8') as f:
                        for line_idx, line in enumerate(f, start=1):
                            if self.emoji_pattern.search(line):
                                emoji_violations.append(f"{fname}:{line_idx} - {line.strip()}")
                                
        # Check app.py as well
        app_path = os.path.join(self.root_dir, 'app.py')
        with open(app_path, encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                if self.emoji_pattern.search(line):
                    emoji_violations.append(f"app.py:{line_idx} - {line.strip()}")

        self.assertEqual(
            len(emoji_violations), 0,
            f"Found {len(emoji_violations)} emoji violations:\n" + "\n".join(emoji_violations[:10])
        )

    def test_theme_configuration(self):
        """Verify .streamlit/config.toml is configured with the white-and-orange theme."""
        config_path = os.path.join(self.root_dir, '.streamlit', 'config.toml')
        self.assertTrue(os.path.exists(config_path))
        
        with open(config_path, encoding='utf-8') as f:
            cfg = toml.load(f)
            
        theme = cfg.get('theme', {})
        self.assertEqual(theme.get('primaryColor', '').upper(), '#F97316')
        self.assertEqual(theme.get('backgroundColor', '').upper(), '#FFFFFF')
        self.assertEqual(theme.get('secondaryBackgroundColor', '').upper(), '#F9FAFB')
        self.assertEqual(theme.get('textColor', '').upper(), '#1F2937')

    def test_leak_free_constraints_preserved(self):
        """Verify target leakage constraints remain strictly enforced in prediction and ranking."""
        df = pd.read_csv(os.path.join(self.root_dir, "data", "online_shoppers_intention.csv"))
        
        # 1. Feature engineering engagement score must exclude Revenue, has_purchase, PageValues
        from src.feature_engineering import compute_behavioural_engagement_score
        ranking_df, rank_meta = compute_behavioural_engagement_score(df)
        signals = rank_meta['signals_used'].values()
        self.assertNotIn('Revenue', signals)
        self.assertNotIn('has_purchase', signals)
        self.assertNotIn('PageValues', signals)
        
        # 2. Decision Tree pipeline must exclude target and proxies
        from src.prediction import train_and_evaluate_model
        _, results = train_and_evaluate_model(df, target_col='Revenue', max_depth=3)
        self.assertNotIn('Revenue', results['feature_columns'])
        self.assertNotIn('has_purchase', results['feature_columns'])
        self.assertNotIn('PageValues', results['feature_columns'])

    def test_validate_dataset_capabilities_none_source_regression(self):
        """Regression test: Ensure validate_dataset_capabilities never fails on None or edge-case source_name."""
        from src.preprocessing import validate_dataset_capabilities
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        # Test None explicitly
        meta_none = validate_dataset_capabilities(df, source_name=None)
        self.assertIsInstance(meta_none, dict)
        self.assertIn('provenance', meta_none)
        self.assertFalse(meta_none['is_synthetic'])
        
        # Test empty string
        meta_empty = validate_dataset_capabilities(df, source_name="")
        self.assertIsInstance(meta_empty, dict)
        
        # Test non-string input (e.g. numeric or object)
        meta_num = validate_dataset_capabilities(df, source_name=999)
        self.assertIsInstance(meta_num, dict)
        self.assertIn("999", meta_num['provenance'])

if __name__ == '__main__':
    unittest.main()

