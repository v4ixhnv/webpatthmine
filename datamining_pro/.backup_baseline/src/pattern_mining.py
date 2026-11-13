"""
Pattern Mining Module for WEBPATHMINE.
Implements FP-Growth frequent itemset mining and association rule generation
using mlxtend, with execution telemetry and human-readable formatting.
"""
from typing import Dict, Any, Tuple, Optional
import time
import warnings
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def run_fpgrowth(
    one_hot_df: pd.DataFrame,
    min_support: float = 0.05,
    max_len: Optional[int] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Executes FP-Growth algorithm on a boolean one-hot encoded transaction DataFrame.
    
    Parameters:
        one_hot_df: Boolean DataFrame where rows are transactions and columns are items.
        min_support: Minimum support threshold (between 0.001 and 1.0).
        max_len: Maximum length of itemsets to discover.
        
    Returns:
        Tuple:
            - frequent_itemsets: DataFrame containing discovered frequent itemsets.
            - execution_time_sec: Execution duration in seconds.
    """
    if one_hot_df.empty:
        raise ValueError("Transaction dataset is empty.")
    if not (0.0001 <= min_support <= 1.0):
        raise ValueError(f"min_support must be between 0.0001 and 1.0, got {min_support}")
        
    # Ensure boolean dtype
    bool_df = one_hot_df.astype(bool)
    
    start_time = time.perf_counter()
    itemsets = fpgrowth(bool_df, min_support=min_support, use_colnames=True, max_len=max_len)
    exec_time = time.perf_counter() - start_time
    
    if not itemsets.empty:
        itemsets['length'] = itemsets['itemsets'].apply(len)
        itemsets['itemsets_str'] = itemsets['itemsets'].apply(lambda x: ", ".join(sorted(list(x))))
        itemsets = itemsets.sort_values(by=['support', 'length'], ascending=[False, False]).reset_index(drop=True)
        
    return itemsets, exec_time

def generate_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Generates association rules from frequent itemsets.
    
    Parameters:
        frequent_itemsets: Output from run_fpgrowth.
        metric: Evaluation metric ('confidence', 'lift', 'support').
        min_threshold: Minimum threshold for selected metric.
        
    Returns:
        pd.DataFrame: Formatted association rules with support, confidence, lift, leverage, conviction.
    """
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    # Check if there are itemsets of length >= 2
    if 'length' in frequent_itemsets.columns:
        valid_lens = (frequent_itemsets['length'] >= 2).sum()
        if valid_lens == 0:
            return pd.DataFrame()
            
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            rules = association_rules(
                frequent_itemsets,
                metric=metric,
                min_threshold=min_threshold
            )
    except Exception as e:
        # Fall back or return empty if no rules found
        return pd.DataFrame()
        
    if rules.empty:
        return rules
        
    # Create human-readable string representations
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ", ".join(sorted(list(x))))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ", ".join(sorted(list(x))))
    rules['rule_repr'] = rules['antecedents_str'] + " -> " + rules['consequents_str']
    
    # Round numerical metrics for clean presentation
    rules['support'] = rules['support'].round(4)
    rules['confidence'] = rules['confidence'].round(4)
    rules['lift'] = rules['lift'].round(4)
    if 'leverage' in rules.columns:
        rules['leverage'] = rules['leverage'].round(4)
    if 'conviction' in rules.columns:
        rules['conviction'] = rules['conviction'].replace([float('inf'), float('-inf')], 99.99).round(2)
        
    rules = rules.sort_values(by=[metric, 'lift'], ascending=[False, False]).reset_index(drop=True)
    return rules

def filter_rules(
    rules_df: pd.DataFrame,
    min_confidence: float = 0.0,
    min_lift: float = 1.0,
    consequent_contains: Optional[str] = None
) -> pd.DataFrame:
    """
    Filters association rules by confidence, lift, or specific consequent item.
    """
    if rules_df.empty:
        return rules_df
        
    filtered = rules_df[
        (rules_df['confidence'] >= min_confidence) &
        (rules_df['lift'] >= min_lift)
    ]
    
    if consequent_contains and consequent_contains.strip():
        search_term = consequent_contains.strip().lower()
        filtered = filtered[filtered['consequents_str'].str.lower().str.contains(search_term)]
        
    return filtered.reset_index(drop=True)
