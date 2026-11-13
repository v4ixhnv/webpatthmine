"""
Feature Engineering and Session Aggregation Module for WEBPATHMINE.
Prepares clickstream and tabular e-commerce data for association rule mining
and sequential navigation analysis.
"""
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder

def clickstream_to_transactions(
    df: pd.DataFrame,
    session_col: str = 'session_id',
    item_col: str = 'page_name',
    timestamp_col: Optional[str] = 'timestamp'
) -> Tuple[List[List[str]], pd.DataFrame, List[str]]:
    """
    Transforms event-level clickstream logs into transaction records for FP-Growth,
    while preserving session boundaries.
    
    Parameters:
        df: Clickstream DataFrame.
        session_col: Column identifying session ID.
        item_col: Column identifying page name, URL, or category.
        timestamp_col: Column identifying timestamp (used for ordering if provided).
        
    Returns:
        Tuple:
            - transactions: List of lists containing unique items per session.
            - one_hot_df: One-hot encoded boolean DataFrame ready for FP-Growth.
            - item_names: List of all unique items across sessions.
    """
    if session_col not in df.columns:
        raise ValueError(f"Session column '{session_col}' not found in dataset.")
    if item_col not in df.columns:
        raise ValueError(f"Item/page column '{item_col}' not found in dataset.")
        
    df_sorted = df.copy()
    if timestamp_col and timestamp_col in df_sorted.columns:
        try:
            df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], errors='coerce')
            df_sorted = df_sorted.sort_values(by=[session_col, timestamp_col])
        except Exception:
            pass  # Fall back to original order if parsing fails
            
    # Group by session and collect unique items
    grouped = df_sorted.groupby(session_col)[item_col].apply(lambda s: list(pd.unique(s.dropna()))).reset_index()
    transactions = [tx for tx in grouped[item_col].tolist() if len(tx) > 0]
    
    if not transactions:
        raise ValueError("No valid transactions could be extracted from the dataset.")
        
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    one_hot_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    return transactions, one_hot_df, te.columns_

def shoppers_to_transactions(df: pd.DataFrame) -> Tuple[List[List[str]], pd.DataFrame, List[str]]:
    """
    Converts UCI Online Shoppers session features into discrete categorical itemsets
    to enable association rule mining on browsing behaviors and purchase outcomes.
    
    Examples of generated items:
        'Admin_Browsed', 'Info_Browsed', 'Product_Heavy (>20)',
        'Low_Bounce_Rate', 'High_Page_Value', 'Returning_Visitor', 'Weekend', 'Purchase'
    """
    transactions: List[List[str]] = []
    
    for _, row in df.iterrows():
        items: List[str] = []
        
        # Section visits
        if row.get('Administrative', 0) > 0:
            items.append('Admin_Browsed')
        if row.get('Informational', 0) > 0:
            items.append('Info_Browsed')
        prod = row.get('ProductRelated', 0)
        if prod > 25:
            items.append('Heavy_Product_Browse')
        elif prod > 5:
            items.append('Moderate_Product_Browse')
        else:
            items.append('Light_Product_Browse')
            
        # Bounce & Exit behavior
        if row.get('BounceRates', 0) < 0.02:
            items.append('Low_Bounce_Rate')
        elif row.get('BounceRates', 0) > 0.08:
            items.append('High_Bounce_Rate')
            
        # Page value indicator
        if row.get('PageValues', 0) > 10:
            items.append('High_Page_Value')
        elif row.get('PageValues', 0) > 0:
            items.append('Moderate_Page_Value')
            
        # Visitor type
        v_type = str(row.get('VisitorType', ''))
        if 'Returning' in v_type:
            items.append('Returning_Visitor')
        elif 'New' in v_type:
            items.append('New_Visitor')
            
        # Month / Special Day / Weekend
        if row.get('Weekend', False) is True:
            items.append('Weekend_Visit')
        month = row.get('Month', None)
        if month in ['Nov', 'Dec', 'May']:
            items.append(f'Peak_Month_{month}')
            
        # Target: Purchase
        rev = row.get('Revenue', None)
        if rev is True or str(rev).lower() == 'true' or rev == 1:
            items.append('Outcome:Purchase')
        elif rev is False or str(rev).lower() == 'false' or rev == 0:
            items.append('Outcome:No_Purchase')
            
        transactions.append(items)
        
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    one_hot_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    return transactions, one_hot_df, te.columns_

def extract_navigation_sequences(
    df: pd.DataFrame,
    session_col: str = 'session_id',
    page_col: str = 'page_name',
    timestamp_col: Optional[str] = 'timestamp'
) -> Tuple[List[List[str]], pd.DataFrame]:
    """
    Extracts ordered navigation paths per session and builds a first-order Markov
    transition probability matrix (Page A -> Page B).
    
    Preserves exact temporal sequence of user navigation.
    
    Returns:
        Tuple:
            - sequences: List of ordered page sequences per session.
            - transition_matrix: Row-normalized transition probability matrix.
    """
    df_sorted = df.copy()
    if timestamp_col and timestamp_col in df_sorted.columns:
        try:
            df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], errors='coerce')
            df_sorted = df_sorted.sort_values(by=[session_col, timestamp_col])
        except Exception:
            pass
            
    grouped = df_sorted.groupby(session_col)[page_col].apply(list).reset_index()
    sequences = grouped[page_col].tolist()
    
    # Calculate transitions
    transitions: Dict[Tuple[str, str], int] = {}
    all_pages = set()
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            src, dst = str(seq[i]), str(seq[i+1])
            all_pages.add(src)
            all_pages.add(dst)
            transitions[(src, dst)] = transitions.get((src, dst), 0) + 1
            
    if not transitions:
        return sequences, pd.DataFrame()
        
    pages_list = sorted(list(all_pages))
    freq_matrix = pd.DataFrame(0, index=pages_list, columns=pages_list)
    
    for (src, dst), count in transitions.items():
        freq_matrix.loc[src, dst] = count
        
    # Row normalize to get transition probabilities
    row_sums = freq_matrix.sum(axis=1)
    prob_matrix = freq_matrix.div(row_sums.replace(0, 1), axis=0).round(3)
    
    return sequences, prob_matrix

def aggregate_events_to_sessions(
    events_df: pd.DataFrame,
    session_col: str = 'session_id',
    timestamp_col: Optional[str] = 'timestamp',
    page_col: str = 'page_name',
    category_col: Optional[str] = 'page_category',
    event_col: Optional[str] = 'event_type',
    dwell_col: Optional[str] = 'dwell_time_seconds'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms raw chronological clickstream event logs into session-level behavioral features,
    while keeping the raw event dataset separate from derived features.
    
    Derived Features:
        - Administrative, Administrative_Duration
        - Informational, Informational_Duration
        - ProductRelated, ProductRelated_Duration
        - Total_Duration, Total_Pageviews
        - Cart_Interactions, Checkout_Activity
        - BounceRates, ExitRates, PageValues
        - Month, Weekend, VisitorType, Revenue (ground truth if available)
        
    Returns:
        Tuple:
            - session_features_df: Derived session-level DataFrame.
            - cleaned_events_df: Chronologically sorted raw events DataFrame.
    """
    if events_df.empty:
        raise ValueError("Events DataFrame is empty.")
    if session_col not in events_df.columns:
        raise ValueError(f"Session column '{session_col}' not found.")
        
    cleaned_events = events_df.copy()
    
    # Sort chronologically if timestamps available
    if timestamp_col and timestamp_col in cleaned_events.columns:
        try:
            cleaned_events[timestamp_col] = pd.to_datetime(cleaned_events[timestamp_col], errors='coerce')
            cleaned_events = cleaned_events.sort_values(by=[session_col, timestamp_col]).reset_index(drop=True)
        except Exception:
            pass
            
    session_records: List[Dict[str, Any]] = []
    
    for sess_id, s_events in cleaned_events.groupby(session_col, sort=False):
        n_events = len(s_events)
        
        # Categorize pages
        admin_count = 0
        admin_dwell = 0.0
        info_count = 0
        info_dwell = 0.0
        prod_count = 0
        prod_dwell = 0.0
        cart_count = 0
        checkout_count = 0
        has_purchase = False
        
        has_dwell = dwell_col and dwell_col in s_events.columns
        
        for _, row in s_events.iterrows():
            page_str = str(row.get(page_col, '')).lower()
            cat_str = str(row.get(category_col, '')).lower() if category_col and category_col in row else ''
            evt_str = str(row.get(event_col, '')).lower() if event_col and event_col in row else ''
            dwell_val = float(row.get(dwell_col, 30.0)) if has_dwell and pd.notnull(row.get(dwell_col)) else 30.0
            
            # Check purchase outcome
            if evt_str in ['purchase', 'order_complete', 'conversion'] or any(k in page_str for k in ['order_confirmation', 'checkout_success', 'thank_you']):
                has_purchase = True
                
            # Check cart interactions
            if evt_str in ['add_to_cart', 'cart'] or 'cart' in page_str or 'cart' in cat_str:
                cart_count += 1
                
            # Check checkout
            if 'checkout' in page_str or 'checkout' in cat_str or 'shipping' in page_str or 'payment' in page_str:
                checkout_count += 1
                
            # Check admin vs info vs product
            if any(k in page_str or k in cat_str for k in ['account', 'login', 'setting', 'admin', 'profile', 'support', 'help']):
                admin_count += 1
                admin_dwell += dwell_val
            elif any(k in page_str or k in cat_str for k in ['faq', 'info', 'about', 'contact', 'terms', 'policy', 'blog']):
                info_count += 1
                info_dwell += dwell_val
            else:
                prod_count += 1
                prod_dwell += dwell_val
                
        # Total duration
        if has_dwell:
            total_duration = round(admin_dwell + info_dwell + prod_dwell, 2)
        elif timestamp_col and timestamp_col in s_events.columns and s_events[timestamp_col].notnull().all():
            t_min = s_events[timestamp_col].min()
            t_max = s_events[timestamp_col].max()
            total_duration = round(float((t_max - t_min).total_seconds()), 2)
            if total_duration == 0:
                total_duration = float(n_events * 35.0)
        else:
            total_duration = float(n_events * 40.0)
            
        # Rates
        bounce_rate = 0.20 if (n_events == 1 or total_duration < 15.0) else 0.0
        exit_rate = round(min(0.20, 1.0 / max(n_events, 1)), 4)
        
        # Page Values proxy
        if has_purchase:
            page_values = round(float(25.0 + checkout_count * 10.0 + cart_count * 5.0), 2)
        elif checkout_count > 0:
            page_values = round(float(12.0 + cart_count * 4.0), 2)
        elif cart_count > 0:
            page_values = round(float(4.0 + cart_count * 2.0), 2)
        else:
            page_values = 0.0
            
        # Extract Month, Weekend from first timestamp
        month = 'Nov'
        weekend = False
        if timestamp_col and timestamp_col in s_events.columns and s_events[timestamp_col].notnull().any():
            first_ts = s_events[timestamp_col].dropna().iloc[0]
            if isinstance(first_ts, pd.Timestamp):
                month = first_ts.strftime('%b')
                weekend = first_ts.weekday() >= 5
                
        visitor_type = 'Returning_Visitor' if n_events > 3 else 'New_Visitor'
        
        calibrated_prod_count = max(prod_count, int(prod_dwell / 20.0)) if (prod_count > 0 or prod_dwell > 0) else 0
        calibrated_admin_count = max(admin_count, int(admin_dwell / 30.0)) if (admin_count > 0 or admin_dwell > 0) else 0
        calibrated_info_count = max(info_count, int(info_dwell / 30.0)) if (info_count > 0 or info_dwell > 0) else 0

        session_records.append({
            'session_id': str(sess_id),
            'Administrative': calibrated_admin_count,
            'Administrative_Duration': round(admin_dwell, 2),
            'Informational': calibrated_info_count,
            'Informational_Duration': round(info_dwell, 2),
            'ProductRelated': calibrated_prod_count,
            'ProductRelated_Duration': round(prod_dwell, 2),
            'BounceRates': bounce_rate,
            'ExitRates': exit_rate,
            'PageValues': page_values,
            'SpecialDay': 0.0,
            'Month': month,
            'OperatingSystems': 2,
            'Browser': 2,
            'Region': 1,
            'TrafficType': 2,
            'VisitorType': visitor_type,
            'Weekend': weekend,
            'Total_Duration': total_duration,
            'Total_Pageviews': n_events,
            'Cart_Interactions': cart_count,
            'Checkout_Activity': checkout_count,
            'Revenue': has_purchase
        })
        
    session_features_df = pd.DataFrame(session_records)
    return session_features_df, cleaned_events

def prepare_unified_session_data(
    df: pd.DataFrame,
    session_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    page_col: Optional[str] = None,
    category_col: Optional[str] = None,
    event_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Universally adapts any uploaded dataset (raw clickstream events or pre-aggregated
    tabular shoppers dataset) into a standardized session format ready for instant conversion
    prediction, behavioral clustering, and FP-Growth pattern mining on a single page.
    """
    from src.preprocessing import detect_dataset_type, map_clickstream_columns
    
    if df is None or df.empty:
        raise ValueError("Cannot prepare session data from an empty dataset.")
        
    dtype_detected = detect_dataset_type(df)
    
    if dtype_detected == 'clickstream':
        mapping = map_clickstream_columns(df)
        s_col = session_col or mapping.get('session_id') or 'session_id'
        t_col = timestamp_col or mapping.get('timestamp')
        p_col = page_col or mapping.get('page') or 'page_name'
        c_col = category_col or mapping.get('category')
        e_col = event_col or mapping.get('event_type')
        
        df_working = df.copy()
        if s_col not in df_working.columns:
            df_working['session_id'] = 'Session_1'
            s_col = 'session_id'
            
        if p_col not in df_working.columns:
            str_cols = df_working.select_dtypes(include=['object', 'string']).columns.tolist()
            p_col = str_cols[0] if str_cols else df_working.columns[0]
            
        session_features_df, cleaned_events_df = aggregate_events_to_sessions(
            df_working,
            session_col=s_col,
            timestamp_col=t_col,
            page_col=p_col,
            category_col=c_col,
            event_col=e_col
        )
        
        try:
            transactions, one_hot_df, item_names = clickstream_to_transactions(
                cleaned_events_df,
                session_col=s_col,
                item_col=p_col,
                timestamp_col=t_col
            )
        except Exception:
            transactions = [['Home_Browse']]
            one_hot_df = pd.DataFrame({'Home_Browse': [True] * len(session_features_df)})
            item_names = ['Home_Browse']
            
        has_event_stream = True
        has_timestamps = t_col is not None and t_col in cleaned_events_df.columns
        
    else:
        # Tabular / UCI Online Shoppers format
        session_features_df = df.copy()
        
        # Ensure session_id column exists
        candidate_sess = [
            c for c in session_features_df.columns 
            if c.lower() in ['session_id', 'sessionid', 'session', 'id', 'user_id', 'client_id']
        ]
        if candidate_sess:
            s_col = candidate_sess[0]
            session_features_df['session_id'] = session_features_df[s_col].astype(str)
        else:
            s_col = 'session_id'
            session_features_df['session_id'] = [f"Session_{i+1:04d}" for i in range(len(session_features_df))]
            
        # Ensure core numerical columns are present for models
        defaults = {
            'Administrative': 0,
            'Administrative_Duration': 0.0,
            'Informational': 0,
            'Informational_Duration': 0.0,
            'ProductRelated': 1,
            'ProductRelated_Duration': 30.0,
            'BounceRates': 0.0,
            'ExitRates': 0.02,
            'PageValues': 0.0,
            'SpecialDay': 0.0,
            'Month': 'May',
            'OperatingSystems': 2,
            'Browser': 2,
            'Region': 1,
            'TrafficType': 2,
            'VisitorType': 'Returning_Visitor',
            'Weekend': False,
            'Cart_Interactions': 0,
            'Checkout_Activity': 0,
            'Total_Duration': 30.0,
            'Total_Pageviews': 1
        }
        for col, default_val in defaults.items():
            if col not in session_features_df.columns:
                session_features_df[col] = default_val
                
        cleaned_events_df = session_features_df[['session_id', 'VisitorType', 'Month']].copy()
        
        # Extract transactions using shoppers_to_transactions
        try:
            transactions, one_hot_df, item_names = shoppers_to_transactions(session_features_df)
        except Exception:
            transactions = [['Product_Browsed']]
            one_hot_df = pd.DataFrame({'Product_Browsed': [True] * len(session_features_df)})
            item_names = ['Product_Browsed']
            
        has_event_stream = False
        has_timestamps = False
        p_col = 'VisitorType'
        t_col = None
        e_col = None
        
    return {
        'dataset_type': dtype_detected,
        'is_single_session': (len(session_features_df) == 1),
        'session_features_df': session_features_df,
        'cleaned_events_df': cleaned_events_df,
        'transactions': transactions,
        'one_hot_df': one_hot_df,
        'item_names': item_names,
        'session_col': 'session_id',
        'timestamp_col': t_col,
        'page_col': p_col,
        'event_col': e_col,
        'total_sessions': len(session_features_df),
        'total_events': len(cleaned_events_df),
        'has_timestamps': has_timestamps,
        'has_event_stream': has_event_stream
    }

