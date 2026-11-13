"""
Data Collection, Inspection, and Preprocessing Module for WEBPATHMINE.
Handles file ingestion, schema validation, data quality audits, and cleaning.
"""
import json
from typing import Dict, Any, Tuple, List, Optional
import io
import pandas as pd
import numpy as np

UCI_ONLINE_SHOPPERS_COLUMNS = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
    'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
    'VisitorType', 'Weekend', 'Revenue'
]

CLICKSTREAM_CANDIDATE_COLUMNS = {
    'session_id': ['session_id', 'sessionid', 'session', 'sess_id', 'user_session', 'visit_id'],
    'timestamp': ['timestamp', 'time', 'datetime', 'event_time', 'created_at', 'date_time', 'event_timestamp'],
    'page': ['page_name', 'page_path', 'page', 'url', 'page_url', 'screen', 'item_name', 'path'],
    'category': ['page_category', 'category', 'section', 'group', 'page_type'],
    'event_type': ['event_type', 'event', 'action', 'type', 'event_name', 'activity']
}

def load_dataset(source: Any) -> pd.DataFrame:
    """
    Load dataset from CSV or JSON file path, StringIO, buffer, or Streamlit UploadedFile.
    
    Parameters:
        source: File path (str), buffer, or UploadedFile.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
        
    Raises:
        ValueError: If file cannot be read or is empty.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
        
    df = None
    file_name = getattr(source, 'name', '') or (source if isinstance(source, str) else '')
    is_json_ext = str(file_name).lower().endswith('.json') or str(file_name).lower().endswith('.jsonl')
    
    try:
        if is_json_ext:
            if hasattr(source, 'read'):
                content = source.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                if hasattr(source, 'seek'):
                    source.seek(0)
                try:
                    data = json.loads(content)
                    df = pd.DataFrame(data)
                except Exception:
                    df = pd.read_json(io.StringIO(content))
            else:
                df = pd.read_json(source)
        else:
            # Try CSV first
            if hasattr(source, 'read'):
                content = source.read()
                if isinstance(content, bytes):
                    text_sample = content[:500].decode('utf-8', errors='ignore').strip()
                else:
                    text_sample = str(content)[:500].strip()
                    
                if hasattr(source, 'seek'):
                    source.seek(0)
                    
                # Check if raw text looks like JSON array or object
                if text_sample.startswith('[') or text_sample.startswith('{'):
                    try:
                        raw_str = content.decode('utf-8', errors='replace') if isinstance(content, bytes) else str(content)
                        data = json.loads(raw_str)
                        df = pd.DataFrame(data)
                    except Exception:
                        df = pd.read_csv(source)
                else:
                    df = pd.read_csv(source)
            elif isinstance(source, str):
                trimmed = source.strip()
                if trimmed.startswith('[') or trimmed.startswith('{'):
                    data = json.loads(trimmed)
                    df = pd.DataFrame(data)
                else:
                    df = pd.read_csv(source)
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")
    except Exception as e:
        # Fallback trial with read_json if read_csv failed
        try:
            if hasattr(source, 'seek'):
                source.seek(0)
            df = pd.read_json(source)
        except Exception:
            raise ValueError(f"Failed to parse uploaded dataset (supported: CSV, JSON): {str(e)}") from e
        
    if df is None or df.empty:
        raise ValueError("The provided dataset is empty (0 rows).")
        
    return df

def validate_activity_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates website activity / clickstream dataset schema, columns, timestamps, and integrity.
    
    Returns:
        Dict[str, Any]: Comprehensive validation results with clear error and warning messages.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    col_mapping = map_clickstream_columns(df)
    session_col = col_mapping.get('session_id')
    timestamp_col = col_mapping.get('timestamp')
    page_col = col_mapping.get('page')
    category_col = col_mapping.get('category')
    event_col = col_mapping.get('event_type')
    
    # 1. Session Column Validation
    if not session_col:
        errors.append(
            "Missing Session Identifier column. Required a column identifying user sessions "
            f"(e.g., {', '.join(CLICKSTREAM_CANDIDATE_COLUMNS['session_id'])})."
        )
        total_sessions = 0
    else:
        total_sessions = int(df[session_col].nunique())
        if total_sessions == 0:
            errors.append(f"Session column '{session_col}' contains zero unique sessions.")
        elif total_sessions == 1:
            warnings.append(
                "Only 1 unique session detected. Pattern mining and clustering provide better insights "
                "across multiple distinct sessions."
            )
            
    # 2. Page / Activity Column Validation
    if not page_col:
        errors.append(
            "Missing Page / Action column. Required a column identifying visited pages or actions "
            f"(e.g., {', '.join(CLICKSTREAM_CANDIDATE_COLUMNS['page'])})."
        )
        
    # 3. Timestamp Validation
    valid_timestamp_pct = 0.0
    if not timestamp_col:
        warnings.append(
            "No explicit timestamp column detected. Chronological ordering will rely on the dataset's "
            "original row index sequence."
        )
    else:
        parsed_ts = pd.to_datetime(df[timestamp_col], errors='coerce')
        valid_ts_count = int(parsed_ts.notnull().sum())
        total_rows = len(df)
        valid_timestamp_pct = round((valid_ts_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        
        if valid_timestamp_pct < 50.0:
            errors.append(
                f"Timestamp column '{timestamp_col}' contains more than 50% invalid or unparseable timestamps "
                f"({valid_timestamp_pct}% valid). Supported formats: ISO 8601, YYYY-MM-DD HH:MM:SS."
            )
        elif valid_timestamp_pct < 95.0:
            warnings.append(
                f"Timestamp column '{timestamp_col}' has {100.0 - valid_timestamp_pct:.1f}% unparseable dates. "
                "Invalid rows will maintain file order."
            )
            
    # 4. Null & Duplicate Checks
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        warnings.append(f"Detected {dup_count} duplicate event rows ({dup_count / len(df) * 100:.1f}% of data).")
        
    null_rows = int(df.isnull().any(axis=1).sum())
    if null_rows > 0:
        warnings.append(f"Dataset contains {null_rows} rows with missing values.")
        
    is_valid = len(errors) == 0
    
    return {
        'is_valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'session_col': session_col,
        'timestamp_col': timestamp_col,
        'page_col': page_col,
        'category_col': category_col,
        'event_col': event_col,
        'total_events': len(df),
        'total_sessions': total_sessions,
        'valid_timestamp_pct': valid_timestamp_pct,
        'columns_found': df.columns.tolist()
    }

def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive audit of dataset schema, nulls, duplicates, and distributions.
    
    Parameters:
        df: Input DataFrame.
        
    Returns:
        Dict[str, Any]: Detailed metrics and summary data.
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    
    # Missing values
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / n_rows * 100).round(2)
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': null_counts.values,
        'Missing %': null_percentages.values,
        'Dtype': df.dtypes.astype(str).values
    })
    total_missing_cells = int(null_counts.sum())
    missing_cell_pct = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
    
    # Duplicates
    dup_count = int(df.duplicated().sum())
    dup_pct = round((dup_count / n_rows * 100), 2) if n_rows > 0 else 0.0
    
    # Column typing
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    # Target candidate detection
    potential_targets = [c for c in df.columns if c.lower() in ['revenue', 'purchased', 'converted', 'target', 'label', 'buy', 'purchase']]
    
    # Statistics
    numeric_stats = df.describe().T if len(numeric_cols) > 0 else pd.DataFrame()
    cat_plus_bool = [c for c in categorical_cols + boolean_cols if c in df.columns]
    if cat_plus_bool:
        categorical_stats = df[cat_plus_bool].astype(str).describe().T
    else:
        categorical_stats = pd.DataFrame()
    
    return {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'total_cells': total_cells,
        'total_missing_cells': total_missing_cells,
        'missing_cell_pct': missing_cell_pct,
        'duplicate_count': dup_count,
        'duplicate_pct': dup_pct,
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'boolean_columns': boolean_cols,
        'missing_summary': missing_summary,
        'numeric_stats': numeric_stats,
        'categorical_stats': categorical_stats,
        'potential_targets': potential_targets
    }

def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Identifies whether dataset is UCI Online Shoppers, Clickstream event log, or Generic.
    """
    cols_lower = [c.lower() for c in df.columns]
    
    # Check for UCI Online Shoppers
    uci_matches = sum(1 for c in ['administrative', 'productrelated', 'pagevalues', 'revenue', 'bouncerates'] if c in cols_lower)
    if uci_matches >= 3:
        return 'uci_online_shoppers'
        
    # Check for Clickstream logs
    has_session = any(c in cols_lower for c in ['session_id', 'sessionid', 'session', 'sess_id'])
    has_page = any(c in cols_lower for c in ['page_path', 'page_name', 'page', 'url', 'screen'])
    if has_session and has_page:
        return 'clickstream'
        
    return 'generic'

def map_clickstream_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Attempts to auto-map clickstream columns.
    """
    mapping: Dict[str, Optional[str]] = {
        'session_id': None,
        'timestamp': None,
        'page': None,
        'category': None,
        'event_type': None
    }
    cols = df.columns.tolist()
    cols_lower = {c.lower(): c for c in cols}
    
    for key, candidates in CLICKSTREAM_CANDIDATE_COLUMNS.items():
        for cand in candidates:
            if cand in cols_lower:
                mapping[key] = cols_lower[cand]
                break
                
    return mapping

def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = False,
    missing_strategy: str = 'none',
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans dataset while preserving data integrity and returning an audit report.
    Guarantees no silent large-scale data deletion.
    
    Parameters:
        df: Input DataFrame.
        drop_duplicates: If True, drops duplicate rows.
        missing_strategy: 'none', 'drop_rows', 'impute'.
        target_col: Target column to protect or format.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and transformation report.
    """
    cleaned = df.copy()
    initial_rows = len(cleaned)
    report: Dict[str, Any] = {
        'initial_rows': initial_rows,
        'duplicates_removed': 0,
        'missing_rows_dropped': 0,
        'imputed_columns': [],
        'transformations': []
    }
    
    if drop_duplicates:
        dup_count = cleaned.duplicated().sum()
        if dup_count > 0:
            cleaned = cleaned.drop_duplicates()
            report['duplicates_removed'] = int(dup_count)
            report['transformations'].append(f"Removed {dup_count} exact duplicate rows.")
            
    if missing_strategy == 'drop_rows':
        before = len(cleaned)
        cleaned = cleaned.dropna()
        dropped = before - len(cleaned)
        report['missing_rows_dropped'] = dropped
        report['transformations'].append(f"Dropped {dropped} rows with missing values.")
    elif missing_strategy == 'impute':
        # Impute numeric with median, categoricals with mode
        for col in cleaned.columns:
            if cleaned[col].isnull().sum() > 0:
                if col == target_col:
                    continue  # Do not impute target column to prevent target leakage/bias
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    median_val = float(cleaned[col].median())
                    cleaned[col] = cleaned[col].fillna(median_val)
                    report['imputed_columns'].append(f"{col} (median: {median_val:.2f})")
                else:
                    mode_val = cleaned[col].mode()
                    if not mode_val.empty:
                        fill_val = mode_val[0]
                        cleaned[col] = cleaned[col].fillna(fill_val)
                        report['imputed_columns'].append(f"{col} (mode: {fill_val})")
        if report['imputed_columns']:
            report['transformations'].append(f"Imputed missing values for: {', '.join(report['imputed_columns'])}")
            
    # Format target column if present
    if target_col and target_col in cleaned.columns:
        if cleaned[target_col].dtype == bool or set(cleaned[target_col].dropna().unique()).issubset({0, 1, '0', '1', 'True', 'False', True, False}):
            # Standardize boolean target
            cleaned[target_col] = cleaned[target_col].astype(bool)
            report['transformations'].append(f"Standardized target column '{target_col}' as boolean.")
            
    report['final_rows'] = len(cleaned)
    return cleaned, report
