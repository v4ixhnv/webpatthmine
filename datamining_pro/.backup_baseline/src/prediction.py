"""
Purchase Intent Prediction Module for WEBPATHMINE.
Implements leak-free Scikit-learn Pipeline (preprocessing + DecisionTreeClassifier),
stratified evaluation, model persistence with joblib, and single-session inference.
"""
from typing import Dict, Any, Tuple, List, Optional
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    criterion: str = 'gini',
    max_depth: Optional[int] = 5,
    min_samples_split: int = 5,
    class_weight: Optional[str] = 'balanced',
    random_state: int = 42
) -> Pipeline:
    """
    Constructs an end-to-end leak-free Scikit-Learn pipeline.
    Combines numerical and categorical preprocessing with a DecisionTreeClassifier.
    """
    transformers = []
    
    if numeric_features:
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_transformer, numeric_features))
        
    if categorical_features:
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_transformer, categorical_features))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return pipeline

def train_and_evaluate_model(
    df: pd.DataFrame,
    target_col: str = 'Revenue',
    feature_cols: Optional[List[str]] = None,
    criterion: str = 'gini',
    max_depth: Optional[int] = 5,
    min_samples_split: int = 5,
    class_weight: Optional[str] = 'balanced',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Splits data with stratification, builds pipeline, trains model, and evaluates comprehensively.
    
    Returns:
        Tuple:
            - pipeline: Trained sklearn Pipeline.
            - metrics: Dictionary containing evaluation scores, confusion matrix, and feature importances.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Standardize target values to 0 and 1
    y_raw = df_clean[target_col]
    if y_raw.dtype == bool or set(y_raw.unique()).issubset({True, False, 1, 0, 'True', 'False', '1', '0'}):
        y = y_raw.astype(str).str.lower().map({'true': 1, '1': 1, 'false': 0, '0': 0}).fillna(0).astype(int)
    else:
        # Categorical or continuous: factorize if binary
        uniques = y_raw.unique()
        if len(uniques) != 2:
            raise ValueError(f"Target column must be binary. Found {len(uniques)} unique values.")
        y = (y_raw == uniques[1]).astype(int)
        
    # Check class distribution
    class_counts = y.value_counts().to_dict()
    imbalance_ratio = round(class_counts.get(0, 1) / max(class_counts.get(1, 1), 1), 2)
    
    # Determine features (prevent target leakage)
    if feature_cols is None:
        feature_cols = [c for c in df_clean.columns if c != target_col]
    else:
        feature_cols = [c for c in feature_cols if c != target_col and c in df_clean.columns]
        
    if not feature_cols:
        raise ValueError("No feature columns available for model training.")
        
    X = df_clean[feature_cols].copy()
    
    # Convert booleans to 0/1 integers for numeric consistency
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
    for b_col in bool_cols:
        X[b_col] = X[b_col].astype(int)
        
    # Identify numerical vs categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    for c_col in cat_cols:
        X[c_col] = X[c_col].astype(str)
    
    # Stratified split
    stratify = y if len(y.unique()) > 1 and min(y.value_counts()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Build and fit pipeline
    pipeline = build_pipeline(
        numeric_features=num_cols,
        categorical_features=cat_cols,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state
    )
    
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['classifier'], 'predict_proba') else None
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    roc_auc = None
    if y_prob is not None and len(np.unique(y_test)) == 2:
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            roc_auc = None
            
    cm = confusion_matrix(y_test, y_pred)
    
    # Extract feature importances
    feature_names = []
    preprocessor = pipeline.named_steps['preprocessor']
    
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        ohe_cols = cat_ohe.get_feature_names_out(cat_cols)
        feature_names.extend(ohe_cols)
        
    clf = pipeline.named_steps['classifier']
    importances = clf.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Package results
    results = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'f1_macro': round(f1_macro, 4),
        'roc_auc': round(roc_auc, 4) if roc_auc else None,
        'confusion_matrix': cm.tolist(),
        'feature_importance': importance_df,
        'class_distribution': {
            'No Purchase (0)': int(class_counts.get(0, 0)),
            'Purchase (1)': int(class_counts.get(1, 0))
        },
        'imbalance_ratio': imbalance_ratio,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_columns': feature_cols,
        'num_columns': num_cols,
        'cat_columns': cat_cols,
        'tree_depth': int(clf.get_depth()),
        'tree_leaves': int(clf.get_n_leaves())
    }
    
    return pipeline, results

def save_model(pipeline: Pipeline, filepath: str = "models/decision_tree_pipeline.joblib") -> str:
    """
    Persists trained pipeline using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    return filepath

def load_model(filepath: str = "models/decision_tree_pipeline.joblib") -> Pipeline:
    """
    Loads persisted pipeline.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No persisted model found at {filepath}")
    return joblib.load(filepath)

def predict_single_session(
    pipeline: Pipeline,
    session_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Takes single session input data, runs through pipeline, and returns predicted label and probability.
    """
    input_df = pd.DataFrame([session_dict])
    
    # Standardize booleans and objects
    for col in input_df.select_dtypes(include=['bool']).columns:
        input_df[col] = input_df[col].astype(int)
    for col in input_df.select_dtypes(include=['object', 'category', 'string']).columns:
        input_df[col] = input_df[col].astype(str)
        
    pred = pipeline.predict(input_df)[0]
    
    prob_purchase = None
    prob_no_purchase = None
    if hasattr(pipeline, 'predict_proba'):
        probs = pipeline.predict_proba(input_df)[0]
        # Classes are [0, 1]
        prob_no_purchase = round(float(probs[0]), 4)
        prob_purchase = round(float(probs[1]), 4)
        
    outcome_label = "Purchase" if pred == 1 else "No Purchase"
    
    return {
        'prediction': int(pred),
        'outcome_label': outcome_label,
        'probability_purchase': prob_purchase,
        'probability_no_purchase': prob_no_purchase,
        'disclaimer': (
            "Model predictions represent probabilistic estimates based on historical patterns, "
            "not a guarantee of shopper conversion."
        )
    }

def get_or_train_reference_pipeline(
    criterion: str = 'gini',
    max_depth: Optional[int] = 5,
    class_weight: Optional[str] = 'balanced',
    filepath: str = "models/decision_tree_pipeline.joblib",
    reference_df: Optional[pd.DataFrame] = None
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Reuses an existing trained Decision Tree pipeline, or trains on the reference
    UCI Online Shoppers dataset if not found.
    Never trains a prediction model using only a single uploaded session.
    """
    if os.path.exists(filepath):
        try:
            pipe = load_model(filepath)
            # Check if criterion matches if specified
            clf = pipe.named_steps.get('classifier')
            if clf and clf.criterion == criterion:
                return pipe, {'reused_persisted_model': True}
        except Exception:
            pass
            
    # Train on reference dataset
    if reference_df is None:
        p = "data/online_shoppers_intention.csv"
        if os.path.exists(p):
            reference_df = pd.read_csv(p)
        else:
            raise FileNotFoundError("Reference dataset data/online_shoppers_intention.csv not found.")
            
    pipe, results = train_and_evaluate_model(
        reference_df,
        target_col='Revenue',
        criterion=criterion,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42
    )
    save_model(pipe, filepath)
    return pipe, results

def batch_predict_sessions(
    pipeline: Pipeline,
    session_df: pd.DataFrame,
    target_col: Optional[str] = 'Revenue'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Predicts Purchase or No Purchase on a batch of sessions using a pre-trained Pipeline.
    
    Guarantees:
        - Prevents target leakage by excluding actual purchase outcome from prediction inputs.
        - Matches feature schema expected by the pipeline.
        - Calculates evaluation metrics only when valid ground-truth labels are present.
        - Never fabricates evaluation metrics when ground-truth is absent.
        
    Returns:
        Tuple:
            - predicted_df: Copy of session_df with predictions and probabilities.
            - summary: Prediction counts, rates, and optional evaluation metrics.
    """
    if session_df.empty:
        raise ValueError("Session DataFrame is empty.")
        
    df_pred = session_df.copy()
    
    # Check for ground-truth labels
    has_ground_truth = False
    y_true = None
    if target_col and target_col in df_pred.columns:
        raw_target = df_pred[target_col]
        if raw_target.notnull().any():
            has_ground_truth = True
            if raw_target.dtype == bool or set(raw_target.dropna().unique()).issubset({True, False, 1, 0, 'True', 'False', '1', '0'}):
                y_true = raw_target.astype(str).str.lower().map({'true': 1, '1': 1, 'false': 0, '0': 0}).fillna(0).astype(int)
            else:
                has_ground_truth = False
                
    # PREVENT TARGET LEAKAGE: Exclude target and outcome columns from features
    leakage_cols = [c for c in df_pred.columns if c.lower() in [
        'revenue', 'purchased', 'converted', 'target', 'label', 'outcome', 'outcome:purchase'
    ]]
    X_input = df_pred.drop(columns=[c for c in leakage_cols if c in df_pred.columns], errors='ignore').copy()
    
    # Identify expected features from pipeline preprocessor
    preprocessor = pipeline.named_steps.get('preprocessor')
    expected_num = []
    expected_cat = []
    
    if preprocessor and hasattr(preprocessor, 'transformers_'):
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                expected_num.extend(cols)
            elif name == 'cat':
                expected_cat.extend(cols)
                
    # Schema alignment: ensure all expected columns are present
    for col in expected_num:
        if col not in X_input.columns:
            X_input[col] = 0.0
        else:
            X_input[col] = pd.to_numeric(X_input[col], errors='coerce').fillna(0.0)
            
    for col in expected_cat:
        if col not in X_input.columns:
            X_input[col] = 'Nov' if col == 'Month' else ('New_Visitor' if col == 'VisitorType' else 'Other')
        else:
            X_input[col] = X_input[col].astype(str).fillna('Other')
            
    # Standardize booleans and objects
    for col in X_input.select_dtypes(include=['bool']).columns:
        X_input[col] = X_input[col].astype(int)
    for col in X_input.select_dtypes(include=['object', 'category', 'string']).columns:
        X_input[col] = X_input[col].astype(str)
        
    # Reorder or select columns expected by preprocessor
    all_expected = expected_num + expected_cat
    if all_expected:
        X_aligned = X_input[all_expected].copy()
    else:
        X_aligned = X_input.copy()
        
    # Run prediction through pipeline
    preds = pipeline.predict(X_aligned)
    pred_labels = ["Purchase" if p == 1 else "No Purchase" for p in preds]
    
    probs_purchase = None
    probs_no_purchase = None
    if hasattr(pipeline, 'predict_proba'):
        prob_arr = pipeline.predict_proba(X_aligned)
        probs_no_purchase = [round(float(p[0]), 4) for p in prob_arr]
        probs_purchase = [round(float(p[1]), 4) for p in prob_arr]
        
    # Append predictions to output DataFrame
    df_pred['Predicted_Conversion'] = [bool(p) for p in preds]
    df_pred['Predicted_Intent'] = pred_labels
    if probs_purchase is not None:
        df_pred['Purchase_Probability'] = probs_purchase
        df_pred['No_Purchase_Probability'] = probs_no_purchase
        
    # Summary telemetry
    total_sess = len(df_pred)
    purchase_count = int(sum(preds))
    no_purchase_count = total_sess - purchase_count
    conversion_rate = round(purchase_count / total_sess * 100, 2) if total_sess > 0 else 0.0
    
    summary: Dict[str, Any] = {
        'total_sessions': total_sess,
        'predicted_purchases': purchase_count,
        'predicted_no_purchases': no_purchase_count,
        'predicted_conversion_rate': conversion_rate,
        'has_ground_truth': has_ground_truth,
        'evaluation': None
    }
    
    # Ground truth evaluation metrics (NEVER fabricated)
    if has_ground_truth and y_true is not None:
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds).tolist()
        
        summary['evaluation'] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': cm,
            'true_purchases': int(y_true.sum()),
            'true_no_purchases': int((y_true == 0).sum())
        }
        
    return df_pred, summary
