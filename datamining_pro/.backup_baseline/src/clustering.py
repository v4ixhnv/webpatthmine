"""
Behavioral Segmentation Module for WEBPATHMINE.
Implements K-Means clustering, silhouette validation, inertia elbow curve,
PCA dimensionality reduction, and objective data-grounded cluster profiling.
"""
import os
import joblib
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

DEFAULT_BEHAVIORAL_FEATURES = [
    'Administrative_Duration',
    'Informational_Duration',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues'
]

def prepare_clustering_data(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Selects numerical behavioral features, handles missing values, and standardizes data.
    
    Parameters:
        df: Input DataFrame.
        feature_cols: Selected numerical columns.
        
    Returns:
        Tuple:
            - scaled_data: Scaled numpy array.
            - scaler: Fitted StandardScaler.
            - valid_df: DataFrame subset containing clean rows.
    """
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Features not found in dataset: {missing_cols}")
        
    subset = df[feature_cols].copy()
    
    # Impute or drop nulls for clustering
    if subset.isnull().any().any():
        subset = subset.fillna(subset.median())
        
    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset)
    
    return scaled, scaler, subset

def compute_elbow_and_silhouette(
    scaled_data: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes Inertia and Silhouette scores across a range of k values
    to assist with optimal cluster selection.
    """
    k_values = list(k_range)
    inertias: List[float] = []
    silhouette_scores: List[Optional[float]] = []
    
    n_samples = scaled_data.shape[0]
    
    for k in k_values:
        if k >= n_samples:
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(scaled_data)
        inertias.append(float(km.inertia_))
        
        if 2 <= k < n_samples:
            score = float(silhouette_score(scaled_data, km.labels_))
            silhouette_scores.append(round(score, 4))
        else:
            silhouette_scores.append(None)
            
    return {
        'k_values': k_values[:len(inertias)],
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }

def fit_kmeans(
    scaled_data: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray, Optional[float]]:
    """
    Fits KMeans clustering model.
    
    Returns:
        Tuple:
            - model: Fitted KMeans instance.
            - labels: Cluster assignments for each row.
            - silhouette: Silhouette score (if valid, else None).
    """
    n_samples = scaled_data.shape[0]
    if n_clusters < 2 or n_clusters >= n_samples:
        raise ValueError(f"Number of clusters must be between 2 and {n_samples - 1}.")
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    
    sil_score = None
    if n_samples > n_clusters:
        try:
            sil_score = float(silhouette_score(scaled_data, labels))
        except Exception:
            sil_score = None
            
    return kmeans, labels, sil_score

def project_pca(
    scaled_data: np.ndarray,
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, List[float]]:
    """
    Reduces scaled feature space to 2 or 3 principal components for visualization.
    """
    pca = PCA(n_components=min(n_components, scaled_data.shape[1]), random_state=random_state)
    coords = pca.fit_transform(scaled_data)
    var_ratios = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    return coords, var_ratios

def generate_cluster_profiles(
    raw_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generates quantitative profiles and objective descriptions for each cluster.
    Strictly grounds all behavioral descriptions in empirical metrics.
    
    Returns:
        Tuple:
            - mean_profile_df: Mean values per cluster.
            - size_df: Cluster counts and proportions.
            - descriptions: Objective qualitative notes based on z-scores.
    """
    df_profile = raw_df[feature_cols].copy()
    df_profile['Cluster'] = [f"Cluster {l}" for l in labels]
    
    # Cluster sizes
    size_counts = pd.Series(labels).value_counts().sort_index()
    size_pcts = (size_counts / len(labels) * 100).round(2)
    size_df = pd.DataFrame({
        'Cluster': [f"Cluster {k}" for k in size_counts.index],
        'Session Count': size_counts.values,
        'Percentage (%)': size_pcts.values
    })
    
    # Feature means per cluster
    mean_profile = df_profile.groupby('Cluster').mean().round(3)
    
    # Dataset overall means & std for relative comparisons
    overall_mean = raw_df[feature_cols].mean()
    overall_std = raw_df[feature_cols].std().replace(0, 1e-6)
    
    # Generate objective data-backed descriptions
    descriptions: List[Dict[str, Any]] = []
    
    for c_id in sorted(np.unique(labels)):
        c_name = f"Cluster {c_id}"
        c_means = mean_profile.loc[c_name]
        
        key_traits = []
        for feat in feature_cols:
            diff_z = (c_means[feat] - overall_mean[feat]) / overall_std[feat]
            if diff_z >= 0.75:
                key_traits.append(f"Substantially higher {feat} (avg: {c_means[feat]:.2f} vs overall {overall_mean[feat]:.2f})")
            elif diff_z <= -0.75:
                key_traits.append(f"Substantially lower {feat} (avg: {c_means[feat]:.2f} vs overall {overall_mean[feat]:.2f})")
                
        if not key_traits:
            key_traits.append("Behavioral metrics align closely with dataset baseline averages.")
            
        descriptions.append({
            'cluster': c_name,
            'size': int(size_counts[c_id]),
            'pct': float(size_pcts[c_id]),
            'characteristics': key_traits
        })
        
    return mean_profile, size_df, descriptions

def save_clustering_model(
    model: KMeans,
    scaler: StandardScaler,
    feature_cols: List[str],
    filepath: str = "models/kmeans_model.joblib"
) -> str:
    """Saves fitted KMeans model, scaler, and feature list to joblib bundle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    joblib.dump(bundle, filepath)
    return filepath

def load_clustering_model(filepath: str = "models/kmeans_model.joblib") -> Tuple[KMeans, StandardScaler, List[str]]:
    """Loads fitted KMeans model bundle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Clustering model not found at {filepath}")
    bundle = joblib.load(filepath)
    return bundle['model'], bundle['scaler'], bundle['feature_cols']

def get_or_train_reference_clustering(
    reference_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    n_clusters: int = 3,
    filepath: str = "models/kmeans_model.joblib"
) -> Tuple[KMeans, StandardScaler, List[str]]:
    """
    Reuses existing pre-trained KMeans model, or trains on reference dataset if missing.
    Never trains a new clustering model on a single uploaded session.
    """
    if os.path.exists(filepath):
        try:
            return load_clustering_model(filepath)
        except Exception:
            pass
            
    # Train on reference dataset
    if reference_df is None:
        p = "data/online_shoppers_intention.csv"
        if os.path.exists(p):
            reference_df = pd.read_csv(p)
        else:
            raise FileNotFoundError("Reference dataset data/online_shoppers_intention.csv not found.")
            
    cols = feature_cols or [f for f in DEFAULT_BEHAVIORAL_FEATURES if f in reference_df.columns]
    scaled, scaler, _ = prepare_clustering_data(reference_df, cols)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(scaled)
    
    save_clustering_model(km, scaler, cols, filepath)
    return km, scaler, cols

def assign_clusters(
    model: KMeans,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, Optional[float], np.ndarray, List[float]]:
    """
    Standardizes features and assigns sessions to clusters using the pre-trained model.
    Guarantees:
        - Works for any session batch size (1 to N).
        - Does not retrain the model on uploaded data.
        - Computes silhouette score only if valid (N >= 2 and distinct clusters).
        
    Returns:
        Tuple:
            - cluster_labels: np.ndarray of cluster assignments
            - silhouette: Optional float silhouette score
            - pca_coords: 2D projection coords for visualization
            - var_ratios: PCA explained variance ratios
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
        
    # Align features with expected schema
    subset = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            subset[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            subset[col] = 0.0
            
    # Standardize using pre-fitted scaler
    scaled = scaler.transform(subset)
    
    # Predict clusters using pre-trained model
    labels = model.predict(scaled)
    
    # Calculate silhouette score if valid
    n_samples = len(df)
    unique_clusters = len(np.unique(labels))
    sil_score = None
    if n_samples >= 2 and 2 <= unique_clusters < n_samples:
        try:
            sil_score = round(float(silhouette_score(scaled, labels)), 4)
        except Exception:
            sil_score = None
            
    # 2D PCA for visual mapping
    n_comp = min(2, scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    if n_samples >= 2:
        pca_coords = pca.fit_transform(scaled)
        var_ratios = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
        if n_comp == 1:
            pca_coords = np.column_stack([pca_coords, np.zeros(n_samples)])
            var_ratios.append(0.0)
    else:
        pca_coords = np.zeros((1, 2))
        var_ratios = [100.0, 0.0]
        
    return labels, sil_score, pca_coords, var_ratios
