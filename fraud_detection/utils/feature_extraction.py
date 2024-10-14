# feature_extraction.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def extract_features(data, method='pca', n_components=2):
    """
    Extract features from the data using PCA or t-SNE.

    Parameters:
    - data: Data to be transformed.
    - method: Method to use for feature extraction (default: 'pca').
    - n_components: Number of components to retain (default: 2).

    Returns:
    - transformed_data: Transformed data.
    """
    if method == 'pca':
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
    elif method == 'tsne':
        tsne = TSNE(n_components=n_components, random_state=42)
        transformed_data = tsne.fit_transform(data)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    return transformed_data
