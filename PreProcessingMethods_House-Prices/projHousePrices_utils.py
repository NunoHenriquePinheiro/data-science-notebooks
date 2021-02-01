from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

import pandas as pd

class AdditionalFeature:
    def __init__(self, label, accuracy):
        self.label = label
        self.accuracy = accuracy
        
    def __repr__(self):
        return f'{self.label}: {self.accuracy}'

def compute_pca(data_frame, components_number=None):
    """
    Summary: Computes PCA for a given dataframe.
    Params: 'data_frame' - The dataframe.
            'components_number' - The number of components to return by the PCA. Defaults to 'None'.
    Returns: 1. A dataframe with the components.
             2. An array with the ratio of explained variance of each component.
    """
    pca_result = PCA(n_components=components_number)
    pca_result.fit(data_frame)
    return pd.DataFrame(pca_result.transform(data_frame)), pca_result.explained_variance_ratio_

def get_ohe_dataframe(data_series, column_prefix):
    """
    Summary: Performs One-Hot Encoding (OHE) in a given data series.
    Params: 'data_series' - The series of data to encode.
            'column_prefix' - Prefix for the names of the newly created encoding columns.
    Returns: A dataframe with the encoded columns.
    """
    df_ohe = pd.get_dummies(data_series)
    df_ohe.columns = [column_prefix + str(col) for col in df_ohe.columns]
    return df_ohe

def get_dataframe_scaled(data_frame, min_limit=0, max_limit=1):
    """
    Summary: Scales a given dataframe to the interval between the given limits.
    Params: 'data_frame' - The dataframe whose series are to scale.
            'min_limit' - Minimum limit of the scaled feature. Defaults to 0.
            'max_limit' - Maximum limit of the scaled feature. Defaults to 1.
    Returns: The scaled dataframe.
    """
    mmscaler = MinMaxScaler(feature_range=(min_limit, max_limit))
    df_scaled = pd.DataFrame(mmscaler.fit_transform(data_frame.values))
    df_scaled.columns = data_frame.columns
    return df_scaled

def get_series_scaled_dataframe(data_series, column_label, min_limit=0, max_limit=1):
    """
    Summary: Scales a given series to the interval between the given limits.
    Params: 'data_series' - The series of data to scale.
            'column_label' - Name/label of the new scaled column to retrieve.
            'min_limit' - Minimum limit of the scaled feature. Defaults to 0.
            'max_limit' - Maximum limit of the scaled feature. Defaults to 1.
    Returns: A dataframe with the scaled feature.
    """
    mmscaler = MinMaxScaler(feature_range=(min_limit, max_limit))
    df_scaled = pd.DataFrame(mmscaler.fit_transform(data_series.to_numpy().reshape(-1, 1)))
    df_scaled.columns = [column_label]
    return df_scaled

def get_dataframe_zscore(data_frame):
    """
    Summary: Scales a given dataframe to its columns' zscores.
    Params: 'data_frame' - The dataframe whose series are to scale.
    Returns: The scaled dataframe.
    """
    zscore = StandardScaler()
    df_zscore = pd.DataFrame(zscore.fit_transform(data_frame.values))
    df_zscore.columns = data_frame.columns
    return df_zscore

def get_series_zscore_dataframe(data_series, column_label):
    """
    Summary: Scales a given series to its zscore.
    Params: 'data_series' - The series of data to scale.
            'column_label' - Name/label of the new scaled column to retrieve.
    Returns: A dataframe with the zscore feature.
    """
    zscore = StandardScaler()
    df_zscore = pd.DataFrame(zscore.fit_transform(data_series.to_numpy().reshape(-1, 1)))
    df_zscore.columns = [column_label]
    return df_zscore
