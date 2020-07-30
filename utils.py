import numpy as np
from geopy.geocoders import Nominatim

from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
from nltk.corpus import stopwords 

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import iqr
from sklearn.impute import SimpleImputer

import pandas as pd
from datetime import datetime

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]

class TextSelector2(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]

class ModifiedSimpleImputer(SimpleImputer):
    def transform(self, X):
        return super().transform(X).flatten()

class NumberWords(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        s = X[self.field].apply(lambda x: len(Tokenizer(x)))
        return np.vstack(s.values)
    
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words

class QuantizeCat(object):
    def __init__(self, features, treshold):
        if type(features) is not list:
            self.features = [features]
        else:
            self.features = features
        
        self.treshold = treshold
        self.features_classes = {}

    def _get_classes(self, X, feature):
            X_copy = X.copy()
            X_temp = X_copy[feature].value_counts().to_frame().reset_index()
            X_temp['%'] = X_temp[feature].cumsum()/X_temp[feature].sum()
            self.features_classes[feature] = X_temp.loc[X_temp['%']< self.treshold, 'index'].to_list()

    def fit(self, X, y=None):
        for feature in self.features:
            self._get_classes(X, feature)
        return self

    def transform(self, X):
        for feature in self.features:
            X_copy = X.copy()
            X_copy.loc[:, feature] = X_copy[feature].apply(
                lambda x: x 
                if x in self.features_classes[feature] 
                else 'Other')
        return X_copy


class DateProcess(object):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X.loc[:, self.field] = X.loc[:, self.field].astype('datetime64[ns]').copy()
        
        X_temp = pd.DataFrame()
        X_temp[f'{self.field}_day'] = X[self.field].dt.day
        X_temp[f'{self.field}_month'] = X[self.field].dt.month
        X_temp[f'{self.field}_year'] = X[self.field].dt.year
        X_temp[f'{self.field}_quarter'] = X[self.field].dt.quarter
        X_temp[f'{self.field}_dayofweek'] = X[self.field].dt.dayofweek
        X_temp[f'{self.field}_dayofyear'] = X[self.field].dt.dayofyear
        return X_temp

# class SpecialCate(object):
#     def fit(self, X, y=None):

#     def transform(self, X):
        

class RemoveOutliers(object):
    def __init__(self, features):
        if type(features) is not list:
            self.features = [features]
        else:
            self.features = features

        self.threshold = 1.5
        self.outliers_limits = {}
        self.truncate_min_max = {}
    
    def fit(self, X, y=None):

        """
        Find the outliers_limits dict with the min and max values
        for each column to be considered an outlier, using iqr.
        """
        for feature in self.features:
            data = X[feature]
            iqr_value = iqr(data)
            q1 = np.quantile(data, q=0.25, axis=0)
            q3 = np.quantile(data, q=0.75, axis=0)
            min_tresh = q1 - self.threshold * iqr_value
            max_tresh = q3 + self.threshold * iqr_value
            self.outliers_limits[feature] = (min_tresh, max_tresh)
        
        self._fit_truncate(X)
        return self

    def _get_outlier_mask(self, X):
        outlier_mask = X[self.features].copy()
        for col in self.outliers_limits:
            condition = (X[col] < self.outliers_limits[col][0]) | (
                    X[col] > self.outliers_limits[col][1])
            outlier_mask[col] = condition
       
        return outlier_mask

    def _fit_truncate(self, X):
        mask = self._get_outlier_mask(X)
        for feature in self.features:
            min_value = X[feature].loc[~mask[feature]].min()
            max_value = X[feature].loc[~mask[feature]].max()
            self.truncate_min_max[feature] = (min_value, max_value)
        return self
        
    def transform(self, X):
        return self._transform_truncate(X)
    
    def _transform_truncate(self, X):
        X_copy = X.copy()
        mask = self._get_outlier_mask(X_copy)
        X_copy = X_copy[self.features]
        for feature in self.features:
            min_tresh = self.outliers_limits[feature][0]
            min_value = self.truncate_min_max[feature][0]
            max_value = self.truncate_min_max[feature][1]
            outlier_indexes = mask.index[mask[feature]].tolist()
            if outlier_indexes:
                for index in outlier_indexes:
                    if X_copy[feature][index] <= min_tresh:
                        X_copy.at[index, feature] = min_value
                    else:
                        X_copy.at[index, feature] = max_value
        return X_copy


def remove_outliers_iqr_label(series):
    label = series.copy()
    Q1 = label.quantile(0.25)
    Q3 = label.quantile(0.75)
    IQR = Q3 - Q1
    mask = (label.values > (Q1 - 1.5 * IQR)) & (label.values < (Q3 + 1.5 * IQR))
    min_value = label.loc[mask].min()
    max_value = label.loc[mask].max()
    outlier_indexes = np.argwhere(~mask)[:,0]
    for index in outlier_indexes:
        if label.iloc[index] <= Q1:
            label.iloc[index] = Q1
        else:
            label.iloc[index] = Q3
    return label
    


def get_address(lat, lon):
    """
    Example
    'lat': '41.245308550014954',
    'lon': '-75.85359729855945',
    """
    geolocator = Nominatim(timeout=10000)
    location = geolocator.reverse(f"{lat},{lon}")
    return location.raw['address']
