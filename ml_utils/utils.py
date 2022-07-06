#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:45:27 2022

@author: luigi
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import datetime


def merge_categorical_values(df, feature, min_count):
    
    def merge_fun(df_row, feature, values_to_merge):
        if df_row[feature] in values_to_merge:
            return "Other"
        else:
            return df_row[feature]
        
    values_to_merge = df[feature].value_counts()[df[feature].value_counts()<min_count].index.to_list()
    values_unique = list(df[feature].unique())
            
    df[feature+"_merged"] = df.apply(merge_fun, axis=1, feature=feature, values_to_merge=values_to_merge)
    
    feature_map = {value: value for value in values_unique if value not in values_to_merge}
    for value in values_to_merge:
        feature_map[value] = "Other"
    return df, feature_map



def preprocess_dataset(dataset, features_all, features_oneHotEncode, features_ordinalEncode, features_standardize, 
                       target_label, drop_oneHotEncoder='if_binary',
                       val_size=0.2, test_size=0.2, stratify_by_y=False, encode_y=True,
                       random_seed=42):   
    '''
    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset aggregato
    features_all : list
        lista di feature da considerare, e.g. features_all = ['COMMODITY', 'Anno_Nascita', 'popresidente', 'tipocomune', 'firma_day']
    features_oneHotEncode : list
        lista di feature sui cui eseguire il one-hot-encoding, e.g. features_oneHotEncode = ['COMMODITY', 'tipocomune']
    features_ordinalEncode : list
        ...
    features_standardize : list
        lista di feature sui cui eseguire la standardizzazione, e.g. features_standardize = ['Anno_Nascita', 'firma_day']
    drop_oneHotEncoder : string
        drop type of one hot encoder: {'first', 'if_binary', None}
    target_label : string
        target column name
    val_size: float
        val_size (only for split_by_date=False), e.g. 0.2
    test_size : float
        test_size (only for split_by_date=False), e.g. 0.2
    random_seed : int
        random seed for the train/val/test split procedure (only for split_by_date=False)
        
    Returns
    -------
    (X_train_preprocessed, y_train_encoded) : tuple
        training numpy arrays
    (X_val_preprocessed, y_val_encoded) : tuple
        training numpy arrays
    (X_test_preprocessed, y_test_encoded) : tuple
        training numpy arrays
    (n_features, n_output) : tuple
        number of output (encoded) features, number of output classes
    (idxs_train, idxs_val, idxs_test) : tuple
        dataset indexes for train, val and test sets
    (enc, scaler, le) : tuple
        sklearn.preprocessing objects used for the features preprocessing
    features_preprocessed : list
        list of encoded output features
        
    Examples
    --------
    >>> from sklearn import datasets
    >>> import pandas as pd
    >>> from utils import preprocess_dataset
    
    >>> boston = datasets.load_boston()
    
    >>> df = pd.DataFrame([
        ['green', 'XL', 7.2, 'class1'],
        ['blue', 'L', 2.1, 'class2'], 
        ['red', 'S', 9.2, 'class3'],
        ['blue', 'L', 6.2, 'class2'], 
        ['red', 'L', 6.2, 'class3'], 
        ['green', 'L', 6.2, 'class2'], 
        ['green', 'L', 6.2, 'class1']])
    
    >>> df.columns = ['color', 'size', 'price', 'classlabel']
    
    >>> ordinal_encoding_map = {
        'XL': 3,
        'L': 2,
        'M': 1,
        'S': 0}
    
    >>> df['size'] = df['size'].map(ordinal_encoding_map)
    
    >>> preprocessed_data = preprocess_dataset(dataset=df,
                       features_all=['color', 'size', 'price'],
                       features_oneHotEncode=['color'],
                       features_standardize=['size', 'price'],
                       target_label='classlabel')
    '''
    
    features_to_preprocess = features_oneHotEncode + features_ordinalEncode + features_standardize
    features_untouch = [feature for feature in features_all if (feature not in features_to_preprocess)]

    X = dataset[features_all]
    y = dataset[target_label]
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=val_size+test_size, random_state=random_seed, stratify=y if stratify_by_y else None)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size/(val_size+test_size), random_state=random_seed, stratify=y_val_test if stratify_by_y else None)

    idxs_train = X_train.index
    idxs_val = X_val.index
    idxs_test = X_test.index
    
    
    X_train_preprocessed = np.zeros((X_train.shape[0], 0))
    X_val_preprocessed = np.zeros((X_val.shape[0], 0))
    X_test_preprocessed = np.zeros((X_test.shape[0], 0))
    
    features_preprocessed = []
    scaler = StandardScaler()
    enc_onehot = OneHotEncoder(sparse=False, drop=drop_oneHotEncoder)
    enc_ordinal = OrdinalEncoder()
    
    # OneHotEncoding
    if features_oneHotEncode:
        
        X_train_oneHotEncode = X_train[features_oneHotEncode]
        X_val_oneHotEncode = X_val[features_oneHotEncode]
        X_test_oneHotEncode = X_test[features_oneHotEncode]
        
        X_train_oneHotEncoded = enc_onehot.fit_transform(X_train_oneHotEncode)
        X_val_oneHotEncoded = enc_onehot.transform(X_val_oneHotEncode)
        X_test_oneHotEncoded = enc_onehot.transform(X_test_oneHotEncode)
        
        X_train_preprocessed = np.concatenate((X_train_preprocessed, X_train_oneHotEncoded), axis=1)
        X_val_preprocessed = np.concatenate((X_val_preprocessed, X_val_oneHotEncoded), axis=1)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_oneHotEncoded), axis=1)
        
        values_todrop = []
        try:
            drop_idx_isnull = not enc_onehot.drop_idx_
        except ValueError:
            drop_idx_isnull = False
        for idx_feature in range(len(features_oneHotEncode)):
            if drop_idx_isnull or (enc_onehot.drop_idx_[idx_feature]==None):
                values_todrop.append(None)
            else:
                values_todrop.append(enc_onehot.drop_idx_[idx_feature])
        
        features_oneHotEncoded = []
        for idx_feature, feature in enumerate(features_oneHotEncode):
            for idx_value, value in enumerate(enc_onehot.categories_[idx_feature]):
                if (values_todrop[idx_feature]==None) or (idx_value!=values_todrop[idx_feature]):
                    features_oneHotEncoded.append(feature + '_' + str(value))
        features_preprocessed += features_oneHotEncoded
        
    else:
        features_oneHotEncoded = []
        
    # OrdinalEncoding
    if features_ordinalEncode:
        X_train_ordinalEncode = X_train[features_ordinalEncode]
        X_val_ordinalEncode = X_val[features_ordinalEncode]
        X_test_ordinalEncode = X_test[features_ordinalEncode]
        
        X_train_ordinalEncoded = enc_ordinal.fit_transform(X_train_ordinalEncode)
        try:
            X_val_ordinalEncoded = enc_ordinal.transform(X_val_ordinalEncode)
            X_test_ordinalEncoded = enc_ordinal.transform(X_test_ordinalEncode)
        except ValueError as err:
            print(err)
            enc_ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_val_ordinalEncoded = enc_ordinal.transform(X_val_ordinalEncode)
            X_test_ordinalEncoded = enc_ordinal.transform(X_test_ordinalEncode)
            
        X_train_preprocessed = np.concatenate((X_train_preprocessed, X_train_ordinalEncoded), axis=1)
        X_val_preprocessed = np.concatenate((X_val_preprocessed, X_val_ordinalEncoded), axis=1)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_ordinalEncoded), axis=1)

        features_ordinalEncoded = [feature+'_ordenc' for feature in features_ordinalEncode]
        features_preprocessed += features_ordinalEncoded
        
    else:
        features_ordinalEncoded = []
    
    # Standardization
    if features_standardize:
        
        X_train_standardize = X_train[features_standardize]
        X_val_standardize = X_val[features_standardize]
        X_test_standardize = X_test[features_standardize]
        
        scaler = StandardScaler()
        X_train_standardized = scaler.fit_transform(X_train_standardize)
        X_val_standardized = scaler.transform(X_val_standardize)
        X_test_standardized = scaler.transform(X_test_standardize)
        
        X_train_preprocessed = np.concatenate((X_train_preprocessed, X_train_standardized), axis=1)
        X_val_preprocessed = np.concatenate((X_val_preprocessed, X_val_standardized), axis=1)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_standardized), axis=1)
        
        n_scaler = X_train_standardized.shape[1]
        features_standardized = [feature+'_std' for feature in features_standardize]
        features_preprocessed += features_standardized
        
    else:
        features_standardized = []
        
        
    # Untouched features
    if features_untouch:
        X_train_untouched = X_train[features_untouch]
        X_val_untouched = X_val[features_untouch]
        X_test_untouched = X_test[features_untouch]

        X_train_preprocessed = np.concatenate((X_train_preprocessed, X_train_untouched), axis=1)
        X_val_preprocessed = np.concatenate((X_val_preprocessed, X_val_untouched), axis=1)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_untouched), axis=1)
        
        features_preprocessed += features_untouch
        
    else:
        features_untouch = []
        
    # Target label encoding
    le = LabelEncoder()
    if encode_y:
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)
    
    # preprocessed dataset creation
    X_train_preprocessed = X_train_preprocessed.astype('float')
    X_val_preprocessed = X_val_preprocessed.astype('float')
    X_test_preprocessed = X_test_preprocessed.astype('float')
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    n_features = X_train_preprocessed.shape[1]
    
    return (X_train_preprocessed, y_train), (X_val_preprocessed, y_val), \
        (X_test_preprocessed, y_test), (n_features), (idxs_train, idxs_val, idxs_test), \
            (enc_onehot, enc_ordinal, scaler, le), (features_preprocessed, features_oneHotEncoded, features_ordinalEncoded, features_standardized, features_untouch)


def preprocess_test_dataset(dataset_test, features_all, features_oneHotEncode, features_standardize, enc, scaler):
    '''

    Parameters
    ----------
    dataset_test : pandas.DataFrame
        Dataset aggregato di test
    features_all : list
        lista di feature da considerare
    features_oneHotEncode : list
        lista di feature sui cui eseguire il one-hot-encoding
    features_standardize : list
        lista di feature sui cui eseguire la standardizzazione
    enc : sklearn.preprocessing.OneHotEncoder
        fitted encoder used to one-hot-encode features in features_oneHotEncode
    scaler : sklearn.preprocessing.StandardScaler
        fitted scaler used to scale features in features_standardize

    Returns
    -------
    X_test_preprocessed : numpy.array
        test data
    idxs_test : pandas.core.indexes.base.Index
        dataset indexes for train set

    '''
    X_test = dataset_test
    idxs_test = X_test.index
    X_test_preprocessed = np.zeros((X_test.shape[0], 0))
    features_untouch = [feature for feature in features_all if (feature not in features_oneHotEncode) and (feature not in features_standardize)]
    
    if features_oneHotEncode:
        X_test_oneHotEncode = X_test[features_oneHotEncode]
        X_test_oneHotEncoded = enc.transform(X_test_oneHotEncode)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_oneHotEncoded), axis=1)
        
    if features_standardize:
        X_test_standardize = X_test[features_standardize]
        X_test_standardized = scaler.transform(X_test_standardize)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_standardized), axis=1)
    
    if features_untouch:
        X_test_untouched = X_test[features_untouch]
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_untouched), axis=1)
        
    X_test_preprocessed = X_test_preprocessed.astype('float')
    
    return X_test_preprocessed, idxs_test
    
    
class Filter:
    '''
    A class used to filter your dataset.
    
    Examples
    --------
    >>> from sklearn import datasets
    >>> rom pandas import pd
    
    >>> iris = datasets.load_iris()
    >>> df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    >>> df['class_label'] = iris['target']
    
    >>> df_filtered = Filter.filter_by_minmax(df, field='sepal length (cm)', minmax_values=(5., 6.), verbose=True)
    
    Removed 83 out of 150 rows (55.33% removed)

    
    '''
    
    def verbose_filtering(func):
        def inner(dataset, verbose=False, **kwargs):
            old_len = len(dataset)
            dataset = func(dataset, **kwargs)
            new_len = len(dataset)
            if verbose:
                print(f'Removed {old_len - new_len} out of {old_len} rows ({((old_len - new_len) / old_len)*100:.2f}% removed)')
            return dataset
        return inner
    
    @verbose_filtering
    def filter_by_value(dataset, field, value):
        return dataset[dataset[field]==value]
    
    @verbose_filtering
    def filter_by_value_major_then(dataset, field, threshold):
        return dataset[dataset[field]>=threshold]

    @verbose_filtering
    def filter_by_value_minor_then(dataset, field, threshold):
        return dataset[dataset[field]<=threshold]
    
    @verbose_filtering
    def filter_by_outlier(dataset, field, quantile_coeff):
        series = dataset[field]
        lower_limit = series.quantile(quantile_coeff)
        upper_limit = series.quantile(1. - quantile_coeff)
        
        dataset_out = dataset[(series>=lower_limit) & (series <= upper_limit)]
        
        return dataset_out
    
    @verbose_filtering
    def filter_by_minmax(dataset, field, minmax_values):
        series = dataset[field]
        lower_limit = minmax_values[0]
        upper_limit = minmax_values[1]
        
        dataset_out = dataset[(series>=lower_limit) & (series <= upper_limit)]
        
        return dataset_out
        
    @verbose_filtering
    def filter_by_in_list(dataset, field, list_):
        return dataset[dataset[field].isin(list_)]

    
    





