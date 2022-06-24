#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:45:27 2022

@author: luigi
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import datetime


def merge_categorical_values(df, feature, min_count):
    def merge_fun(df_row, feature, values_to_merge):
        if df_row[feature] in values_to_merge:
            return "Other"
        else:
            return df_row[feature]
        
    values_to_merge = df[feature].value_counts()[df[feature].value_counts()<min_count].index.to_list()
    
    def apply_fun(df_row):
        return merge_fun(df_row, feature, values_to_merge)
            
    df[feature+"_merged"] = df.apply(apply_fun, axis=1)
    
    return df[feature+"_merged"]



def preprocess_dataset(dataset, features_all, features_oneHotEncode, features_standardize, 
                       target_label, drop_oneHotEncoder='first',
                       drop_train_duplicates=False, split_by_date=False, train_size=0.6, val_size=0.2, 
                       test_size=0.2, random_seed=42):   
    '''

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset aggregato
    features_all : list
        lista di feature da considerare, e.g. features_all = ['COMMODITY', 'Anno_Nascita', 'popresidente', 'tipocomune', 'firma_day']
    features_oneHotEncode : list
        lista di feature sui cui eseguire il one-hot-encoding, e.g. features_oneHotEncode = ['COMMODITY', 'tipocomune']
    features_standardize : list
        lista di feature sui cui eseguire la standardizzazione, e.g. features_standardize = ['Anno_Nascita', 'firma_day']
    drop_oneHotEncoder : string
        drop type of one hot encoder: {'first', 'if_binary', None}
    target_label : string
        target column name
    drop_train_duplicates : bool
        whenever to drop train duplicates
    split_by_date : bool
        perform the train/val/split based on data_firma_plico_a
    train_size: float
        train_size (only for split_by_date=False), e.g. 0.6
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
    '''
    
    def drop_train_duplicates_func(X_train, y_train):
        df_train = pd.concat([X_train, y_train], axis=1)
        df_train = df_train.drop_duplicates()
        X_train = df_train.drop(target_label, axis=1)
        y_train = df_train[target_label]
        return X_train, y_train
    
    features_untouch = [feature for feature in features_all if (feature not in features_oneHotEncode) and (feature not in features_standardize)]

    if split_by_date:
        dataset['data_firma_plico_a'] = dataset['data_firma_plico_a'].astype('datetime64[ns]')
        max_year = dataset['data_firma_plico_a'].max().year
        max_month = dataset['data_firma_plico_a'].max().month
        validation_date = datetime.datetime(max_year, max_month-1, 1)
        test_date = datetime.datetime(max_year, max_month, 1)
        dataset_train = dataset[dataset['data_firma_plico_a']<validation_date]
        dataset_val = dataset[(dataset['data_firma_plico_a']>=validation_date) & (dataset['data_firma_plico_a']<test_date)]
        dataset_test = dataset[dataset['data_firma_plico_a']>=test_date]
        X_train, X_val, X_test = dataset_train[features_all], dataset_val[features_all], dataset_test[features_all]
        y_train, y_val, y_test = dataset_train[target_label], dataset_val[target_label], dataset_test[target_label]
    else:
        X = dataset[features_all]
        y = dataset[target_label]
        X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=val_size+test_size, random_state=random_seed, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size/(val_size+test_size), random_state=random_seed, stratify=y_val_test)

    if drop_train_duplicates:   X_train, y_train = drop_train_duplicates_func(X_train, y_train)

    idxs_train = X_train.index
    idxs_val = X_val.index
    idxs_test = X_test.index
    
    X_train_preprocessed = np.zeros((X_train.shape[0], 0))
    X_val_preprocessed = np.zeros((X_val.shape[0], 0))
    X_test_preprocessed = np.zeros((X_test.shape[0], 0))
    
    features_preprocessed = []
    scaler = StandardScaler()
    enc = OneHotEncoder(sparse=False, drop=drop_oneHotEncoder)
    
    # OneHotEncoding
    if features_oneHotEncode:
        
        X_train_oneHotEncode = X_train[features_oneHotEncode]
        X_val_oneHotEncode = X_val[features_oneHotEncode]
        X_test_oneHotEncode = X_test[features_oneHotEncode]
        
        X_train_oneHotEncoded = enc.fit_transform(X_train_oneHotEncode)
        X_val_oneHotEncoded = enc.transform(X_val_oneHotEncode)
        X_test_oneHotEncoded = enc.transform(X_test_oneHotEncode)
        
        X_train_preprocessed = np.concatenate((X_train_preprocessed, X_train_oneHotEncoded), axis=1)
        X_val_preprocessed = np.concatenate((X_val_preprocessed, X_val_oneHotEncoded), axis=1)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_oneHotEncoded), axis=1)
        
        features_oneHotEncoded = []
        for idx, feature in enumerate(features_oneHotEncode):
            for value in enc.categories_[idx][1:]:
                features_oneHotEncoded.append(feature + '_' + str(value))
        features_preprocessed += features_oneHotEncoded
        
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
        
    # Untouched features
    if features_untouch:
        
        X_train_untouched = X_train[features_untouch]
        X_val_untouched = X_val[features_untouch]
        X_test_untouched = X_test[features_untouch]
        
        X_train_preprocessed = np.concatenate((X_train_preprocessed, X_train_untouched), axis=1)
        X_val_preprocessed = np.concatenate((X_val_preprocessed, X_val_untouched), axis=1)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_untouched), axis=1)
        
        features_preprocessed += features_untouch
        
    # Target label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    
    # preprocessed dataset creation
    X_train_preprocessed = X_train_preprocessed.astype('float')
    X_val_preprocessed = X_val_preprocessed.astype('float')
    X_test_preprocessed = X_test_preprocessed.astype('float')
    
    n_features = X_train_preprocessed.shape[1]
    n_output = len(np.unique(y_train_encoded))
    
    return (X_train_preprocessed, y_train_encoded), (X_val_preprocessed, y_val_encoded), \
        (X_test_preprocessed, y_test_encoded), (n_features, n_output), (idxs_train, idxs_val, idxs_test), (enc, scaler, le), features_preprocessed


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
    
    
    
    


def preprocess_dataset_correct_sparsematrix(dataset, features_all, features_oneHotEncode, features_standardize, target_label, test_size=0.2, random_seed=42):   

    features_untouch = [feature for feature in features_all if (feature not in features_oneHotEncode) and (feature not in features_standardize)]


    X = dataset[features_all]
    y = dataset[target_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y)


    idxs_train = X_train.index
    idxs_test = X_test.index
    
    X_train_preprocessed = np.zeros((X_train.shape[0], 0))
    X_test_preprocessed = np.zeros((X_test.shape[0], 0))
    
    features_preprocessed = []
    scaler = StandardScaler()
    enc = OneHotEncoder()
    
    # OneHotEncoding
    if features_oneHotEncode:
        
        X_train_oneHotEncode = X_train[features_oneHotEncode]
        X_test_oneHotEncode = X_test[features_oneHotEncode]
        
        X_train_oneHotEncoded = enc.fit_transform(X_train_oneHotEncode)
        X_test_oneHotEncoded = enc.transform(X_test_oneHotEncode)
        
        X_train_preprocessed = np.concatenate((X_train_preprocessed, X_train_oneHotEncoded.todense()), axis=1)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_oneHotEncoded.todense()), axis=1)
        
        features_oneHotEncoded = []
        for idx, feature in enumerate(features_oneHotEncode):
            for value in enc.categories_[idx][1:]:
                features_oneHotEncoded.append(feature + "_" + str(value))
        features_preprocessed += features_oneHotEncoded
        
    # Standardization
    if features_standardize:
        
        X_train_standardize = X_train[features_standardize]
        X_test_standardize = X_test[features_standardize]
        
        scaler = StandardScaler()
        X_train_standardized = scaler.fit_transform(X_train_standardize)
        X_test_standardized = scaler.transform(X_test_standardize)
        
        X_train_preprocessed = np.concatenate((X_train_preprocessed, X_train_standardized), axis=1)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_standardized), axis=1)
        
        n_scaler = X_train_standardized.shape[1]
        features_standardized = [feature+"_std" for feature in features_standardize]
        features_preprocessed += features_standardized
        
    # Untouched features
    if features_untouch:
        
        X_train_untouched = X_train[features_untouch]
        X_test_untouched = X_test[features_untouch]
        
        X_train_preprocessed = np.concatenate((X_train_preprocessed, X_train_untouched), axis=1)
        X_test_preprocessed = np.concatenate((X_test_preprocessed, X_test_untouched), axis=1)
        
        features_preprocessed += features_untouch
        
    # Target label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # preprocessed dataset creation
    X_train_preprocessed = X_train_preprocessed.astype("float")
    X_test_preprocessed = X_test_preprocessed.astype("float")
    
    n_features = X_train_preprocessed.shape[1]
    n_output = len(np.unique(y_train_encoded))
    
    return (X_train_preprocessed, y_train_encoded), (X_test_preprocessed, y_test_encoded), (n_features, n_output), features_preprocessed