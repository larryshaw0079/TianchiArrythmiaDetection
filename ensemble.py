import os
import sys
import pdb

import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb

from tqdm import tqdm
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features


category_path = 'data/train/hf_round1_arrythmia.txt'
train_label_path = 'data/train/hf_round1_label.txt'
test_label_path = 'data/test/hf_round1_subA.txt'
NUM_CLASSES = 55

with open(category_path, 'r', encoding='utf8') as f:
    category = f.read().split('\n')[:-1]
category2int = {name:idx for idx, name in enumerate(category)}

with open(train_label_path, 'r', encoding='utf8') as f:
    contents = f.readlines()
    train_labels = {}
    for i, line in enumerate(contents):
        line = line.split('\t')
        train_labels[line[0]] = np.zeros(NUM_CLASSES)
        for label in line[3:-1]:
            train_labels[line[0]][category2int[label]] = 1

labels = np.array([np.array(list(train_labels.values())[idx]).astype(np.int) for idx in range(len(train_labels.keys()))])

all_extracted_features = pd.DataFrame()
for idx in tqdm(range(len(train_labels.keys()))):
    data = np.loadtxt('data/train/raw_data/%s'%(list(train_labels.keys())[idx]), delimiter=' ', skiprows=1)
    df = pd.DataFrame(data)
    df['id'] = 0
    df['time'] = range(len(df))

    extracted_features = extract_features(df, column_id="id", column_sort="time", 
                                          n_jobs=14, default_fc_parameters=EfficientFCParameters(), disable_progressbar=True)
    all_extracted_features = all_extracted_features.append(extracted_features)

all_extracted_features.to_csv('data/train/features.csv', index=False)

for i in labels.shape[1]:
    y = labels[:,i].reshape(-1)
    feature_filterd = select_features(all_extracted_features, y)
    relevant_features = relevant_features.union(set(feature_filterd.columns))

print(len(relevant_features))

extracted_features_filtered = all_extracted_features[list(relevant_features)]
extracted_features_filtered.to_csv('data/train/features_filtered.csv', index=False)
