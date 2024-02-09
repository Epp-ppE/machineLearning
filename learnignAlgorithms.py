from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
# print(dftrain.head())
# print(dftrain.describe())

# Print the first row of the dataframe
# print(dftrain.loc[0], y_train.loc[0])

# Print the first n rows of the dataframe
# n = 2
# print(dftrain.head(n))

# Shape of the dataframe
dfShape = dftrain.shape
print("Row:", dfShape[0])
print("Column:", dfShape[1])
print(dfShape)

# print(dftrain["age"].hist(bins=20))

# dftrain.sex.value_counts().plot(kind='barh')
# Alt: dftrain.age.hist(bins=20)

# survived ratio of male and female
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

# plt.show()

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)