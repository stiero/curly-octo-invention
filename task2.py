#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:35:56 2020

@author: tauro
"""


from sklearn.datasets import load_boston
import numpy as np

boston = load_boston()

X = pd.DataFrame(boston.data, columns = boston.feature_names)
y = pd.DataFrame(boston.target, columns = ["target"])


from sklearn.datasets import make_regression

X, y = make_regression(n_samples=10000, n_features=5, n_informative=5)

cat_column = np.random.choice(range(0,4000), 10000)

X = pd.concat(X, pd.DataFrame(cat_column, columns = ['cat']))


