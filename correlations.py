# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:08:22 2020

@author: joshu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/joshu/Documents/training_data.csv', index_col='PolicyNo', low_memory=False)

num_feat = df.select_dtypes(include = ['float64', 'int64'])

corr = num_feat.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(15, 12))

cmap = sns.color_palette("coolwarm", as_cmap=True)

sns.heatmap(corr[(corr >= 0.4) | (corr <= -0.4)], mask=mask,
            cmap=cmap, vmax=.3, center=0, annot=True, annot_kws={"size": 6},
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
