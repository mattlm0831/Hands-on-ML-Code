# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:36:42 2019

@author: mlm14013work
"""

import os
import tarfile
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from six.moves import urllib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH  = os.path.join("datasets", "housing")
HOUSING_URL   = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
    
    
    
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def display_histogram(data):
    data.hist(bins = 50, figsize=(20,15))
    plt.savefig("histogram.png")
    plt.show()
    
    
def create_split(data):
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
        
    return strat_train_set, strat_test_set



def save_plot(data):
    data.plot(kind = "scatter", x="longitude", y = "latitude", alpha = .4, s=data["population"]/100, label = "population", figsize = (10, 7), c = "median_house_value", cmap=plt.get_cmap("jet"), colorbar = True)
    plt.legend()
    plt.savefig("heat_map_housing.png")