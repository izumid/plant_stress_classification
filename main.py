import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

from preprocessing import window_size

def main():
	print("Experiment 1 use train-test-split, from 2 to 4 cross validation was used...\n")
	dict = {
		1: "Large duplicate values: train test split with small number of windows."
		,2: "Unique values unbalanced: select unique values of each stimuli and class. For example, cold has differente size sample of unique values between stresses and non stressed."
		,3: "Unique values balanced: set the same sample size to all stimuli and class (estressed x non stressed)."
		,4: "Large duplicate values: huge number of different windows."
	}
	for key in dict:
		print(f"{key}: {dict[key]}")
	
	experiment = int(input("Type number of desired experiment: "))

	match experiment:
		case 1:
			exec(open("01_large_duplicate_values_0.0.1.py").read())
		case 2:
			exec(open("02_unique_values_unbalanced_0.0.1").read())
		case 3:
			exec(open("03_unique_values_balanced_0.0.0.py").read())
		case 4:
			exec(open("04_large_duplicate_values_0.0.0.py").read())
		case _: 
			"chose a valid option"
	


if __name__ == "__main__":
	main()