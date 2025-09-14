import os
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import md_util as ut


##########################################################
## 						CLASSIFY						##
##########################################################

# MARK: Classify
def classify(path_origin,path_destination,list_classifier,random_state,verbose,k_fold_split,skip_file_exists,execution_sort_ascending,window=None,show_debug_message=False):
	"""
		Description:

		Arguments:
		
	"""
	
	try:
		int_verbose = int(verbose)
		scaler = MinMaxScaler()
		if not os.path.exists(path_destination): os.makedirs(path_destination)
		rounds = 1
		file = []

		if k_fold_split > 0: 
			skf = StratifiedKFold(n_splits=k_fold_split,shuffle=True,random_state=random_state)
			total_rounds = len(file)*k_fold_split*len(list_classifier)
			file = os.listdir(path_origin)
		else:  
			window_file = os.listdir(path_origin)
			first_key_value = next(iter(window.values()))
			total_rounds = len(first_key_value)*len(list_classifier)
			
			for basefile in window_file:
				window_size, window_sample_size = Path(basefile).stem.split("x")
				window_aux = [int(window_size), int(window_sample_size)]
				if window_aux in first_key_value: file.append(basefile)

		if skip_file_exists: 
			for existing_file in os.listdir(path_destination):
				path_file_absolute =  os.path.join(path_destination,existing_file)
				if existing_file in file and os.path.isfile(path_file_absolute): file.remove(existing_file)

		dict_column_type = {
			"model": "str"
			,"window": "int"
			,"samples_summarized": "int"
			,"accuracy_train": "float"
			,"accuracy_test": "float"
			,"presicion": "float"
			,"recall": "float"
			,"f1_score": "float"
		}
		
		if not execution_sort_ascending is None:
			file_window = [(int(Path(f).stem.split('x')[0]),f) for f in file]
			if execution_sort_ascending: sort = sorted(file_window)
			else: sort = sorted(file_window, key=lambda x: x[0], reverse=True)
			file = [element[1] for element in sort]
		
		train_test_execution = 1
		for window_filename in file:
					
			window_name = Path(window_filename).stem
			window_size_sample = window_name.split('x')
			window_size = int(window_size_sample[0])
			window_sample_size = int(window_size_sample[1])

			df_train = pd.read_feather(os.path.join(path_origin,window_filename))
			print(df_train.head())
			X_label =  df_train.iloc[:, 0] #df_train["stimulus_stage"]
			X = df_train.iloc[:, 1:-1]
			y = df_train.iloc[:, -1]
			result = []

			for model in list_classifier:
				
				if not (model != "DT" and model != "XGB" and model != "RF"): 
					min_samples_leaf = int((window_sample_size * 0.8) *0.1)
					min_samples_split = int(min_samples_leaf*0.1)
					max_depth = 3
					if min_samples_leaf <= 1: min_samples_leaf = 2
					if min_samples_split <= 1: min_samples_split = 2

				match model:
					case "DUM": classifier = DummyClassifier(random_state=random_state,strategy="stratified")
					case "DT": classifier = DecisionTreeClassifier(random_state=random_state,criterion="gini",min_samples_split=min_samples_split,max_depth=max_depth,min_samples_leaf=min_samples_split)
					case "NB": classifier = GaussianNB(priors=None, var_smoothing=1e-09)
					case "KNN": classifier = KNeighborsClassifier(n_neighbors=5,weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
					case "XGB": classifier = xgb.XGBClassifier(random_state=random_state,verbosity=int_verbose,objective="binary:logistic",min_child_weight=min_samples_leaf,max_depth=max_depth,eta=0.1,gamma=5)
					case "RF": classifier = RandomForestClassifier(random_state=random_state,verbose=int_verbose,criterion="gini",min_samples_split=min_samples_split,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=1000)
					case "MLP": classifier = MLPClassifier(random_state=random_state,verbose=verbose,solver="adam",activation="logistic",max_iter=1000,hidden_layer_sizes=(1,2))
					case "SVM": classifier = svm.SVC(random_state=random_state,verbose=verbose,probability=False,C=1.0, kernel='rbf',degree=3,gamma='scale',coef0=0.0,shrinking=True,tol=0.001,cache_size=200,class_weight=None,max_iter=-1,decision_function_shape='ovr', break_ties=False)

				if k_fold_split:
					execution = 1
					for train_index, test_index in skf.split(X, X_label):
						ut.debug(message=f"Model: {model}({window_filename}). Cross validation fold[{execution}] ({(rounds/total_rounds)*100:.2f}%)",show=show_debug_message)

						X_train, X_test = X.iloc[train_index], X.iloc[test_index]
						y_train, y_test = y.iloc[train_index], y.iloc[test_index]

						X_train_scaled = scaler.fit_transform(X_train)
						X_test_scaled = scaler.transform(X_test)

						classifier.fit(X_train_scaled,y_train)
						y_train_predicted = classifier.predict(X_train_scaled)
						y_predicted = classifier.predict(X_test_scaled)

						data = []
						data = [
							model
							,window_size
							,window_sample_size
							,str(accuracy_score(y_train, y_train_predicted)*100)
							,str(accuracy_score(y_test, y_predicted)*100)
							,str(precision_score(y_test,y_predicted)*100)
							,str(recall_score(y_test,y_predicted)*100)
							,str(f1_score(y_test,y_predicted)*100)
						]

						result.append(data)

						rounds+=1
						execution+=1
				else:
					
					X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=random_state)
				
					ut.debug(message=f"Model: {model}({window_filename}). Train Test execution[{train_test_execution}] ({(rounds/total_rounds)*100:.2f}%)", show=show_debug_message)

					X_train_scaled = scaler.fit_transform(X_train)
					X_test_scaled = scaler.transform(X_test)

					classifier.fit(X_train_scaled,y_train)
					y_train_predicted = classifier.predict(X_train_scaled)
					y_predicted = classifier.predict(X_test_scaled)
					
					data = []
					data = [
						model
						,window_size
						,window_sample_size
						,str(accuracy_score(y_train, y_train_predicted)*100)
						,str(accuracy_score(y_test, y_predicted)*100)
						,str(precision_score(y_test,y_predicted)*100)
						,str(recall_score(y_test,y_predicted)*100)
						,str(f1_score(y_test,y_predicted)*100)
					]

					result.append(data)
					rounds+=1
					train_test_execution+=1

			df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dict_column_type.items()})
			df = pd.DataFrame(result,columns=df.columns.tolist())
			df.to_feather(os.path.join(path_destination,window_filename))
	except Exception as error:
		ut.log_file(filename="log_file",header_message="classify: plant stress")