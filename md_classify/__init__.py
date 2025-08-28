import os
import pandas as pd

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
import md_logfile as lf


# MARK: Classify
def classify(path_base,list_window,path_destination,list_classifier,random_state,verbose,k_fold_split,filename="result"):
	"""
		Description:

		Arguments:
		
	"""
	try:
		int_verbose = int(verbose)
		scaler = MinMaxScaler()
		if not os.path.exists(path_destination): os.makedirs(path_destination)
		
		rounds = 0
		header_txt = ["model","window","samples_summarized","accuracy_train","accuracy_test","presicion","recall","f1_score"]
		
		if k_fold_split: 
			skf = StratifiedKFold(n_splits=k_fold_split,shuffle=True,random_state=random_state)
			total_rounds = len(list_window)*k_fold_split*len(list_classifier)
		else:  total_rounds = len(list_window)*len(list_classifier)

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
		
		
		for MxN in list_window:
			window = str(MxN[0])+'x'+str(MxN[1])
			path_txt = os.path.join(path_destination,f"{filename}_skfold_{MxN[0]}x{MxN[1]}.txt")
			if os.path.exists(path_txt): os.remove(path_txt)
			#with open(path_txt, mode="a") as file: file.write(";".join(map(str, header_txt)) + "\n")
			#df_train = pd.read_feather(os.path.join(path_base,folder_window,"dataset.feather"))
			df_train = pd.read_feather(os.path.join(path_base,f"{window}.feather"))
			print(df_train.head())
			X_label =  df_train.iloc[:, 0] #df_train["stimulus_stage"]
			X = df_train.iloc[:, 1:-1]
			y = df_train.iloc[:, -1]
			result = []


			for model in list_classifier:
				if not (model != "DT" and model != "XGB" and model != "RF"): 
					min_samples_leaf = int((MxN[1] * 0.8) *0.1)
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
						print(f"Model: {model}({window}). Cross validation fold[{execution}] ({(rounds/total_rounds)*100:.2f}%)")

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
							,MxN[0]
							,MxN[1]
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
					execution = 1
					X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=random_state)
				
					print(f"Model: {model}({window}). Train Test execution[{execution}] ({(rounds/total_rounds)*100:.2f}%)")

					X_train_scaled = scaler.fit_transform(X_train)
					X_test_scaled = scaler.transform(X_test)

					classifier.fit(X_train_scaled,y_train)
					y_train_predicted = classifier.predict(X_train_scaled)
					y_predicted = classifier.predict(X_test_scaled)
					
					data = []
					data = [
						model
						,MxN[0]
						,MxN[1]
						,str(accuracy_score(y_train, y_train_predicted)*100)
						,str(accuracy_score(y_test, y_predicted)*100)
						,str(precision_score(y_test,y_predicted)*100)
						,str(recall_score(y_test,y_predicted)*100)
						,str(f1_score(y_test,y_predicted)*100)
					]

					result.append(data)
					rounds+=1
					execution+=1


			df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dict_column_type.items()})
			df = pd.DataFrame(result,columns=df.columns.tolist())
			df.to_feather(os.path.join(path_destination,f"{filename}.feather"))
	except Exception as error:
		lf.log_file(filename="log_file",header_message="classify: plant stress")