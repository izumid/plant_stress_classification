import os
from datetime import datetime
import os.path
from pathlib import Path
import zipfile


import numpy as np
#import pandas as pd


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier


def code_debug(message,var=None,debug=False):
	if not debug is False:
		if not var is None: print(f"[Debug] {message} {var}.")
		else: print(f"[Debug]: {message}.")


def logTime(start,model,window,path_save):
	end = datetime.now()
	message = (f"{model}'s with [{window}] duration (hhmmssms): {str((end-start))}. (Registred log: {datetime.now()})\n")
	if not os.path.exists(path_save): os.makedirs(path_save)

	with open(os.path.join(path_save,'log_times.txt'), 'a') as log_file: log_file.write(message)


def debugTxt(destination_path,file_name,result,mode):
	root = os.getcwd()
	destination_path = os.path.join(root,destination_path)
	if not os.path.exists(destination_path): os.makedirs(destination_path)
	
	file = open(os.path.join(destination_path,file_name+".txt"),mode)
	for item in result: file.write(item+"\n")
	file.close()


def dinamic_predict(X_train,y_train,X_test,model,verbose=False,random_state=42):
	int_verbose = int(verbose)
	match model:
		case "DUM": classifier = DummyClassifier(random_state=random_state,strategy="stratified")
		case "DT": classifier = DecisionTreeClassifier(random_state=random_state,criterion="entropy",class_weight="balanced")
		case "NB": classifier = GaussianNB(priors=None, var_smoothing=1e-09)
		case "KNN": classifier = KNeighborsClassifier(n_neighbors=5,weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
		case "XGB": classifier = xgb.XGBClassifier(random_state=random_state,verbosity=int_verbose,objective="binary:logistic")
		case "RF": classifier = RandomForestClassifier(random_state=random_state,verbose=int_verbose,criterion="entropy",class_weight="balanced")
		case "MLP": classifier = MLPClassifier(random_state=random_state,verbose=verbose,solver="adam",activation="logistic",max_iter=1000,hidden_layer_sizes=(100, 100))
		case "SVM": classifier = svm.SVC(random_state=random_state,verbose=verbose,probability=False,C=1.0, kernel='rbf',degree=3,gamma='scale',coef0=0.0,shrinking=True,tol=0.001,cache_size=200,class_weight=None,max_iter=-1,decision_function_shape='ovr', break_ties=False)

	classifier.fit(X_train,y_train)
	y_train_pred = classifier.predict(X_train)
	y_pred = classifier.predict(X_test)

	return y_train_pred,y_pred

# MARK: dinamic_training
def dinamic_training(X_before,y_before,X_after,y_after,model,base_name,window_name,k_fold_split,verbose,random_state):
	result = []
	
	if k_fold_split > 0:
		kfold = KFold(shuffle=True,n_splits=k_fold_split,random_state=random_state)

		for train_index, test_index in kfold.split(X_before):
			X_train = np.vstack((X_before[train_index],X_after[train_index]))
			y_train = np.append(y_before[train_index],y_after[train_index])
			
			X_test = np.vstack((X_before[test_index],X_after[test_index]))
			y_test = np.append(y_before[test_index],y_after[test_index])

			# Seems that xgb model waits a differente y shape, understand better:
			# https://discuss.xgboost.ai/t/invalid-classes-inferred-from-unique-values-of-y-expected-0-1-2-1387-1388-1389-got-0-1-2-18609-24127-41850/2806
			if model == "XGB": y_train = (LabelEncoder()).fit_transform(y_train)	
			
			#print("tain->",X_train.shape,y_train.shape)
			#print("test->",X_test.shape,y_test.shape)

			if 1==1:
				y_train_predicted,y_predicted = dinamic_predict(X_train,y_train,X_test,model,verbose,random_state)

				result.append([
					base_name
					,model
					,window_name
					,str(accuracy_score(y_train, y_train_predicted))
					,str(accuracy_score(y_test, y_predicted))
					,str(precision_score(y_test,y_predicted))
					,str(recall_score(y_test,y_predicted))
					,str(f1_score(y_test,y_predicted))
					#,str(y_test)
					#,str(y_predicted)
				])
	else:
		#X = np.vstack((X_before,X_after))
		#y = np.append(y_before,y_after)
		X = np.concatenate((X_before,X_after))
		y = np.append(y_before,y_after)
		
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.94, shuffle=True, random_state=42)
		y_train_predicted,y_predicted = dinamic_predict(X_train,y_train,X_test,model,verbose,random_state)

		result.append([
			base_name
			,model
			,window_name
			,str(accuracy_score(y_train, y_train_predicted))
			,str(accuracy_score(y_test, y_predicted))
			,str(precision_score(y_test,y_predicted))
			,str(recall_score(y_test,y_predicted))
			,str(f1_score(y_test,y_predicted))
			#,str(y_test)
			#,str(y_predicted)
		])

	return(result)

# MARK: run_train_test
def run_train_test(path_base,path_destination,list_model,file_name_predictor,k_fold_split,verbose,random_state,test,debug):
	file_name_predictor = file_name_predictor + ".npy"
	file_name_target = file_name_predictor.replace('X','y')
	list_stimuli = [folder for folder in os.listdir(path_base)]
	if test == True: list_stimuli = list_stimuli[:2]
	end = int(len(list_stimuli))-1
	#result = np.empty((0,), dtype=object)
	#result = ["Stimulus","Model","Window","TrainAccuracy","Accuracy","Precision","Recall","F1Score"]
	result = []

	if k_fold_split > 0: sub_folder = "cross_validation"
	else: sub_folder = "train_test_split"
	
	path_destination = os.path.join(path_destination,sub_folder)
	if not os.path.exists(path_destination): os.makedirs(path_destination)

	for model in list_model:
		# Scanning half of folders
		code_debug(message=f"""The classifier "{model}" is using the stimuli {list_stimuli}. Traning mode: {sub_folder}""",debug=debug)

		for i in range(0,end,2):
			# Becauase we're getting just first word of stimuli e.g "Cold" After becomes "Cold"
			stimulus = list_stimuli[i][:list_stimuli[i].find(" ")]
			code_debug(message="Executed stimulus:",var=stimulus,debug=debug)

			for folder_window in os.listdir(os.path.join(path_base,list_stimuli[i])):
				after_folder = os.path.join(path_base,list_stimuli[i],folder_window)
				before_folder = os.path.join(path_base,list_stimuli[i+1],folder_window)
				print(f"{model}'s start stimuli: {list_stimuli[i]} & {list_stimuli[i+1]} ({folder_window})")
				
				code_debug(message="Processing:",var=(after_folder,before_folder),debug=debug)
				
				X_before = np.load(os.path.join(before_folder,file_name_predictor))
				y_before = np.load(os.path.join(before_folder,file_name_target))					
				X_after = np.load(os.path.join(after_folder,file_name_predictor))
				y_after = np.load(os.path.join(after_folder,file_name_target))
				#X = np.concatenate((X1,X2))
				#y = np.append(y1,y2)
		
				"""
				code_debug(message="X info: ",var=X_before.shape,debug=debug)
				code_debug(message="X info: ",var=X_after.shape,debug=debug)

				code_debug(message="y info: ",var=y_before.shape,debug=debug)
				code_debug(message="y info: ",var=y_after.shape,debug=debug)
				"""

				start = datetime.now()

				dc = dinamic_training(
							X_before,y_before,X_after,y_after
							,model=model,base_name=stimulus,window_name=folder_window
							,k_fold_split=k_fold_split
							,verbose=verbose,random_state=random_state
						)
				#result = np.vstack( (result, dc))
				#result = np.append(result, dc, axis=0)

				for array_sub in dc:
					result.append(array_sub)

				#print("result: ", result)
				
				#logTime(start,model,folder_window,path_save=os.path.join(Path(path_base).parent,"duration"))
				logTime(start,model,folder_window,path_save=path_destination)
				#if test == True: break
				#save_csv(result,model,path_destination,"result",start,backup=True,log=True)


	np.save(os.path.join(path_destination,"result.npy"), result)