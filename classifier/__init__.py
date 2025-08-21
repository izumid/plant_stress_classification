import os
from datetime import datetime
import os.path
from pathlib import Path
import zipfile

import numpy as np
#import pandas as pd

from sklearn.model_selection import StratifiedKFold
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


# MARK:Excute fit and test data
def run_train_test(path_base,path_destination,list_model,file_name_predictor,k_fold_split,verbose,random_state,test,debug):
	file_name_predictor = file_name_predictor + ".npy"
	file_name_target = file_name_predictor.replace('X','y')

	list_stimuli = [folder for folder in os.listdir(path_base)]
	if test == True: list_stimuli = list_stimuli[:2]
	end = int(len(list_stimuli))-1
	int_verbose = int(verbose)
	result = []

	if k_fold_split > 0: sub_folder = "cross_validation"
	else: sub_folder = "train_test_split"
	
	path_destination = os.path.join(path_destination,sub_folder)
	if not os.path.exists(path_destination): os.makedirs(path_destination)

	for model in list_model:
		# Scanning half of folders
		code_debug(message=f"""The classifier "{model}" is using the stimuli {list_stimuli}. Traning mode: {sub_folder}""",debug=debug)
		if not (model != "DT" and model != "XGB" and model != "RF"): 
			min_samples_leaf = int((int(folder_window[:folder_window.find('x')]) * 0.8) *0.1)
			min_samples_split = int(min_samples_leaf*0.1)
			max_depth = 3
			if min_samples_leaf <= 1: min_samples_leaf = 2
			if min_samples_split <= 1: min_samples_split = 2

		match model:
			# -- old --
			#case "DT": classifier = DecisionTreeClassifier(random_state=random_state,criterion="entropy",class_weight="balanced")
			#case "RF": classifier = RandomForestClassifier(random_state=random_state,verbose=int_verbose,criterion="entropy",class_weight="balanced")
			#case "MLP": classifier = MLPClassifier(random_state=random_state,verbose=verbose,solver="adam",activation="logistic",max_iter=1000,hidden_layer_sizes=(100, 100))
			case "DUM": classifier = DummyClassifier(random_state=random_state,strategy="stratified")
			case "DT": classifier = DecisionTreeClassifier(random_state=random_state,criterion="gini",min_samples_split=min_samples_split,max_depth=max_depth,min_samples_leaf=min_samples_split)
			case "NB": classifier = GaussianNB(priors=None, var_smoothing=1e-09)
			case "KNN": classifier = KNeighborsClassifier(n_neighbors=5,weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
			case "XGB": classifier = xgb.XGBClassifier(random_state=random_state,verbosity=int_verbose,objective="binary:logistic",min_child_weight=min_samples_leaf,max_depth=max_depth,eta=0.1,gamma=5)
			case "RF": classifier = RandomForestClassifier(random_state=random_state,verbose=int_verbose,criterion="gini",min_samples_split=min_samples_split,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=1000)
			case "MLP": classifier = MLPClassifier(random_state=random_state,verbose=verbose,solver="adam",activation="logistic",max_iter=1000,hidden_layer_sizes=(1,2))
			case "SVM": classifier = svm.SVC(random_state=random_state,verbose=verbose,probability=False,C=1.0, kernel='rbf',degree=3,gamma='scale',coef0=0.0,shrinking=True,tol=0.001,cache_size=200,class_weight=None,max_iter=-1,decision_function_shape='ovr', break_ties=False)


		for i in range(0,end,2):
			# Becauase we're getting just first word of stimuli e.g "Cold" After becomes "Cold"
			stimulus = list_stimuli[i][:list_stimuli[i].find(" ")]
			code_debug(message="Executed stimulus:",var=stimulus,debug=debug)

			for folder_window in os.listdir(os.path.join(path_base,list_stimuli[i])):
				after_folder = os.path.join(path_base,list_stimuli[i],folder_window)
				before_folder = os.path.join(path_base,list_stimuli[i+1],folder_window)
				print(f"{model}'s start stimuli: {list_stimuli[i]} & {list_stimuli[i+1]} ({folder_window})")
				
				code_debug(message="Processing:",var=(after_folder,before_folder),debug=debug)

				#X = np.concatenate((X1,X2))
				#y = np.append(y1,y2)
		
				start = datetime.now()

				if k_fold_split > 0:
					#kfold = KFold(shuffle=True,n_splits=k_fold_split,random_state=random_state)
					skf = StratifiedKFold(n_splits=k_fold_split,shuffle=True,random_state=random_state)

					X_before = np.load(os.path.join(before_folder,file_name_predictor))
					y_before = np.load(os.path.join(before_folder,file_name_target))					
					X_after = np.load(os.path.join(after_folder,file_name_predictor))
					y_after = np.load(os.path.join(after_folder,file_name_target))

					X = np.vstack((X_before,X_after))
					y = np.appen(y_before,y_after)

					#for train_index, test_index in kfold.split(X_before):
					for train_index, test_index in skf.split(X, y):
						X_train, X_test = X.iloc[train_index], X.iloc[test_index]
						y_train, y_test = y.iloc[train_index], y.iloc[test_index]

						#X_train = np.vstack((X_before[train_index],X_after[train_index]))
						#y_train = np.append(y_before[train_index],y_after[train_index])
						
						#X_test = np.vstack((X_before[test_index],X_after[test_index]))
						#y_test = np.append(y_before[test_index],y_after[test_index])

						if model == "XGB": y_train = (LabelEncoder()).fit_transform(y_train)	
							
						classifier.fit(X_train,y_train)
						y_train_predicted = classifier.predict(X_train)
						y_predicted = classifier.predict(X_test)

						result.append([
							stimulus
							,model
							,folder_window
							,str(accuracy_score(y_train, y_train_predicted)*100)
							,str(accuracy_score(y_test, y_predicted)*100)
							,str(precision_score(y_test,y_predicted)*100)
							,str(recall_score(y_test,y_predicted)*100)
							,str(f1_score(y_test,y_predicted)*100)
							#,str(y_test)
							#,str(y_predicted)
						])

				else:
					X_before_train = np.load(os.path.join(before_folder,"X_train.npy"))
					y_before_train = np.load(os.path.join(before_folder,"y_train.npy"))					
					X_after_train = np.load(os.path.join(after_folder,"X_train.npy"))
					y_after_train = np.load(os.path.join(after_folder,"y_train.npy"))

					X_before_test = np.load(os.path.join(before_folder,"X_train.npy"))
					y_before_test = np.load(os.path.join(before_folder,"y_train.npy"))					
					X_after_test = np.load(os.path.join(after_folder,"X_train.npy"))
					y_after_test = np.load(os.path.join(after_folder,"y_train.npy"))

					X_train = np.concatenate((X_before_train,X_after_train))
					X_test = np.concatenate((X_before_test,X_after_test))
					y_train = np.append(y_before_train,y_after_train)
					y_test = np.append(y_before_test,y_after_test)

					classifier.fit(X_train,y_train)
					y_train_predicted = classifier.predict(X_train)
					y_predicted = classifier.predict(X_test)

					result.append([
						stimulus
						,model
						,folder_window
						,str(accuracy_score(y_train, y_train_predicted)*100)
						,str(accuracy_score(y_test, y_predicted)*100)
						,str(precision_score(y_test,y_predicted)*100)
						,str(recall_score(y_test,y_predicted)*100)
						,str(f1_score(y_test,y_predicted)*100)
						#,str(y_test)
						#,str(y_predicted)
					])

				logTime(start,model,folder_window,path_save=path_destination)
				#if test == True: break
				#save_csv(result,model,path_destination,"result",start,backup=True,log=True)

	np.save(os.path.join(path_destination,"result.npy"), result)
	#np.savetxt(os.path.join(path_destination,"result.csv"), result)

# MARK: Verify results
def check_result(path_destination,read_filename,csv=False,debug=False):
	st = os.path.join(path_destination,"00_standard")
	mn = os.path.join(path_destination,"01_min_max")
	zs = os.path.join(path_destination,"02_z_score")

	if not os.path.exists(st): os.makedirs(st)
	if not os.path.exists(mn): os.makedirs(mn)
	if not os.path.exists(zs): os.makedirs(zs)
	aux = r"05_result\train_test_split"

	stardard= np.load(os.path.join(st,aux,read_filename))
	min_max = np.load(os.path.join(mn,aux,read_filename))
	z_score = np.load(os.path.join(zs,aux,read_filename))

	if debug == True:
		print(stardard,"\r\n")
		print(min_max,"\r\n")
		print(z_score,"\r\n")
	
	if csv == True:
		np.savetxt(os.path.join(st,"st_output.csv"), stardard, delimiter=",", fmt="%s", encoding="utf-8")
		np.savetxt(os.path.join(mn,"mn_output.csv"), min_max, delimiter=",", fmt="%s", encoding="utf-8")
		np.savetxt(os.path.join(zs,"zs_output.csv"), z_score, delimiter=",", fmt="%s", encoding="utf-8")


def check_result_simple(path_destination,read_filename,csv=False,debug=False):
	array_np = np.load(os.path.join(path_destination,read_filename+".npy"))

	if debug == True:
		print(array_np,"\r\n")

	if csv == True:
		np.savetxt(os.path.join(path_destination, read_filename+".npy"), delimiter=",", fmt="%s", encoding="utf-8")