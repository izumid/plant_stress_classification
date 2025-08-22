import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import json

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

import md_logfile as lf

# MARK: New Window Size
def new_window_size(total_sample_size,start=1,step=1,reverse=True):
	list_window = []

	for i in range(start, total_sample_size, step):
		if total_sample_size % i ==0:
			sample = (total_sample_size/i)
			window = total_sample_size / sample
			if window >= 10 and sample >=100: list_window.append([int(window),int(sample)])
	
	aux = [[y,x] for x,y in list_window if [y,x] not in list_window]
	list_window = list_window + aux
	
	test =[]
	for x in list_window:
		if int(x[0] * x[1]) == 21888: test.append(True)
		else: test.append(False)

	list_window.sort(key=lambda x: x[0], reverse=reverse)
	
	return(list_window)


def try_cast(value,type=None):
	if type == float:
		try: return(float(value))
		except: return(value) 
	else:
		try: return(int(value))
		except: return(value) 

def read_config(path_absolute):
	try:
		with open(path_absolute, 'r') as f: data = json.load(f)
		return(data)
	except FileNotFoundError: print("File not found.")
	except json.JSONDecodeError: print("Invalid JSON format in file.")


# MARK: Windowing
def windowing(list_dataframe,x_column_name,y_column_name,list_window,path_destination,summarize,csv=False):
	"""
		While get slices of samples summarized then with statisticial distribution measures.
	"""

	cols = {
		"mean": "float64"
		,"inter_quartile_range": "float64"
		,"variance": "float64"
		,"standard_deviation": "float64"
		,"skew": "float64"
		,"kustosis": "float64"
		,"applied_stimulus": "int64"
	}

	filename = "dataset"

	for m,n in list_window:
		
		fix_window_data = []
		folder_window = f"{str(m)}x{str(n)}"
	
		for dataframe in list_dataframe:
			data = np.array(dataframe[x_column_name].copy())
			applied_stimulus = dataframe[y_column_name].iloc[0]
			for i in (range(0,len(data),n)):
				window = np.array(data[i:i+n])
				if summarize: fix_window_data.append([np.mean(window),stats.iqr(window),np.var(window),np.std(window),stats.skew(window),stats.kurtosis(window),applied_stimulus])
				else: fix_window_data.append([window.tolist(),applied_stimulus])

		if summarize:
			df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in cols.items()})
			df = pd.DataFrame(fix_window_data,columns=df.columns.tolist())
		else: 
			df = pd.DataFrame(fix_window_data, columns=["eletric_variation_value","applied_stimulus"])
		
		path_aux = os.path.join(path_destination,folder_window)
		
		if not os.path.exists(path_aux): os.makedirs(path_aux)
		
		if csv: df.to_csv(os.path.join(path_aux,filename+".csv"),sep=';',quotechar='"',encoding="utf-8-sig")
		else: df.to_feather(os.path.join(path_aux,filename+".feather"))

# MARK: Debug Code
def debug_code(debug,message,var=None):
	"""
	Description:
		Print messages across the process to verify data behaviour.

	Args:
		message(str): text to identify the code process the message are about;
		var(any): variable values to validade;
		debug(bool): true print's the messages;
	"""	

	if debug: 
		if var is None: print(f"{message};\r\n")
		else: print(f"{message}: \r\n{var};\r\n")


# MARK: Data Preprocessing
def rawdata(x_column_name,y_column_name,path_unified_resized,path_parent_root,unique_value,random_state,sample_size,path_class_split):
	
		print("pre processing...")

		dict_result = {x_column_name: "float", y_column_name: "int"}
		df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dict_result.items()})
		
		try:
			# -- Unify Data --
			# Unify each stimili into a single file
			if not os.path.exists(path_unified_resized): os.makedirs(path_unified_resized)

			for folder in os.listdir(path_parent_root):
				if "lectrodes" not in folder:
					path_absolute = os.path.join(path_parent_root,folder)
					np_array = [np.loadtxt(os.path.join(path_absolute,file)) for file in sorted(os.listdir(path_absolute))]
					np_array = np.concatenate(np_array)
					if unique_value: np_array = np.unique(np_array)

					rng = np.random.default_rng(random_state)
					np_array[:] = rng.permutation(np_array)
					np_array = np_array[:sample_size]
					print(np_array.shape,folder)
					
					np.save(os.path.join(path_unified_resized,f"{folder}.npy"), np_array)

		except Exception as error:
			lf.log_file(filename="log_file",header_message="Unify data",message=error)
		
		
		try:
			#-- Gather data (Single Dataframe) --
			if not os.path.exists(path_unified_resized): os.makedirs(path_unified_resized)
			
			for filename in sorted(os.listdir(path_unified_resized)):
				name_stimulus = Path(filename).stem.replace(' ','_').lower()
				original_array = np.load(os.path.join(path_unified_resized,filename))
				electo_values = np.random.choice(original_array, size=sample_size, replace=False)

				if "before" in name_stimulus: applied_stimulus = 0
				else: applied_stimulus = 1

				"""
				for i,char in enumerate(name_stimulus):
					if char == '_': count_char = i
			
				name_stimulus_prefix = name_stimulus[:count_char]
				"""

				temp_df = pd.DataFrame({
					x_column_name: electo_values
					,y_column_name: np.repeat(applied_stimulus, sample_size)
				})

				df = pd.concat([df, temp_df], ignore_index=True)
			
			df.to_feather(os.path.join(path_unified_resized,"unified_resized.feather"))
		except Exception as error:
			lf.log_file(filename="log_file",header_message="Gather data (Single Dataframe)",message=error)
			
		try:
			# -- Separate Not Event from Event --
			df_non_stimuled = df.query(f"{y_column_name} == 0").copy()
			df_stimuled = df.query(f"{y_column_name} == 1").copy()
			if not os.path.exists(path_class_split): os.makedirs(path_class_split)

			# -- Seperate class to correct windowing --
			#	1.Not Event
			df_non_stimuled = df_non_stimuled.sample(frac=1, random_state=random_state)
			df_non_stimuled.to_feather(os.path.join(path_class_split,"non_stimuled.feather"))
			
			#	2.Not Event
			df_stimuled = df_stimuled.sample(frac=1,random_state=random_state)
			df_stimuled.to_feather(os.path.join(path_class_split,"stimuled.feather"))
		except Exception as error:
			lf.log_file(filename="log_file",header_message="Separate Not Event from Event",message=error)
				

# MARK: Classify
def classify(path_base,list_window,path_destination,list_classifier,random_state,verbose,k_fold_split,filename="result"):
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
		folder_window = str(MxN[0])+'x'+str(MxN[1])
		path_txt = os.path.join(path_destination,f"{filename}_skfold_{MxN[0]}x{MxN[1]}.txt")
		if os.path.exists(path_txt): os.remove(path_txt)
		#with open(path_txt, mode="a") as file: file.write(";".join(map(str, header_txt)) + "\n")
		df_train = pd.read_feather(os.path.join(path_base,folder_window,"dataset.feather"))
		X = df_train.iloc[:, :-1]
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
				for train_index, test_index in skf.split(X, y):
					print(f"Model: {model}({folder_window}). Cross validation fold[{execution}] ({(rounds/total_rounds)*100:.2f}%)")

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
			
				print(f"Model: {model}({folder_window}). Train Test execution[{execution}] ({(rounds/total_rounds)*100:.2f}%)")

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


def result_feather_read(path_destination,filename,filter_model=False,dummy=False):
	pd.set_option('display.max_columns', None)  # Display all columns
	pd.set_option('display.max_rows', None)     # Display all rows

	path_absolute = os.path.join(path_destination,filename+".feather")
	df = pd.read_feather(path_absolute)
	if dummy == False:  df = df.query(f"model != 'DUM'")
	if filter_model != False: df = df.query(f"model == '{filter_model}'")
	print(df)


# MARK: Main
def main(config):
	path_parent_root = os.path.join(os.path.dirname(os.getcwd()),"original_data")
	#path_root = os.path.join(os.getcwd(),"experiments_data",Path(os.path.realpath(__file__)).stem)
	path_root = os.path.join(os.getcwd(),r"data")
	unique_value=config["unique_value"]
	balanced_sample = config["balanced_sample"]
	summarize = config["summarize"]

	if unique_value:
		if balanced_sample: path_unified_resized = os.path.join(path_root,r"01_unified_resized\01_value_unique\01_balanced")
		else: 				path_unified_resized = os.path.join(path_root,r"01_unified_resized\01_value_unique\02_unbalanced")
	else:
		if balanced_sample: path_unified_resized = os.path.join(path_root,r"01_unified_resized\02_value_duplicate\01_balanced")
		else: 				path_unified_resized = os.path.join(path_root,r"01_unified_resized\02_value_duplicate\02_unbalanced")

	path_class_split = os.path.join(path_root,"02_class_split")

	if summarize: path_base = os.path.join(path_root,"03_summarized_window")
	else: path_base = os.path.join(path_root,"04_window")

	#path_split = os.path.join(path_root,"04_split")
	path_result = os.path.join(path_root,r"05_experiment_result",config["destination_folder_name"])
	debug = config["debug"]

	x_column_name = "electro_value"
	y_column_name = "applied_stimulus"

	pd.set_option('display.max_colwidth', None)
	
	if len(config["list_window"]) < 0: list_window = new_window_size(total_sample_size=config["sample_size"])
	else: list_window = config["list_window"]

	sample_size = int(list_window[0][0] * list_window[0][1])
	print(f"sample size: {sample_size}")
	list_classifier = config["list_classifier"]
	k_fold_split = config["k_fold_split"]

	# used process
	if config["pre_processing_data"]:
		rawdata(
			x_column_name=x_column_name
			,y_column_name=y_column_name
			,path_unified_resized=path_unified_resized
			,path_parent_root=path_parent_root
			,unique_value=unique_value
			,random_state=config["random_state"]
			,sample_size=sample_size
			,path_class_split=path_class_split	
		)

		df_non_stimuled = pd.read_feather(os.path.join(path_class_split,"non_stimuled.feather"))
		df_stimuled = pd.read_feather(os.path.join(path_class_split,"stimuled.feather"))

		windowing(list_dataframe=[df_non_stimuled,df_stimuled],x_column_name=x_column_name,y_column_name=y_column_name,list_window=list_window,path_destination=path_base,summarize=summarize,csv=False)
		

	if config["classify"]:

		if k_fold_split == 0: debug_code(message="Classification runnning: Train test split(80%-20%)",debug=debug)
		else: debug_code(message="Classification running K-fold", var=k_fold_split,debug=debug)

		
		debug_code(message="path_destination", var=path_result,debug=debug)

		classify(
			path_base=path_base
			,list_window=list_window
			,path_destination=path_result
			,list_classifier=list_classifier
			,random_state = config["random_state"]
			,verbose = config["verbose"]
			,k_fold_split = k_fold_split
			#,filename=str(Path(os.path.realpath(__file__)).stem)
		)

		#result_feather_read(path_destination=path_result,filename=str(Path(os.path.realpath(__file__)).stem)+"_stratified_kfold")


def mult_text_to_csv(path_origin,path_destination,all_has_title,filename):
	data = []
	count = 0

	for file in os.listdir(path_origin):
		with open(os.path.join(path_origin,file)) as file:
			if all_has_title == True and count != 0: next(file)
			for line in file:
				aux = line.split(";")
				data.append([try_cast(x,float) if aux.index(x) > 2 else try_cast(x,int) for x in aux])
		count +=1
	
	header = data.pop(0)
	header = [x.replace("\n",'') for x in header]

	df = pd.DataFrame(data, columns=header)
	df.sort_values(by="window",ascending=False,kind="stable",inplace=True)
	df.to_csv(os.path.join(path_destination,f"{filename}.csv"),sep=';',quotechar='"',encoding="utf-8-sig")
	df.to_feather(os.path.join(path_destination,f"{filename}.feather"))

if __name__ == "__main__":
	
	try: 
		config = read_config(os.path.join(os.getcwd(),"config/config.json"))
		
		# if int(input("Type 1 to show windows list: ")):

		# 	list_window = config["list_window"]
		# 	if len(list_window) < 0:
		# 		sample_size = int(list_window[0][0] * list_window[0][1])
		# 		print(new_window_size(total_sample_size=config["sample_size"]))
		# 	else: print(list_window)
				
		# if int(input("Type 1 to start process: ")):
		main(config)

		# if int(input("Unify separated txt base files: ")):
		# 	path_origin = os.path.join(os.getcwd(),"experiments_data",Path(os.path.realpath(__file__)).stem,"05_result")
		# 	path_destination = os.path.join(Path(path_origin).parent, "06_result_unified")
		# 	if not os.path.exists(path_destination): os.makedirs(path_destination)
		# 	mult_text_to_csv(path_origin=path_origin, path_destination=path_destination, all_has_title=True,filename=Path(os.path.realpath(__file__)).stem+"_stratified_kfold")

	except Exception as e: 
		print(f"An error occurred: {e}")

	if 1== 0:
		while True:
			user_input = input('Type "Abnegation" to exit: ')
			if user_input == "Abnegation": break