import os
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

				
# MARK: Window Size
def window_size(sample_size,n=200,min_sample=100,test_combination=1000,debug=False):
	denominators = []
	check = []
	aux=[]
	window_size_collection = []
	# n = 200
	#window_registries_limit = 100

	while not window_size_collection:
		for i in range(1,test_combination):
			if(sample_size / i) % i == 0 : denominators.append(i)
		
		count=0
		for i in denominators: 
			for j in range(1,10):
				window = int(i*(n/j))
				if (sample_size % window == 0) and (window >= min_sample):
					if window not in aux:
						aux.append(window)
						window_size_collection.append((int(window), int(sample_size/window)))
						count+=1
		
		if debug == True: 
			for i in check: print(i)
		aux = sorted(aux)
		window_size_collection = sorted(window_size_collection)
		sample_size = sample_size - 10

	return(sample_size+10,window_size_collection)


def code_debug(message,var=None,debug=False):
	if not debug is False:
		if not var is None: print(f"[Debug]: {message}, variable value: {var}.")
		else: print(f"[Debug]: {message}.")

# MARK: Validation Data
def data_validation(path_base):
	base_previous = []
	count = 0
	old_file = "Null"
	for file in os.listdir(path_base):
		path_absolute = os.path.join(path_base,file)
		base_current = np.load(path_absolute)
		
		print(f"{count}. '{file}'{base_current.shape} base its equal to '{old_file}': {np.array_equal(base_current,base_previous)}!")
		print(f"\tFirsts registries of current file: {base_current[:10]}\n\tLasts: {base_current[-10:]}\n\n")
		base_previous = base_current
		old_file = file
		count += 1

# MARK: Union
def data_union(path_base,path_destination,sample_size=None,shuffle=False,debug=False):
	if not os.path.exists(path_destination): os.makedirs(path_destination)

	for folder in os.listdir(path_base):
		if "lectrodes" not in folder:
			path_absolute = os.path.join(path_base,folder)
			np_array = np.concatenate([np.loadtxt(os.path.join(path_absolute,file)) for file in sorted(os.listdir(path_absolute))])
			code_debug(message=f"data_union() -> folder: {folder},({np_array.shape}), np_array: {np_array}",debug=debug)
		
			if shuffle == True:
				code_debug(message="Shuflling",var=shuffle,debug=debug)
				rng = np.random.default_rng(42)
				np_array[:] = rng.permutation(np_array) 
			
			if not sample_size is None:  np_array = np_array[:(sample_size)]
			
			np.save(os.path.join(path_destination,folder+".npy"), np_array)

	if debug == True: data_validation(path_destination)
				
# MARK: Split Data
def data_split(path_base,path_destination,debug=False):

	for file in os.listdir(path_base):
		data = np.load(os.path.join(path_base,file))
		file = Path(file).stem
		size_train = int(len(data) * 0.8)
		size_test = int(len(data) * 0.2)
		
		# print(f"train size: {size_train}; test size: {size_test}; sum: {size_train+size_test};")

		data_train = data[:size_train]
		data_test= data[-size_test:]
		print(f"folder: {file},({len(data_train)}), data: {data_train}")
		print(f"folder: {file},({len(data_test)}), data: {data_test}")
		
		aux  = os.path.join(path_destination,file)
		if not os.path.exists(aux): os.makedirs(aux)

		np.save(os.path.join(aux,"train.npy"), data_train)
		np.save(os.path.join(aux,"test.npy"), data_test)
	
	if debug == True: data_validation(aux)


def aux_getdata(path_base,type_split,x_column_name):
	list_stimulus_order = []
	data = []

	for folder_stimulus in sorted(os.listdir(path_base)):
		list_stimulus_order.append(folder_stimulus)
		path_aux = os.path.join(path_base,folder_stimulus)

		for file in os.listdir(path_aux):
			if type_split in file: data.append([folder_stimulus,np.load(os.path.join(path_aux,file))])
	

	labels = np.repeat([item[0] for item in data], [len(item[1]) for item in data])
	values = np.concatenate([item[1] for item in data])
	dataframe = pd.DataFrame({"stimulus": labels, x_column_name: values})
	
	return(dataframe)

def aux_save(dataframe,path_destination,type_split):
	for stimulus in dataframe['stimulus'].unique().tolist():
		np_array = dataframe.query(f"stimulus == '{stimulus}'")["standard_scaler"].to_numpy()
		#path_aux = os.path.join(path_destination,stimulus)

		#if not os.path.exists(path_aux): os.makedirs(path_aux)
		#np.save(os.path.join(path_aux,f"{type_split}.npy"), np_array)
		
		if not os.path.exists(path_destination): os.makedirs(path_destination)
		np.save(os.path.join(path_destination,f"{stimulus}_{type_split}.npy"), np_array)

# MARK: ZScore Class (Easier Process)
class ZScoreScaler:
	def fit(self, data):
		self.data = data
		return self
	
	def transform(self, data):
		return stats.zscore(data)

# MARK: Change Scale
def change_scale(path_base,path_destination,x_column_name,debug=False):
	# How the interest is in identify stimulus presence or not, all the stimulus training or test are joined to get the min and max of all stimulus

	# Train data	
	df_train = aux_getdata(path_base=path_base,type_split="train",x_column_name=x_column_name)
	code_debug("Dataframe train header",df_train.head(),debug)

	scaler_std = StandardScaler().fit(df_train[[x_column_name]])
	scaler_mm = MinMaxScaler().fit(df_train[[x_column_name]])
	scaler_zs = ZScoreScaler().fit(df_train[[x_column_name]])

	df_train["standard_scaler"] = scaler_std.transform(df_train[[x_column_name]])
	df_train["min_max_scaler"] = scaler_mm.transform(df_train[[x_column_name]])
	#df_train["z_score"] = stats.zscore(df_train[x_column_name])
	df_train["z_score"] = scaler_zs.transform(df_train[x_column_name])
	code_debug("Dataframe train scaled header",df_train.head(),debug)

	aux_save(dataframe=df_train,path_destination=path_destination,type_split="train")

	# Test data
	df_test =  aux_getdata(path_base=path_base,type_split="test",x_column_name=x_column_name)
	code_debug("Dataframe test header",df_train.head(),debug)
	df_test["standard_scaler"] = scaler_std.transform(df_test[[x_column_name]])
	df_test["min_max_scaler"] = scaler_mm.transform(df_test[[x_column_name]])
	#df_test["z_score"] = stats.zscore(df_test[x_column_name])
	df_test["z_score"] = scaler_zs.transform(df_test[x_column_name])
	code_debug("Dataframe test scaled header",df_train.head(),debug)
	aux_save(dataframe=df_test,path_destination=path_destination,type_split="test")

	#if not os.path.exists(path_destination): os.makedirs(path_destination)
	#dataframe.to_feather(os.path.join(path_destination,filename))


# MARK:Fixed Window
def fixed_window(path_base,path_destination,list_window,debug=False):
	for file in os.listdir(path_base):
		find_underscore = file.find('_')
		folder_stimulus_name =  file[:find_underscore]
		filename = f"X{file[find_underscore:]}"

		absolute_path = os.path.join(path_base,file)

		if os.path.isfile(absolute_path):
			data = np.load(absolute_path)
			for m,n in list_window:
				
				#window(M) x sample(N) = [MxN]
				fix_window_data = []

				c=0
				for i in (range(0,len(data),n)):
					window = data[i:i+n]
					code_debug(message=f"window {c}: [{i}, {i+n}]. Size {len(window)}", debug=debug)
					fix_window_data.append(window)
					c+=1

				#path_absolute = os.path.join(path_destination,Path(file).stem,f"{str(m)}x{str(n)}")
				path_absolute = os.path.join(path_destination,folder_stimulus_name,f"{str(m)}x{str(n)}")
				if not os.path.exists(path_absolute): os.makedirs(path_absolute)

				np.save(os.path.join(path_absolute,filename), fix_window_data)


# MARK: Summarize
def summarize(path_base,path_destination,file_name_predictor,debug=False):
	file_name_predictor = file_name_predictor+".npy"
	
	for folder in os.listdir(path_base):
		aux =  os.path.join(path_base,folder)
		for sub_folder in os.listdir(aux):
			path_aux = os.path.join(aux,sub_folder)
			for filename in os.listdir(path_aux):
				if filename[0] == "X":
					path_base_absolute = os.path.join(path_aux,filename)
				#if os.path.isfile(path_base_absolute):
					np_array = np.load(path_base_absolute)
				
					summarized_array = np.array([
						[np.mean(w),stats.iqr(w),np.var(w),np.std(w),stats.skew(w),stats.kurtosis(w)] for w in np_array
					])
				
					code_debug(message=f"File{file_name_predictor}.\r\nsummarized_array shape: {summarized_array.shape}.\r\nsummarized_array values: {summarized_array}",debug=debug)

					path_absolute = os.path.join(path_destination,folder,sub_folder)
					if not os.path.exists(path_absolute): os.makedirs(path_absolute)
					np.save(os.path.join(path_absolute,filename), summarized_array)
		
# MARK: Target Value
def windowed_target_variable(path_base):
	for folder_stimulus in os.listdir(path_base):
		path_folder_stimulus = os.path.join(path_base,folder_stimulus)
		
		for folder_window in os.listdir(path_folder_stimulus):
			path_folder_window = os.path.join(path_folder_stimulus,folder_window)
			if not os.path.exists(path_folder_window): os.makedirs(path_folder_window)

			for filename in os.listdir(path_folder_window):
				if filename[0] == 'X':
					path_absolute_predictor = os.path.join(path_folder_window,filename)

					if os.path.isfile(path_absolute_predictor):
						npl = np.load(path_absolute_predictor)

						if path_folder_window.find("fter") != -1: array_target = np.ones(len(npl), dtype=int)
						else: array_target = np.zeros(len(npl), dtype=int)
						
						#array_target = array_target.reshape(-1,1)

						path_absolute_target = os.path.join(path_folder_window,filename.replace('X','y'))
						np.save(path_absolute_target,array_target)