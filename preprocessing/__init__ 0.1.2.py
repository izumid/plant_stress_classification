import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

from pathlib import Path
from scipy import stats

import pandas as pd

class ZScoreScaler:
	def fit(self, data):
		self.data = data
		return self
	
	def transform(self, data):
		return zscore(data)
				

def code_debug(message,var=None,debug=False):
	if not debug is False:
		if not var is None: print(f"[Debug]: {message}, variable value: {var}.")
		else: print(f"[Debug]: {message}.")


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

def data_union(path_base,path_destination,sample_size=None,shuffle=False,debug=False):
	if not os.path.exists(path_destination): os.makedirs(path_destination)

	for folder in os.listdir(path_base):
		if "lectrodes" not in folder:
			path_absolute = os.path.join(path_base,folder)
			data = [np.loadtxt(os.path.join(path_absolute,file)) for file in sorted(os.listdir(path_absolute))]
			data = np.concatenate(data)
			if not sample_size is None:  data = data[:(sample_size)]
			code_debug(message=f"data_union() -> folder: {folder},({len(data)}), data: {data}",debug=debug)

			if shuffle == True:
				code_debug(message="Shuflling",var=shuffle,debug=debug)
				rng = np.random.default_rng(42)
				data[:] = rng.permutation(data) 
			
			np.save(os.path.join(path_destination,folder+".npy"), data)

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


# MARK: Change Scale
def change_scale(path_base,path_destination,column_name,type_split,debug=False):
	filename = f"{type_split}.feather"
	# How the interest is in identify stimulus presence or not, all the stimulus training or test are joined to get the min and max of all stimulus
	list_stimulus_order = []
	data = []

	for folder_stimulus in sorted(os.listdir(path_base)):
		list_stimulus_order.append(folder_stimulus)
		path_aux = os.path.join(path_base,folder_stimulus)

		for file in os.listdir(path_aux):
			if type_split in file:			
				data.append([folder_stimulus,np.load(os.path.join(path_aux,file))])

	labels = np.repeat([item[0] for item in data], [len(item[1]) for item in data])
	values = np.concatenate([item[1] for item in data])

	df = pd.DataFrame({"stimulus": labels, column_name: values})
	df["standard_scaler"] = StandardScaler().fit_transform(df[[column_name]])
	df["min_max_scaler"] = MinMaxScaler().fit_transform(df[[column_name]])
	df["z_score"] = zscore(df[column_name])
	code_debug("Dataframe train header",df.head(),debug)

	for stimulus in df['stimulus'].unique().tolist():
		np_array = df.query(f"stimulus == '{stimulus}'")["standard_scaler"].to_numpy()
		#path_aux = os.path.join(path_destination,stimulus)

		#if not os.path.exists(path_aux): os.makedirs(path_aux)
		#np.save(os.path.join(path_aux,f"{type_split}.npy"), np_array)
		
		if not os.path.exists(path_destination): os.makedirs(path_destination)
		np.save(os.path.join(path_destination,f"{stimulus}_{type_split}.npy"), np_array)

	#if not os.path.exists(path_destination): os.makedirs(path_destination)
	#dataframe.to_feather(os.path.join(path_destination,filename))


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