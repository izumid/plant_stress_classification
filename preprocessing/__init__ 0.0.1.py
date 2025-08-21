import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

from pathlib import Path
from scipy import stats

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

# Deprecated
def data_normalization(path_base,path_destination,separated_stimuli=False,debug=False):
	
	if debug == False:
		if separated_stimuli == False:
			# How the interest is in identify stimulus presence or not, all the stimulus training or test are joined to get the min and max of all stimulus
			if not os.path.exists(path_destination): os.makedirs(path_destination)

			data_norm = [np.load(os.path.join(path_base,file)) for file in sorted(os.listdir(path_base)) if "train" in file]
			data_norm = np.concatenate(data_norm).reshape(-1,1)
			scaler = MinMaxScaler().fit(data_norm)
			
			for file in os.listdir(path_base):
				abs_path = os.path.join(path_base,file)
				if os.path.isfile(abs_path):
					data = np.load(os.path.join(path_base,file)).reshape(-1,1)
					data = scaler.transform(data)
					data = data.reshape(-1)
					print(f"folder: {file}{data.shape}, data: {data}")
					np.save(os.path.join(path_destination,Path(file).stem), data)
		else:
			# Its a good a ideia to compare difference of gathared and separated stimulus normalization
			for file in os.listdir(path_base):
				data = np.load(os.path.join(path_base,file)).reshape(-1,1)
				scaler = MinMaxScaler().fit(data)
				print(f"For file '{file}':\n\tMax value is {scaler.data_max_}.\n\tMin value is: {scaler.data_min_}.\n")
				data = scaler.transform(data)
				print(data[:10],data[-10:])
				print(f"For normalized file '{file}':\n\tMax value is {np.max(data)}.\n\tMin value is: {np.min(data)}.\n\n\n\n")
			
				np.save(os.path.join(path_destination,Path(file).stem+"_separated"), data)
	else:
		data_validation(path_destination)


def get_fitted_scaler(path_base,scaler_type):
	data_fit = [np.load(os.path.join(path_base,file)) for file in sorted(os.listdir(path_base))]
	data_fit = np.concatenate(data_fit).reshape(-1,1)
	scaler = None

	match scaler_type:
		case "standard":
			scaler = (StandardScaler()).fit(data_fit)
		case "min_max":
			scaler = (MinMaxScaler()).fit(data_fit)
		case "z_score":
			scaler = ZScoreScaler().fit(data_fit)
	return(scaler)


# MARK: Change Scale
def change_scale(path_base,path_destination,scaler_type,debug=False):
	# How the interest is in identify stimulus presence or not, all the stimulus training or test are joined to get the min and max of all stimulus
	if not os.path.exists(path_destination): os.makedirs(path_destination)

	scaler = get_fitted_scaler(path_base,scaler_type)

	for file in os.listdir(path_base):
		path_absolute = os.path.join(path_base,file)
		if os.path.isfile(path_absolute):
			data = np.load(os.path.join(path_base,file)).reshape(-1,1)
			data = scaler.transform(data)
			data = data.reshape(-1)
			
			code_debug(message=f"data_normalization() -> folder: {file}{data.shape}, data: {data}",debug=debug)
			np.save(os.path.join(path_destination,Path(file).stem+".npy"), data)
	if debug == True: data_validation(path_destination)


def window_size(sample_size,n,min_sample,test_combination,debug=False):
	denominators = []
	check = []
	aux=[]
	window_size_collection = []
	# n = 200
	#window_registries_limit = 100

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

	return(window_size_collection)


def fixed_window(path_base,path_destination,list_window,debug=False):
	
	for file in os.listdir(path_base):
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
				
				#path_absolute = os.path.join(path_destination,f"{m}x{n}")
		
				path_absolute = os.path.join(path_destination,Path(file).stem,f"{str(m)}x{str(n)}")
				if not os.path.exists(path_absolute): os.makedirs(path_absolute)

				np.save(os.path.join(path_absolute,"X.npy"), fix_window_data)


def summarize(path_base,file_name_predictor,path_destination,debug=False):
	file_name_predictor = file_name_predictor+".npy"
	
	for folder in os.listdir(path_base):
		aux =  os.path.join(path_base,folder)
		for sub_folder in os.listdir(aux):
			path_base_absolute = os.path.join(aux,sub_folder,file_name_predictor)

			if  os.path.isfile(path_base_absolute):
				np_array = np.load(path_base_absolute)
			
				summarized_array = np.array([
					[np.mean(w),stats.iqr(w),np.var(w),np.std(w),stats.skew(w),stats.kurtosis(w)] for w in np_array
				])
			
				code_debug(message=f"File{file_name_predictor}.\r\nsummarized_array shape: {summarized_array.shape}.\r\nsummarized_array values: {summarized_array}",debug=debug)

				path_absolute = os.path.join(path_destination,folder,sub_folder)
				if not os.path.exists(path_absolute): os.makedirs(path_absolute)
				np.save(os.path.join(path_absolute,file_name_predictor), summarized_array)
		

def windowed_target_variable(path_base,file_name_target):
	file_name_target = file_name_target+".npy"

	for folder_stimulus in os.listdir(path_base):
		path_folder_stimulus = os.path.join(path_base,folder_stimulus)
		
		for folder_window in os.listdir(path_folder_stimulus):
			path_folder_window = os.path.join(path_folder_stimulus,folder_window)
			if not os.path.exists(path_folder_window): os.makedirs(path_folder_window)

			for file in os.listdir(path_folder_window):
				path_absolute_predictor = os.path.join(path_folder_window,file)

				if os.path.isfile(path_absolute_predictor):
					npl = np.load(path_absolute_predictor)

					if path_folder_window.find("fter") != -1: array_target = np.ones(len(npl), dtype=int)
					else: array_target = np.zeros(len(npl), dtype=int)
					
					#array_target = array_target.reshape(-1,1)

					path_absolute_target = os.path.join(path_folder_window,file_name_target)
					np.save(path_absolute_target,array_target)