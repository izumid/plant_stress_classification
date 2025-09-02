import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import md_util as ut


def new_window_size(total_sample_size,start=1,step=1,reverse=True,debug=False):
	list_window = []

	for i in range(start, total_sample_size, step):
		if total_sample_size % i == 0:
			sample = (total_sample_size/i)
			window = total_sample_size / sample
			if window >= 10 and sample >=100: list_window.append([int(window),int(sample)])
	
	aux = [[y,x] for x,y in list_window if [y,x] not in list_window]
	list_window = list_window + aux
	list_window.sort(key=lambda x: x[0], reverse=reverse)

	if debug:list_window = [min(list_window, key=lambda x: x[0])]
	
	return(list_window)


def window_approved(sample_class_size,window):
	avoid_decimal = sample_class_size / window
	avoid_remainder = sample_class_size % window
	
	if 1 == 0:
		missed = [524000,327500,262000,209600,163750,131000,104800,81875,65500]
		if window in missed:
			print(window, avoid_decimal,avoid_remainder)

	if avoid_decimal > 0 and avoid_remainder == 0: 
		window = (sample_class_size / window)
		return(window)

#print(window_check_valid(3648,114))
#print(type(window_check_valid(96980,114)))


def window_shape(sample_class_size,minimum_observation_length=None,start=1,step=1,reverse=True,imbalanced=False,show_debug_message=False):
	"""
		Description:
			if experiment is unbalanced must guarantee that always get the same number of observations to each windows, avoinding this: 
				[Fixed Window Dataset] File Cold after.npy length(3647). Summarizing window[0:9698];
		Arguments:
		
	"""

	list_window = []

	for i in range(start,sample_class_size,step):
		window = window_approved(sample_class_size=sample_class_size,window=i)
		
		if not window is None:
			window_sample = sample_class_size / window
			if window >= 10 and window_sample >= 100: 
				list_window.append([int(window),int(window_sample)])
	
	if 1 == 0:
		for m,n in list_window:
			print(f"m: {m}, n: {n}")

	window_opposite_combination = [[n,m] for m,n in list_window if [n,m] not in list_window and not window_approved(sample_class_size,n) is None]
	list_window = list_window + window_opposite_combination
	list_window.sort(key=lambda x: x[0], reverse=reverse)

	if 1 == 0:
		if imbalanced:
			check = []
			#minimum_observation_length = min(sample_unique_length.values())
			for window in list_window:

				sample_length = window[1]
						
				if (minimum_observation_length / sample_length) < 0 or (minimum_observation_length % sample_length) != 0: check.append(True)
				else: check.append(False)
		
			filtered = [val for val, flag in zip(list_window, check) if not flag]
			list_window = filtered
	else:
		if imbalanced:
			check = []
			for window in list_window:
				sample_length = window[1]
				valid_window = window_approved(minimum_observation_length,sample_length)

				if valid_window is None: check.append(True)
				else: check.append(False)
		
			filtered = [val for val, flag in zip(list_window, check) if not flag]
			list_window = filtered

	if show_debug_message:list_window = [min(list_window, key=lambda x: x[0])]
	
	return(list_window)


def validated_window_size():

	sample_unique_length = {
			"Cold after": 3648,
			"Cold before": 3648,
			"Low light after": 10971,
			"Low light before":	10971,
			"Manitol after": 33871,
			"Manitol before": 33871
	}
	
	sample_class_size = sum(sample_unique_length.values())
	minimum_observation_length =  min(sample_unique_length.values())

	#"if sample_class_size / i > 0 and sample_class_size % i == 0:" VS "if sample_class_size % i == 0:" #Seems no diference
	
	# -- old -- 
	#window = new_window_size(total_sample_size=5240000,debug=False) 

	# -- new -- 
	#window = (window_shape(sample_class_size=5_240_000,show_debug_message=False)) 1st and 4th experiment
	window = window_shape(sample_class_size=sample_class_size,minimum_observation_length=minimum_observation_length,imbalanced=True,show_debug_message=False) #2nd experiment
	#window = (window_shape(sample_class_size=21888,show_debug_message=False)) # 3th experiment
	print(f"window total: {len(window)}")
	for w in window:
		print(f"{w[0]}, {w[1]}")

#validated_window_size()


def find_min_sample_value():
	sample_unique_length = {
			"Cold after": 3648,
			"Cold before": 3648,
			"Low light after": 10971,
			"Low light before":	10971,
			"Manitol after": 33871,
			"Manitol before": 33871
	}
	sample_class_size = sum(sample_unique_length.values())
	minimum_observation_length =  min(sample_unique_length.values())
	print(minimum_observation_length,"AAAA")

	check = []
	for i in range(0,3648,1):
		print(i)
		#window = window_shape(sample_class_size=sample_class_size,minimum_observation_length=minimum_observation_length,imbalanced=True,show_debug_message=False)
		window = window_shape(sample_class_size=96980,minimum_observation_length=i,imbalanced=True,show_debug_message=False)
		if window != []: check.append((i,window))
	

	combination = sorted(check, key=lambda x: len(x[1]), reverse=True)

	for combination in check:
		print(combination)

#find_min_sample_value()


def imbalanced_window():
	#samples = [3648,10971,33871] #0
	#samples = [3600,10000,30000] #10
	#samples = [3000,10000,30000] #11
	samples = [3000,9000,30000] #25
	#samples = [3600,9000,30000] #17
	
	work_all_sample_size = []

	if 1==0:
		for i in range(1,3648,1):
			#print(i)
			aux = []
			for s in samples:
				x = window_approved(s,i)
				if x is None: break
				else: aux.append(x)

			if len(aux) == 3: work_all_sample_size.append(aux)
	else:
		for s in samples:
			aux = []
			for i in range(1,3648,1):
				#print(s, i)
				window = window_approved(s,i)
				if not window is None: aux.append(window)

			work_all_sample_size.append((s,aux))
			
	
	for sample_size in work_all_sample_size:
		print(sample_size)

	x = work_all_sample_size[0][1]
	x.extend(work_all_sample_size[1][1])
	x.extend(work_all_sample_size[2][1])
	x.sort()

	print("\r\nsorted:",x)

	print("\r\nrepeated")
	valid = []
	n = 3
	for item in set(x):  # Iterate through unique elements to avoid redundant checks
		if x.count(item) == n: valid.append(int(item))

	print(len(valid),valid)

#imbalanced_window()


def unique_total_value():
	path_relative = r"data\00_stimulus_data_joined"
	path_origin = os.path.join(os.getcwd(), path_relative)

	for file in os.listdir(path_origin):
		stimulus_data = np.load(os.path.join(path_origin,file))
		stimulus_data = np.unique(stimulus_data)
		print(f"File: {file}, unique: {len(stimulus_data)}")

#unique_total_value()

#MARK: Win. Dataset
def fixed_window_dataset(path_destination,list_window,path_origin,event_basefile_therm,show_debug_message,dataset_structure):
	"""
		Description: 

		Arguments:
			x_column_name(string): ;
			y_column_name(string): ;
			path_origin(string): ;
			path_parent_root(string): ;
			sample_size(int): ;
	"""

	try:
		if not os.path.exists(path_destination): 
			os.makedirs(path_destination)

			for m,n in list_window:
				fix_window_data = []
				file_name = f"{str(m)}x{str(n)}"				
				ut.debug(message=f"[Fixed Window Dataset] data origin: {path_origin}",show=show_debug_message)
				ut.debug(message=f"[Fixed Window Dataset] File: {file_name}",show=show_debug_message)

				for stimulus_file in os.listdir(path_origin):
					#stimulus_dataset = pd.read_feather(os.path.join(path_origin,stimulus_file))
					stimulus_stage = os.path.splitext(stimulus_file)[0].replace(' ','_').lower()
					stimulus_value = np.load(os.path.join(path_origin,stimulus_file))
					
					if event_basefile_therm in stimulus_stage: stimulus_applied = 0
					else: stimulus_applied = 1

					for index_start in (range(0,len(stimulus_value),n)):
						index_end = index_start+n
						window = np.array(stimulus_value[index_start:index_end])
						ut.debug(message=f"[Fixed Window Dataset] File {stimulus_file} length({len(stimulus_value)}). Summarizing window[{index_start}:{index_end}]",show=show_debug_message)

						fix_window_data.append(
							[
								stimulus_stage
								,np.mean(window)
								,stats.iqr(window)
								,np.var(window)
								,np.std(window)
								,stats.skew(window)
								,stats.kurtosis(window)
								,stimulus_applied
							]
						)

				df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dataset_structure.items()})
				df = pd.DataFrame(fix_window_data,columns=df.columns.tolist())
				df.to_feather(os.path.join(path_destination,file_name+".feather"))
	except Exception as error:
		ut.log_file(filename="log_file",header_message="dataset_stimulus_class: generate dataset with stimulus_name_stage, stimulus_value, applied_stimulus")


##########################################################
## 						WRANGLING						##
##########################################################


#MARK: Win. Shape
def window_shape(total_sample_size,sample_unique_length=None,start=1,step=1,reverse=True,imbalanced=False,show_debug_message=False):
	"""
		Description:
			if experiment is unbalanced must guarantee that always get the same number of observations to each windows, avoinding this: 
				[Fixed Window Dataset] File Cold after.npy length(3647). Summarizing window[0:9698];
		Arguments:
		
	"""

	list_window = []

	for i in range(start, total_sample_size, step):
		if total_sample_size % i ==0:
			sample = (total_sample_size/i)
			window = total_sample_size / sample
			if window >= 10 and sample >=100: list_window.append([int(window),int(sample)])
	
	aux = [[y,x] for x,y in list_window if [y,x] not in list_window]
	list_window = list_window + aux
	list_window.sort(key=lambda x: x[0], reverse=reverse)

	if imbalanced:
		minimum_observation_length = min(sample_unique_length.values())

		for i in range(len(list_window)):
			window = list_window[i] 
			if (window[1] / minimum_observation_length) < 0 and (window[1] % minimum_observation_length) == 0:
				list_window.remove(i)


	if show_debug_message:list_window = [min(list_window, key=lambda x: x[0])]
	
	return(list_window)


#MARK: Join Data
def join_stimulus_data(random_state,path_destination,path_origin,show_debug_message):
	"""
		Description: Unify each stimulus per class to his own single file, limiting the dataset with the desired sample size;

		Arguments:
			path_destination(string): 
			path_origin(string): input data path. In that case saved in N .txt fikes;
			path_parent_root(string): the main output folder to save data as numpy arrays and start processing the steps;
			random_state(int): pseudo random to chose registries (observations) of dataset;
			sample_size(int): observations size of the main dataset;
			show_debug_message(boolean): if true print a message, variables values are optional;
	"""


	np.random.seed(random_state) #setting random seed globaly

	try:
		if not os.path.exists(path_destination): 
			ut.debug(message="[Joining stimulus data]",show=show_debug_message)
			os.makedirs(path_destination)

			for folder in os.listdir(path_origin):
				if "lectrodes" not in folder:
					path_absolute = os.path.join(path_origin,folder)
					joined_data = [np.loadtxt(os.path.join(path_absolute,file)) for file in sorted(os.listdir(path_absolute))]
					joined_data = np.concatenate(joined_data)
					np.random.shuffle(joined_data) 

					ut.debug(var=f"{joined_data.shape},{folder}",message="join_stimulus_data",show=show_debug_message)
					np.save(os.path.join(path_destination,f"{folder}.npy"), joined_data)

	except Exception as error:
		ut.log_file(filename="log_file",header_message="stimulus_category_class_reduced: txt files to numpy")


#MARK: Exp. Data
def experiment_data(path_destination,path_origin,sample_size,unique_value,balanced_sample,sample_unique_length,show_debug_message):
	"""
		Description: Unify each stimulus per class to his own single file;

		Arguments:
			path_origin(string): ;
			path_parent_root(string): ;
			unique_value(boolean): ;
			random_state(int): ;
			sample_size(int): ;		
	"""
	
	# setting random seed globaly;
	# Randomly collecting data again to ensure diversity of observations;
	# np.random.seed(random_state) 

	try:
		if not os.path.exists(path_destination):
			os.makedirs(path_destination)

			for file in os.listdir(path_origin):
				stimulus_data = np.load(os.path.join(path_origin,file))
				#stimulus_data_size = len(stimulus_data)

				if unique_value: 
					stimulus_data = np.unique(stimulus_data)
					if balanced_sample: sample_size = sample_unique_length["Cold after"]
					else: sample_size = sample_unique_length[Path(file).stem]
				
				ut.debug(message=f"[Experiment Data] File: {file}, sample size: {sample_size}, data length: {len(stimulus_data)}",show=show_debug_message)
				stimulus_data = stimulus_data[:(sample_size-1)] # As we dealing the indexes -1, don't do that if was lenght
				
				#if stimulus_data_size >= sample_size: 
					#stimulus_data = np.random.choice(stimulus_data,size=sample_size,replace=False) 
				#else: stimulus_data = np.random.choice(stimulus_data,size=stimulus_data_size,replace=False)

				
				np.save(os.path.join(path_destination,file), stimulus_data)

	except Exception as error:
		ut.log_file(filename="log_file",header_message="Experiment Data")


#MARK: Split Window
def window_fixed(path_destination,list_window,path_origin,show_debug_message,dataset_structure):
	"""
		Description: 

		Arguments:
			path_origin(string): ;
			path_parent_root(string): ;
			sample_size(int): ;
	"""

	try:
		if not os.path.exists(path_destination): 
			os.makedirs(path_destination)

			for m,n in list_window:
				window_data = []
				file_name = f"{str(m)}x{str(n)}"				
				ut.debug(message=f"[Fixed Window Dataset] data origin: {path_origin}",show=show_debug_message)
				ut.debug(message=f"[Fixed Window Dataset] File: {file_name}",show=show_debug_message)

				for stimulus_file in os.listdir(path_origin):
					stimulus_value = np.load(os.path.join(path_origin,stimulus_file))

					for index_start in (range(0,len(stimulus_value),n)):
						index_end = index_start+n
						stimulus_data = np.array(stimulus_value[index_start:index_end])
						ut.debug(message=f"[Fixed Window ] File {stimulus_file} length({len(stimulus_value)}). Summarizing window[{index_start}:{index_end}]",show=show_debug_message)

						window_data.append(stimulus_data)

				np.save(os.path.join(path_destination,f"{file_name}.npy"), window_data)
	except Exception as error:
		ut.log_file(filename="log_file",header_message="dataset_stimulus_class: generate dataset with stimulus_name_stage, stimulus_value, applied_stimulus")



#MARK: Summarize
def summarize(path_destination,path_origin,event_basefile_therm,show_debug_message,dataset_structure):
	"""
		Description: 

		Arguments:
			path_origin(string): ;
			path_parent_root(string): ;
			sample_size(int): ;
	"""

	try:
		if not os.path.exists(path_destination): 
			os.makedirs(path_destination)

			for window_file in os.listdir(path_origin):
				fix_window_data = []
				stimulus_stage = os.path.splitext(window_file)[0].replace(' ','_').lower()
				stimulus_data = np.load(os.path.join(path_origin,window_file))
				
				ut.debug(message=f"[Fixed Window Dataset] data origin: {path_origin}",show=show_debug_message)
				ut.debug(message=f"[Fixed Window Dataset] File: {window_file}",show=show_debug_message)

				if event_basefile_therm in stimulus_stage: stimulus_applied = 0
				else: stimulus_applied = 1

				for window in stimulus_data:
					#ut.debug(message=f"[Fixed Window Dataset] File {window_file} length({len(stimulus_value)}). Summarizing window[{index_start}:{index_end}]",show=show_debug_message)

					fix_window_data.append(
						[
							stimulus_stage
							,np.mean(window)
							,stats.iqr(window)
							,np.var(window)
							,np.std(window)
							,stats.skew(window)
							,stats.kurtosis(window)
							,stimulus_applied
						]
					)

				df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dataset_structure.items()})
				df = pd.DataFrame(fix_window_data,columns=df.columns.tolist())
				df.to_feather(os.path.join(path_destination,window_file+".feather"))
	except Exception as error:
		ut.log_file(filename="log_file",header_message="dataset_stimulus_class: generate dataset with stimulus_name_stage, stimulus_value, applied_stimulus")