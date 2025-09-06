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


def window_approved(sample_class_size,window_size,window_size_limit,window_sample_size_limit):
	avoid_decimal = sample_class_size / window_size
	avoid_remainder = sample_class_size % window_size

	if avoid_decimal > 0 and avoid_remainder == 0: 
		window_sample_size = (sample_class_size / window_size)

		if window_size >= window_size_limit and window_sample_size >= window_sample_size_limit and window_size.is_integer() and window_sample_size.is_integer():
			return([int(window_size),int(window_sample_size)])
		

#print(window_check_valid(3648,114))
#print(type(window_check_valid(96980,114)))

# MARK: Window
def window_shape(sample_class_size,start=1,step=1,window_size_limit=10,window_sample_size_limit=10,reverse=True,sample_unique_length=None,balanced=True,show_debug_message=False):
	"""
		Description:

		Arguments:
			sample_class_size(int): ;
			start=(int): ;
			step=(int): ;
			window_size_limit(int): ;
			window_sample_size_limit(int): ;
			reverse(bool): ;
			sample_unique_length(int): ;
			balanced(bool): ;
			show_debug_message(bool): ;

		Note: if experiment is unbalanced must guarantee that always get the same number of observations to each windows, avoinding this: 
			[Fixed Window Dataset] File Cold after.npy length(3647). Summarizing window[0:9698];
		Arguments:
		
	"""
	if balanced:
		stimulus_window={}
		window = []
		for window_size in range(start,sample_class_size,step):
			window_size_possible = window_approved(sample_class_size=sample_class_size,window_size=window_size,window_size_limit=window_size_limit,window_sample_size_limit=window_sample_size_limit)
			if not window_size_possible is None: window.append(window_size_possible)

		window.sort(key=lambda x: x[0], reverse=reverse)
		
		if show_debug_message: window = [min(window, key=lambda x: x[0])]
		stimulus_window[sample_class_size] = window

	else:
		stimulus_window_indiviual = []
		unique_sample_value = list(set(sample_unique_length.values()))
		max_window_size = min(unique_sample_value)

		for stimulus_sample_size in unique_sample_value:
			sample_size = []
			for window_size_test in range(1,max_window_size,1):
				sample_size_possible = window_approved(sample_class_size=stimulus_sample_size,window_size=window_size_test,window_size_limit=10,window_sample_size_limit=10)
				if not sample_size_possible is None: sample_size.append(sample_size_possible)
			stimulus_window_indiviual.append((stimulus_sample_size,sample_size))

		window_each_sample_size = [item for sublist in stimulus_window_indiviual for item in sublist[1]]
		#window_each_sample_size.sort(reverse=True)
		#print("\r\nwindow_each_sample_size",len(window_each_sample_size), window_each_sample_size)

		#window[1] = sample, considering it to each window have the same size, avoiding bias summarizing large data in on stimulus and less in others
		window_sample_size = [window[1] for window in window_each_sample_size]
		stimulus_class_diferent = len(stimulus_window_indiviual)
		sample_size_valid = [item for item in set(window_sample_size) if window_sample_size.count(item) == stimulus_class_diferent]

		sample_size_valid.sort(reverse=True)
			
		stimulus_window = {}
		for stimulus_sample in unique_sample_value:
			#window size = stimuli_sample/sample_size
			#window = [[stimulus_sample/sample_size,sample_size] 
			window = []
			for sample_size in sample_size_valid: 
				sample_window_size  = stimulus_sample / sample_size
				if sample_window_size.is_integer(): window.append([int(sample_window_size),sample_size])

			if show_debug_message: window = [min(window, key=lambda x: x[0])]

			window.sort(reverse=True)
			stimulus_window[stimulus_sample] = window

	return(stimulus_window)


#MARK: Imbalance Search
def imbalanced_window_search():
	#samples = [3648,10971,33871] #0
	#samples = [3600,10000,30000] #4
	#samples = [3000,10000,30000] #3
	#samples = [3000,9000,30000] #7
	#samples = [3600,9000,30000] #7
	#samples = [3600,9600,28800] #8
	#samples = [3600,10800,32400] #10
	samples = [3600,9000,28800] #9
	
	work_all_sample_size = []

	for s in samples:
		aux = []
		for i in range(1,min(samples),1):
			#print(s, i)
			window = window_approved(s,i,window_size_limit=10,window_sample_size_limit=100)
			if not window is None: aux.append(window)

		work_all_sample_size.append((s,aux))
			
	
	for sample_size in work_all_sample_size:
		print(sample_size)

	window_each_sample_size = work_all_sample_size[0][1]
	window_each_sample_size.extend(work_all_sample_size[1][1])
	window_each_sample_size.extend(work_all_sample_size[2][1])
	window_each_sample_size.sort(reverse=True)

	window_size = [window[0] for window in window_each_sample_size]
	#print("\r\nsorted:",x)

	valid = []
	n = 3
	for item in set(window_size):  # Iterate through unique elements to avoid redundant checks
		if window_size.count(item) == n: valid.append(int(item))
	
	valid.sort(reverse=True)
	print(f"\r\nvalid windows({len(valid)}), data: {valid}")

#imbalanced_window_search() 


def unique_total_value():
	path_relative = r"data\00_stimulus_data_joined"
	path_origin = os.path.join(os.getcwd(), path_relative)

	for file in os.listdir(path_origin):
		stimulus_data = np.load(os.path.join(path_origin,file))
		stimulus_data = np.unique(stimulus_data)
		print(f"File: {file}, unique: {len(stimulus_data)}")

#unique_total_value()


#MARK: Win. Dataset
'''def fixed_window_dataset(path_destination,list_window,path_origin,event_basefile_therm,show_debug_message,dataset_structure):
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
		ut.log_file(filename="log_file",header_message="dataset_stimulus_class: generate dataset with stimulus_name_stage, stimulus_value, applied_stimulus")'''


def fixed_window_dataset(path_destination_windowed,path_destination_summarized_window,window,path_origin,event_basefile_therm,show_debug_message,dataset_structure):
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
		if not os.path.exists(path_destination_windowed): os.makedirs(path_destination_windowed)
		if not os.path.exists(path_destination_summarized_window): os.makedirs(path_destination_summarized_window)
		
		for value in window.values():
			for w in value:
				window_size = w[0]
				window_sample_size = w[1]
				fixed_window_data = []
				summarized_data = []
				file_name = f"{str(window_size)}x{str(window_sample_size)}"				
				ut.debug(message=f"[Fixed Window Dataset] data origin: {path_origin}",show=show_debug_message)
				ut.debug(message=f"[Fixed Window Dataset] File: {file_name}",show=show_debug_message)

				for stimulus_file in os.listdir(path_origin):
					#stimulus_dataset = pd.read_feather(os.path.join(path_origin,stimulus_file))
					stimulus_stage = os.path.splitext(stimulus_file)[0].replace(' ','_').lower()
					stimulus_value = np.load(os.path.join(path_origin,stimulus_file))
					
					if event_basefile_therm in stimulus_stage: stimulus_applied = 0
					else: stimulus_applied = 1

					for index_start in (range(0,len(stimulus_value),window_sample_size)):
						index_end = index_start + window_sample_size
						window = np.array(stimulus_value[index_start:index_end])
						ut.debug(message=f"[Fixed Window Dataset] File {stimulus_file} length({len(stimulus_value)}). Summarizing window[{index_start}:{index_end}]",show=show_debug_message)
						fixed_window_data.append(window)
						summarized_data.append(
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

				np.save(os.path.join(path_destination_windowed,f"{file_name}.npy"),fixed_window_data)
				df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dataset_structure.items()})
				df = pd.DataFrame(summarized_data,columns=df.columns.tolist())
				df.to_feather(os.path.join(path_destination_summarized_window,f"{file_name}.feather"))
	except Exception as error:
		ut.log_file(filename="log_file",header_message="dataset_stimulus_class: generate dataset with stimulus_name_stage, stimulus_value, applied_stimulus")


##########################################################
## 						WRANGLING						##
##########################################################

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
		if not os.path.exists(path_destination): os.makedirs(path_destination)

		for file in os.listdir(path_origin):
			stimulus_data = np.load(os.path.join(path_origin,file))
			#stimulus_data_size = len(stimulus_data)

			if unique_value: 
				stimulus_data = np.unique(stimulus_data)
				if balanced_sample: sample_size = sample_unique_length["Cold after"]
				else: sample_size = sample_unique_length[Path(file).stem]
			
			ut.debug(message=f"[Experiment Data] File: {file}, sample size: {sample_size}, data length: {len(stimulus_data)}",show=show_debug_message)
			stimulus_data = stimulus_data[:(sample_size)]
			
			#if stimulus_data_size >= sample_size: 
				#stimulus_data = np.random.choice(stimulus_data,size=sample_size,replace=False) 
			#else: stimulus_data = np.random.choice(stimulus_data,size=stimulus_data_size,replace=False)

			
			np.save(os.path.join(path_destination,file), stimulus_data)
	except Exception as error:
		ut.log_file(filename="log_file",header_message="Experiment Data")


#MARK: Split Window
def window_fixed(path_destination,window,path_origin,show_debug_message,dataset_structure):
	"""
		Description: 

		Arguments:
			path_origin(string): ;
			path_parent_root(string): ;
			sample_size(int): ;
	"""

	try:
		if not os.path.exists(path_destination): os.makedirs(path_destination)
			
		for value in window.values():
			for w in value:
				window_size = w[0]
				window_sample_size = w[1]

				window_data = []
				file_name = f"{str(window_size)}x{str(window_sample_size)}"				
				ut.debug(message=f"[Window Fixed] data origin: {path_origin}",show=show_debug_message)
				ut.debug(message=f"[Window Fixed] file: {file_name}",show=show_debug_message)

				for stimulus_file in os.listdir(path_origin):
					if "after" in stimulus_file: stimulus_applied = 0
					else: stimulus_applied = 1
					stimulus_value = np.load(os.path.join(path_origin,stimulus_file))

					for index_start in (range(0,len(stimulus_value),window_sample_size)):
						index_end = index_start + window_sample_size
						stimulus_data = [stimulus_value[index_start:index_end],stimulus_applied]
						print(stimulus_data)
						ut.debug(message=f"[Window Fixed] file {stimulus_file} length({len(stimulus_value)}). Summarizing window[{index_start}:{index_end}]. Window length({len(stimulus_data)})",show=show_debug_message)

						window_data.append(stimulus_data)

				np.save(os.path.join(path_destination,f"{file_name}.npy"), window_data)
	except Exception as error:
		ut.log_file(filename="log_file",header_message="window_fixed: split single array 1D into N subarrays (2D) of same dimensions")



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
		if not os.path.exists(path_destination):  os.makedirs(path_destination)

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
			df.to_feather(os.path.join(path_destination,Path(window_file).stem+".feather"))
	except Exception as error:
		ut.log_file(filename="log_file",header_message="summarize: read each file of 2D arrays, summarize then and generate a dataset")