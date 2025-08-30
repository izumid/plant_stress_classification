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


##########################################################
## 						WRANGLING						##
##########################################################

#MARK: Win. Shape
def window_shape(total_sample_size,start=1,step=1,reverse=True,show_debug_message=False):
	"""
		Description:

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
						ut.debug(message=f"[Fixed Window Dataset] Summarizing[{index_start}:{index_end}]",show=show_debug_message)

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