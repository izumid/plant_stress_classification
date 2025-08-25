import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import md_logfile as lf

# MARK: Wrangling
def stimulus_category_class_reduced(path_destination,path_origin,unique_value,random_state,sample_size):
	"""
		Description: Unify each stimulus per class to his own single file;

		Arguments:
			path_origin(string): ;
			path_parent_root(string): ;
			unique_value(boolean): ;
			random_state(int): ;
			sample_size(int): ;		
	"""

	print("pre processing...")

	np.random.seed(random_state) #setting random seed globaly

	try:
		if not os.path.exists(path_destination): os.makedirs(path_destination)

		for folder in os.listdir(path_origin):
			if "lectrodes" not in folder:
				path_absolute = os.path.join(path_origin,folder)
				joined_data = [np.loadtxt(os.path.join(path_absolute,file)) for file in sorted(os.listdir(path_absolute))]
				joined_data = np.concatenate(joined_data)
				joined_data_size = len(joined_data)

				if unique_value: joined_data = np.unique(joined_data)

				if joined_data_size >= sample_size: 
					joined_data = np.random.choice(joined_data,size=sample_size,replace=False) 
				else:
					joined_data = np.random.choice(joined_data,size=joined_data_size,replace=False)

				print(joined_data.shape,folder)
				
				np.save(os.path.join(path_destination,f"{folder}.npy"), joined_data)

	except Exception as error:
		lf.log_file(filename="log_file",header_message="stimulus_category_class_reduced: txt files to numpy")


#def stimulus_class_splited(x_column_name,y_column_name,y_event_value,path_destination,path_origin,sample_size,random_state):
def dataset_stimulus_class(x_column_name,y_column_name,path_destination,path_origin,sample_size):
	"""
		Description: Unify all stimulus into Single Dataframe by class (applied or non apllied stimulus);

		Arguments:
			x_column_name(string): ;
			y_column_name(string): ;
			path_origin(string): ;
			path_parent_root(string): ;
			sample_size(int): ;
	"""
		
	dict_result = {"stimulus_stage": "category", x_column_name: float, y_column_name: int}
	df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dict_result.items()})

	try:
		# Evaluate if that block is working correctly
	
		for filename in sorted(os.listdir(path_origin)):
			#print(filename,"AAAAAAAAAAAAAAAAAAAAAAAA")
			name_stimulus = Path(filename).stem.replace(' ','_').lower()
			x_value = np.load(os.path.join(path_origin,filename))
			
			#print("AAAAAAAA",sample_size, f"original array: {len(original_array)}")
			#if unique_value: sample_size = len(original_array)
			#electo_values = np.random.choice(original_array, size=sample_size, replace=False)
			#print(f"electro values: {len(electo_values)}")

			if "before" in name_stimulus: applied_stimulus = 0
			else: applied_stimulus = 1
			
			#if len(original_array) < sample_size: sample_size = sample_size = len(original_array)

			temp_df = pd.DataFrame({
				#x_column_name: electo_values
				"stimulus_stage": np.repeat(name_stimulus, sample_size)
				,x_column_name: x_value
				,y_column_name: np.repeat(applied_stimulus, sample_size)
			})

			if not os.path.exists(path_destination): os.makedirs(path_destination)
			temp_df.to_feather(os.path.join(path_destination,f"{os.path.splitext(filename)[0].replace(' ','_').lower()}.feather"))
		
		
		"""
			df = pd.concat([df, temp_df], ignore_index=True)
		
		df_non_stimuled = df.query(f"{y_column_name} != {y_event_value}").copy()
		df_stimuled = df.query(f"{y_column_name} == {y_event_value}").copy()
	
		if not os.path.exists(path_destination): os.makedirs(path_destination)

		# Really needs to shuffle again?
		df_non_stimuled = df_non_stimuled.sample(frac=1, random_state=random_state)
		df_non_stimuled.to_feather(os.path.join(path_destination,"non_stimuled.feather"))
		
		df_stimuled = df_stimuled.sample(frac=1,random_state=random_state)
		df_stimuled.to_feather(os.path.join(path_destination,"stimuled.feather"))
		"""

	except Exception as error:
		lf.log_file(filename="log_file",header_message="dataset_stimulus_class: generate dataset with stimulus_name_stage, stimulus_value, applied_stimulus")


# MARK: Windowing
def windowing(list_dataframe,x_column_name,y_column_name,list_window,path_destination,summarize):
	"""
		Description:
			While get slices of samples summarized then with statisticial distribution measures.
			summarized by stimulus_stage and keep the label information

		Arguments:
			list_dataframe
			x_column_name
			y_column_name
			list_window
			path_destination
			summarize
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

	#filename = "dataset"

	for m,n in list_window:
		
		fix_window_data = []
		file_name = f"{str(m)}x{str(n)}"
		print(f"file name readed: {file_name}")
		
		for dataframe in list_dataframe:
			data = np.array(dataframe[x_column_name].copy())
			applied_stimulus = dataframe[y_column_name].iloc[0]

			for i in (range(0,len(data),n)):
				window = np.array(data[i:i+n])
				if summarize: 
					fix_window_data.append(
						[
							np.mean(window)
							,stats.iqr(window)
							,np.var(window)
							,np.std(window)
							,stats.skew(window)
							,stats.kurtosis(window)
							,applied_stimulus
	   					]
					)
				else: fix_window_data.append([window.tolist(),applied_stimulus])

		if summarize:
			df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in cols.items()})
			df = pd.DataFrame(fix_window_data,columns=df.columns.tolist())
		else: 
			df = pd.DataFrame(fix_window_data, columns=["eletric_variation_value","applied_stimulus"])
		
		if not os.path.exists(path_destination): os.makedirs(path_destination)
		df.to_feather(os.path.join(path_destination,file_name+".feather"))


# MARK: NEW
def windowing_new(dataset_structure,path_origin,sample_size,list_window,summarize,path_destination):
	"""
		Description: 

		Arguments:
			x_column_name(string): ;
			y_column_name(string): ;
			path_origin(string): ;
			path_parent_root(string): ;
			sample_size(int): ;
	"""

	cols = {
		"stimulus_stage": "category"
		,"mean": "float64"
		,"inter_quartile_range": "float64"
		,"variance": "float64"
		,"standard_deviation": "float64"
		,"skew": "float64"
		,"kustosis": "float64"
		,"applied_stimulus": "int64"
	}

		
	try:
		# Evaluate if that block is working correctly
	
		for filename in sorted(os.listdir(path_origin)):
			stimulus_name = Path(filename).stem.replace(' ','_').lower()
			x_value = np.load(os.path.join(path_origin,filename))
			
			if "before" in stimulus_name: stimulus_applied = 0
			else: stimulus_applied = 1
			
			#if len(original_array) < sample_size: sample_size = sample_size = len(original_array)

			df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dataset_structure.items()})
			value = [np.repeat(stimulus_name, sample_size), x_value, np.repeat(stimulus_applied, sample_size)]

			idx = 0
			for key in dataset_structure.keys():
				df[key] = value[idx]
				idx+=1
			
			for m,n in list_window:
				
				fix_window_data = []
				file_name = f"{str(m)}x{str(n)}"
				print(f"file name readed: {file_name}")
				

				for stimulus_file in os.listdir(path_origin):
					#stimulus_dataset = pd.read_feather(os.path.join(path_origin,stimulus_file))
					stimulus_stage = os.path.splitext(stimulus_file)[0].replace(' ','_').lower()
					stimulus_value = x_value
					stimulus_applied = stimulus_applied

					for i in (range(0,len(stimulus_value),n)):
						window = np.array(stimulus_value[i:i+n])
						if summarize: 
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
						else: fix_window_data.append([window.tolist(),stimulus_applied])

				if summarize:
					df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in cols.items()})
					df = pd.DataFrame(fix_window_data,columns=df.columns.tolist())
				else: 
					df = pd.DataFrame(fix_window_data, columns=["eletric_variation_value","applied_stimulus"])
				
				if not os.path.exists(path_destination): os.makedirs(path_destination)
				df.to_feather(os.path.join(path_destination,file_name+".feather"))
	except Exception as error:
		lf.log_file(filename="log_file",header_message="dataset_stimulus_class: generate dataset with stimulus_name_stage, stimulus_value, applied_stimulus")

