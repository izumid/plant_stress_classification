import os
import json

import pandas as pd

import md_util as ut
import md_wrangling as wr
import md_classify as cl


def try_cast(value,type=None):
	"""
		Description:

		Arguments:

	"""

	if type == float:
		try: return(float(value))
		except: return(value) 
	else:
		try: return(int(value))
		except: return(value) 


def read_config(path_absolute):
	"""
		Description: read json config file;

		Arguments:
			path_absolute(string): the complete path to file, in other words, "path + filaname.extension";
	"""

	try:
		with open(path_absolute, 'r') as f: data = json.load(f)
		return(data)
	except Exception as error:
		ut.log_file(filename="log_file",header_message="read_config")


def result_feather_read(path_destination,filename,filter_model=False,dummy=False):
	pd.set_option('display.max_columns', None)  # Display all columns
	pd.set_option('display.max_rows', None)     # Display all rows

	path_absolute = os.path.join(path_destination,filename+".feather")
	df = pd.read_feather(path_absolute)
	if dummy == False:  df = df.query(f"model != 'DUM'")
	if filter_model != False: df = df.query(f"model == '{filter_model}'")
	print(df)


# MARK: Main
def main():
	config = read_config(os.path.join(os.getcwd(),"config/config.json"))

	path_original_data = os.path.join(os.path.dirname(os.getcwd()),"original_data")
	path_stimulus_data_joined = os.path.join(os.getcwd(),r"data\00_stimulus_data_joined")
	path_experiment_data =  os.path.join(os.getcwd(),r"data\01_experiment_data")
	path_fixed_window_dataset = os.path.join(os.getcwd(),r"data\2.0_fixed_window")	
	path_fixed_window_summarized_dataset = os.path.join(os.getcwd(),r"data\2.1_summarized_dataset")	
	path_result = os.path.join(os.getcwd(),r"data\03_experiment_result")

	unique_value=config["unique_value"]
	balanced_sample = config["balanced_sample"]
	sample_size = config["sample_size"]
	observation_size = config["x_size_category"] * config["y_size_class"]
	random_state = config["random_state"]
	show_debug_message = config["show_debug_message"]
	test = config["test"]
	k_fold_split = config["k_fold_split"]


	if unique_value:
		if balanced_sample:
			print("3th Experiment")
			unique_data_sample_length = int(config["unique_sample_size"] / observation_size)  #stimuli number (3) * classes: event & non event (2) = 6
			window = wr.window_shape(sample_class_size=unique_data_sample_length,test=test,show_debug_message=show_debug_message)
			
			path_experiment_data = os.path.join(path_experiment_data,r"02_value_unique\01_balanced")
			path_fixed_window_dataset = os.path.join(path_fixed_window_dataset, r"02_value_unique\01_balanced")
			path_fixed_window_summarized_dataset = os.path.join(path_fixed_window_summarized_dataset, r"02_value_unique\01_balanced")
			path_result = os.path.join(path_result, r"02_value_unique\01_balanced")
		else:
			print("2st Experiment")
			unique_data_sample_length = config["unique_imbalanced_sample_length"]
			unique_sample_size = sum(unique_data_sample_length.values())
			window = wr.window_shape(sample_class_size=unique_sample_size,sample_unique_length=unique_data_sample_length,balanced=False,test=test,show_debug_message=show_debug_message)

			path_experiment_data = os.path.join(path_experiment_data,r"02_value_unique\02_imbalanced")
			path_fixed_window_dataset = os.path.join(path_fixed_window_dataset, r"02_value_unique\02_imbalanced")
			path_fixed_window_summarized_dataset = os.path.join(path_fixed_window_summarized_dataset, r"02_value_unique\02_imbalanced")
			path_result = os.path.join(path_result, r"02_value_unique\02_imbalanced")
	else:
		path_experiment_data = os.path.join(path_experiment_data,r"01_value_duplicated\01_balanced")
		path_fixed_window_dataset = os.path.join(path_fixed_window_dataset, r"01_value_duplicated\01_balanced")
		path_fixed_window_summarized_dataset = os.path.join(path_fixed_window_summarized_dataset, r"01_value_duplicated\01_balanced")

		if k_fold_split > 0:
			print("4st Experiment")
			unique_data_sample_length = None
			window = wr.window_shape(sample_class_size=config["sample_size"],test=test,show_debug_message=show_debug_message)
		
			path_result = os.path.join(path_result, r"01_value_duplicated\01_balanced\02_k_fold")
		else:
			print("1st Experiment")
			window = config["window"]
			path_result = os.path.join(path_result, r"01_value_duplicated\01_balanced\01_train_test")
			
	if int(input("Type 1 to show windows list: ")):
		print(f"window structure: {window}")
		key_first_values = next(iter(window.values()))
		
		print(f"\r\nwindows[{len(window)}][{len(key_first_values)}]")
		for k,v in window.items():
			print(f"{k}: {v}\r\n")
	
	#MARK: Pre Processing
	if config["pre_processing_data"]:
		wr.join_stimulus_data(
			path_origin=path_original_data
			,path_destination=path_stimulus_data_joined
			,random_state=random_state
			,show_debug_message=show_debug_message
		)
		
		wr.experiment_data(
			path_origin=path_stimulus_data_joined
			,path_destination=path_experiment_data
			,sample_size=sample_size
			,unique_value=unique_value
			,balanced_sample=balanced_sample
			,unique_data_sample_length=unique_data_sample_length
			,show_debug_message=show_debug_message
		)

		wr.fixed_window_dataset(
			path_origin=path_experiment_data
			,path_destination_windowed = path_fixed_window_dataset
			,path_destination_summarized_window=path_fixed_window_summarized_dataset
			,dataset_structure=config["dataset_structure"]
			,window=window
			,unique_data=unique_value
			,balanced_sample=balanced_sample
			,event_basefile_therm=config["event_basefile_therm"]
			,show_debug_message=show_debug_message
			,verbose=config["verbose"]
		)

	#MARK: Classification
	if config["classify"]:
		list_classifier = config["list_classifier"]
		if k_fold_split == 0: ut.debug(message="Classification runnning: Train test split(80%-20%)",show=show_debug_message)
		else: ut.debug(message=f"Classification running K-fold: {k_fold_split}",show=show_debug_message)

		ut.debug(message="path_destination", var=path_result,show=show_debug_message)

		cl.classify(
			path_origin = path_fixed_window_summarized_dataset
			,path_destination = path_result
			,list_classifier = list_classifier
			,random_state = config["random_state"]
			,verbose = config["verbose"]
			,k_fold_split = k_fold_split
			,skip_file_exists = config["skip_file_exists"]
			,execution_sort_ascending = config["execution_sort_ascending"]
			,window=window
			,show_debug_message=show_debug_message
		)


if __name__ == "__main__":

	try: 
		main()

		while True:
			user_input = input('Type "Abnegation" to exit: ')
			if user_input == "Abnegation": break

	except Exception as e: 
		ut.log_file(filename="log_file",header_message="Error in Main!")