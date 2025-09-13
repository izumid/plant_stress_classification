import os
import json

import pandas as pd

import md_util as ut
import md_wrangling as wr
import md_classify as cl

# MARK: New Window Size



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
	path_parent_root = os.path.join(os.path.dirname(os.getcwd()),"original_data")
	path_original_data = os.path.join(os.path.dirname(os.getcwd()),"original_data")
	#path_root = os.path.join(os.getcwd(),"experiments_data",Path(os.path.realpath(__file__)).stem)
	path_root = os.path.join(os.getcwd(),r"data\custom")
	path_subsampled = os.path.join(os.getcwd(),r"data\00_subsampled_data")
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
	dataset_structure = config["dataset_structure"]
	test = config["test"]

	#wr.stimulus_subsampling(path_destination=path_subsampled,path_origin=path_parent_root,random_state=random_state,sample_size=sample_size)

	if unique_value:
		if balanced_sample:
			print("3th Experiment")
			unique_data_sample_length = int(config["unique_sample_size"] / observation_size)  #stimuli number (3) * classes: event & non event (2) = 6
			window = wr.window_shape(sample_class_size=unique_data_sample_length,test=test,show_debug_message=show_debug_message)
			
			path_root = os.path.join(os.path.join(os.getcwd(),r"data\custom\02_value_unique\01_balanced"))
			path_experiment_data = os.path.join(path_experiment_data,r"02_value_unique\01_balanced")
			path_fixed_window_dataset = os.path.join(path_fixed_window_dataset, r"02_value_unique\01_balanced")
			path_fixed_window_summarized_dataset = os.path.join(path_fixed_window_summarized_dataset, r"02_value_unique\01_balanced")
			path_result = os.path.join(path_result, r"02_value_unique\01_balanced")
		else:
			print("2st Experiment")
			unique_data_sample_length = config["unique_imbalanced_sample_length"]
			unique_sample_size = sum(unique_data_sample_length.values())
			window = wr.window_shape(sample_class_size=unique_sample_size,sample_unique_length=unique_data_sample_length,balanced=False,test=test,show_debug_message=show_debug_message)

			path_root = os.path.join(os.path.join(os.getcwd(),r"data\custom\02_value_unique\02_imbalanced"))
			path_experiment_data = os.path.join(path_experiment_data,r"02_value_unique\02_imbalanced")
			path_fixed_window_dataset = os.path.join(path_fixed_window_dataset, r"02_value_unique\02_imbalanced")
			path_fixed_window_summarized_dataset = os.path.join(path_fixed_window_summarized_dataset, r"02_value_unique\02_imbalanced")
			path_result = os.path.join(path_result, r"02_value_unique\02_imbalanced")
	else:
		print("1st Experiment")
		unique_data_sample_length = None
		window = wr.window_shape(sample_class_size=config["sample_size"],test=test,show_debug_message=show_debug_message)
		
		path_root = os.path.join(os.path.join(os.getcwd(),r"data\custom\01_value_duplicated\01_balanced"))
		path_experiment_data = os.path.join(path_experiment_data,r"01_value_duplicated\01_balanced")
		path_fixed_window_dataset = os.path.join(path_fixed_window_dataset, r"01_value_duplicated\01_balanced")
		path_fixed_window_summarized_dataset = os.path.join(path_fixed_window_summarized_dataset, r"01_value_duplicated\01_balanced")
		path_result = os.path.join(path_result, r"01_value_duplicated\01_balanced")
	
	#if int(input("Type 1 to show windows list: ")):
	
	if 1==1:
		print(f"window structure: {window}")
		key_first_values = next(iter(window.values()))
		
		print(f"\r\nwindows[{len(window)}][{len(key_first_values)}]")
		for k,v in window.items():
			print(f"{k}: {v}\r\n")
	
	# path_base = path_root.replace("custom","03_windowing_new")
	# #path_split = os.path.join(path_root,"04_split")
	# path_result = path_root.replace("custom","05_experiment_result")
	# pd.set_option('display.max_colwidth', None)
	
	
	# if unique_value:
	# 	list_window = wr.window_shape(total_sample_size=unique_sample_size,show_debug_message=show_debug_message)
	# 	if balanced_sample: sample_size = int(unique_sample_size / observation_size)  #stimuli number (3) * classes: event & non event (2) = 6
	# else: 
	# 	list_window = wr.window_shape(total_sample_size=config["sample_size"],show_debug_message=show_debug_message)
	# 	#sample_size = int(sample_size / observation_size) #testing use or dont using calc
		

	#if balanced_sample and unique_value:  sample_size = int((list_window[0][0] * list_window[0][1]) / 6)  #stimuli number (3) * classes: event & non event (2) = 6
	#else:sample_size = int(list_window[0][0] * list_window[0][1])


	# used process
	
	
	if config["pre_processing_data"]:
		path_stimulus_category_class_reduced=path_root.replace("custom","01_stimulus_category_class_reduced")
		path_stimulus_class_splited = path_root.replace("custom","02_dataset_stimulus_class")

		# wr.stimulus_category_class_reduced(
		# 	path_destination=path_stimulus_category_class_reduced
		# 	,path_origin=path_parent_root
		# 	,unique_value=unique_value
		# 	,random_state=random_state
		# 	,sample_size=sample_size
		# )


		# wr.dataset_stimulus_class(
		# 	x_column_name = "electro_value"
		# 	,y_column_name = "applied_stimulus"
		# 	#,y_event_value=config["y_event_value"]
		# 	,path_destination=path_stimulus_class_splited
		# 	,path_origin=path_stimulus_category_class_reduced
		# 	,sample_size=sample_size
		# 	#,random_state=random_state
		# )
		
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

		if 1 == 0:
			wr.window_fixed(
				path_origin=path_experiment_data
				,path_destination=path_fixed_window_dataset
				,window=window
				,dataset_structure=dataset_structure
				,show_debug_message=show_debug_message
			)
		
			wr.summarize(
				path_origin=path_fixed_window_dataset
				,path_destination=path_fixed_window_summarized_dataset
				,event_basefile_therm=config["event_basefile_therm"]
				,dataset_structure=dataset_structure
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

		# rawdata(
		# 	x_column_name=x_column_name
		# 	,y_column_name=y_column_name
		# 	,path_unified_resized=path_root.replace("custom","01_unified_resized")
		# 	,path_parent_root=path_parent_root
		# 	,unique_value=unique_value
		# 	,random_state=config["random_state"]
		# 	,sample_size=sample_size
		# 	,path_class_split=path_class_split	
		# )

		"""
		df_non_stimuled = pd.read_feather(os.path.join(path_stimulus_class_splited,"non_stimuled.feather"))
		df_stimuled = pd.read_feather(os.path.join(path_stimulus_class_splited,"stimuled.feather"))
		"""
		
	if config["classify"]:
		list_classifier = config["list_classifier"]
		k_fold_split = config["k_fold_split"]

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


	# if int(input("Unify separated txt base files: ")):
	# 	path_origin = os.path.join(os.getcwd(),"experiments_data",Path(os.path.realpath(__file__)).stem,"05_result")
	# 	path_destination = os.path.join(Path(path_origin).parent, "06_result_unified")
	# 	if not os.path.exists(path_destination): os.makedirs(path_destination)
	# 	mult_text_to_csv(path_origin=path_origin, path_destination=path_destination, all_has_title=True,filename=Path(os.path.realpath(__file__)).stem+"_stratified_kfold")


if __name__ == "__main__":

	try: 
		main()

		while True:
			user_input = input('Type "Abnegation" to exit: ')
			if user_input == "Abnegation": break

	except Exception as e: 
		ut.log_file(filename="log_file",header_message="Error in Main!")
