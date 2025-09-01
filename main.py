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
def main(config):
	path_parent_root = os.path.join(os.path.dirname(os.getcwd()),"original_data")
	path_original_data = os.path.join(os.path.dirname(os.getcwd()),"original_data")
	#path_root = os.path.join(os.getcwd(),"experiments_data",Path(os.path.realpath(__file__)).stem)
	path_root = os.path.join(os.getcwd(),r"data\custom")
	path_subsampled = os.path.join(os.getcwd(),r"data\00_subsampled_data")
	path_stimulus_data_joined = os.path.join(os.getcwd(),r"data\00_stimulus_data_joined")
	path_experiment_data =  os.path.join(os.getcwd(),r"data\01_experiment_data")
	path_fixed_window_dataset = os.path.join(os.getcwd(),r"data\02_fixed_window_dataset")	
	
	unique_value=config["unique_value"]
	balanced_sample = config["balanced_sample"]
	summarize = config["summarize"]
	sample_size = config["sample_size"]
	unique_sample_size = config["unique_sample_size"]
	observation_size = config["x_size_category"] * config["y_size_class"]
	
	random_state = config["random_state"]
	sample_unique_length = config["sample_unique_length"]
	show_debug_message = config["show_debug_message"]

	#wr.stimulus_subsampling(path_destination=path_subsampled,path_origin=path_parent_root,random_state=random_state,sample_size=sample_size)

	if unique_value:
		if balanced_sample: 
			unique_sample_size = int(unique_sample_size / observation_size)  #stimuli number (3) * classes: event & non event (2) = 6
			list_window = wr.window_shape(total_sample_size=unique_sample_size,show_debug_message=show_debug_message)
			path_root = os.path.join(os.path.join(os.getcwd(),r"data\custom\02_value_unique\01_balanced"))
			path_experiment_data = os.path.join(path_experiment_data,r"02_value_unique\01_balanced")
			path_fixed_window_dataset = os.path.join(path_fixed_window_dataset, r"02_value_unique\01_balanced")
		else:
			unique_sample_size = sum(sample_unique_length.values())
			print(unique_sample_size,"AAAAAAAAAA")
			list_window = wr.window_shape(total_sample_size=unique_sample_size,show_debug_message=show_debug_message)
			path_root = os.path.join(os.path.join(os.getcwd(),r"data\custom\02_value_unique\02_imbalanced"))
			path_experiment_data = os.path.join(path_experiment_data,r"02_value_unique\02_imbalanced")
			path_fixed_window_dataset = os.path.join(path_fixed_window_dataset, r"02_value_unique\02_imbalanced")
	else:
		list_window = wr.window_shape(total_sample_size=config["sample_size"],show_debug_message=show_debug_message)

		path_root = os.path.join(os.path.join(os.getcwd(),r"data\custom\01_value_duplicate\01_balanced"))
		path_experiment_data = os.path.join(path_experiment_data,r"01_value_duplicated\01_balanced")
		path_fixed_window_dataset = os.path.join(path_fixed_window_dataset, r"01_value_duplicate\01_balanced")
	
	
	
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
			#x_column_name = "electro_value"
			#,y_column_name = "applied_stimulus"
			path_origin=path_stimulus_data_joined
			,path_destination=path_experiment_data
			,sample_size=sample_size
			,unique_value=unique_value
			,balanced_sample=balanced_sample
			,sample_unique_length=sample_unique_length
			,show_debug_message=show_debug_message
		)

	
		wr.fixed_window_dataset(
			path_origin=path_experiment_data
			,path_destination=path_fixed_window_dataset
			,dataset_structure=config["dataset_structure"]
			,list_window=list_window
			,event_basefile_therm=config["event_basefile_therm"]
			,show_debug_message=show_debug_message
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

		#windowing(list_dataframe=[df_non_stimuled,df_stimuled],x_column_name=x_column_name,y_column_name=y_column_name,list_window=list_window,path_destination=path_base,summarize=summarize)
		

	if config["classify"]:
		list_classifier = config["list_classifier"]
		k_fold_split = config["k_fold_split"]

		if k_fold_split == 0: ut.debug(message="Classification runnning: Train test split(80%-20%)",show=show_debug_message)
		else: ut.debug(message=f"Classification running K-fold: {k_fold_split}",show=show_debug_message)

		ut.debug(message="path_destination", var=path_result,show=show_debug_message)

		cl.classify(
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
		# 		print(wr.window_shape(total_sample_size=config["sample_size"]))
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