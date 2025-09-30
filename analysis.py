import os
import pandas as pd
from md_analysis import data as data_analysis
from md_analysis import inferencial as inferenical_analysis

def main():
	path_experiment = [
		r"data\3.2_experiment_result_new\01_value_duplicated\01_balanced\01_train_test"
		,r"data\3.2_experiment_result_new\01_value_duplicated\01_balanced\02_k_fold\another_model"
		,r"data\3.2_experiment_result_new\01_value_duplicated\01_balanced\02_k_fold\svm_partial"
		,r"data\3.2_experiment_result_new\01_value_duplicated\01_balanced\02_k_fold\dt"
		,r"data\3.2_experiment_result_new\02_value_unique\01_balanced"
		,r"data\3.2_experiment_result_new\02_value_unique\02_imbalanced"
	]


	#config = read_config(path_absolute=os.path.join(os.getcwd(),r"config\config.json"))

	dataset_structure = {
		"experiment": "int64"
		,"model": "category"
		,"window": "int64"
		,"samples_summarized": "int64"
		,"accuracy_train": "float64"
		,"accuracy_test": "float64"
		,"presicion": "float64"
		,"recall": "float64"
		,"f1_score": "float64"
	}
	type_aggregation = {
		"accuracy_train": "mean"
		,"accuracy_test": "mean"
		,"presicion": "mean"
		,"recall": "mean"
		,"f1_score": "mean"
		,"train_acc_ten_difference": "sum"
		,"train_test_hundred": "sum"
		,"f1_hundred": "sum"
		,"below_dummy": "sum"
		,"invalid_window": "sum"
	}

	new_label_name = {
		"train_acc_ten_difference": "Acurácia variação entre\ntreino e teste exceder 10%"
		,"train_test_hundred": "Acurácia 100%\nem treino e teste"
		,"f1_hundred": "F1-Score 100%" 
		,"below_dummy": "F1-Score demais modelos\nabaixo do Dummy" 
		,"valid_window": "Janelas válidas"
		,"invalid_window": "Janelas invalidas" 
		,"total": "Total"
	}

	label_order = {
		"train_acc_ten_difference": 1
		,"train_test_hundred": 2
		,"f1_hundred": 3
		,"below_dummy": 4
		,"valid_window": 5
		,"invalid_window": 6
		,"total": 7
	}

	column_select = list(label_order.keys())
	column_select.remove("total")
	column_select.insert(0,"experiment")

	#path_analyses = os.path.join(os.getcwd(),r"data\04_analyses")
	path_analyses = os.path.join(os.getcwd(),r"data\4.2_analyses_new")
	path_premisse_frequency = os.path.join(path_analyses,"02_premisse_frequency.feather")
	path_grouped_model = os.path.join(path_analyses,"04_grouped_model.feather")
	path_grouped_window = os.path.join(path_analyses,"05_grouped_window.feather")
	path_melted = os.path.join(path_analyses,"05_melted.feather")
	path_image = os.path.join(path_analyses,"image")
	path_premisse_boolean = os.path.join(path_analyses,"03_premisse_boolean.feather")

	if 1==0:
		temp_path_experiment = [
			r"data\03_experiment_result\01_value_duplicated\01_balanced\01_train_test"
			,r"data\03_experiment_result\01_value_duplicated\01_balanced\02_k_fold\another_model"
			,r"data\03_experiment_result\01_value_duplicated\01_balanced\02_k_fold\svm_partial"
			,r"data\03_experiment_result\02_value_unique\01_balanced"
			,r"data\03_experiment_result\02_value_unique\02_imbalanced"
		]

		data_analysis.filter_data(
			path_experiment=temp_path_experiment
			,column_name="model"
			,remove_label="DT"
			,destination_folder="3.2_experiment_result_new"
		)

	
	if 1 == 1:
		data_analysis.join_result_data(path_experiment=path_experiment,dataset_structure=dataset_structure,path_destination=path_analyses)
		data_analysis.dataset_premisse(path_origin=path_analyses,path_destination_absolute=path_premisse_frequency)
		data_analysis.premisse_boolean(path_absolute_origin=path_premisse_frequency,path_absolute_destination=path_premisse_boolean)

		data_analysis.group_model(
			path_absolute_origin=path_premisse_frequency
			,path_absolute_destination=path_grouped_model
			,group_by=["experiment","model","window","samples_summarized"]
			,type_aggregation=type_aggregation
			,sort_ascending=[True, True, False,True]
		)
		
		data_analysis.group_data(
			path_absolute_origin=path_grouped_model
			,path_absolute_destination=path_grouped_window
			,column_drop=["model"]
			,group_by=["experiment","window"]
			,type_aggregation=type_aggregation
		)

		data_analysis.melt_data(path_origin_absolute=path_premisse_boolean,path_destination_absolute=path_melted,label_order=label_order,column_select=column_select)
		
		data_analysis.chart_experiment_behaviour(path_origin_absolute=path_melted,path_destination=path_image,new_label_name=new_label_name,premisse=True)
		data_analysis.chart_experiment_behaviour(path_origin_absolute=path_melted,path_destination=path_image,new_label_name=new_label_name,premisse=False)
		data_analysis.chart_experiment_grid(path_origin_absolute=path_melted,path_destination=path_image,new_label_name=new_label_name)

		df = pd.read_feather(path_grouped_window)
		df.query("invalid_window == 0", inplace=True)
		f1_series = df["f1_score"].copy()

		data_analysis.chart_window_valid_distribution(data_series=f1_series,path_destination=path_image)
		data_analysis.chart_window_valid_f1score(path_dataset=path_grouped_window,path_destination=path_image)
		data_analysis.chart_window_valid_f1score(path_dataset=path_grouped_window,path_destination=path_image,single_chart=False)
		data_analysis.chart_outlier_grid(path_absolute_dataframe=path_grouped_window,path_destination=path_image)
		data_analysis.chart_outlier(data_series=f1_series,path_destination=path_image)
			
		inferenical_analysis.test_normal_distribution(data_series=f1_series,path_absolute_destination=os.path.join(path_analyses,"test_normal_distribution.json"))
		inferenical_analysis.test_iqr_outlier(path_absolute_dataframe=path_grouped_window,path_destination=path_analyses)
		inferenical_analysis.test_random_permutation(path_absolute_dataframe=path_grouped_window,permutation_number=10_000,alpha=0.05,path_destination=path_analyses,random_state=42)

		normalized_series,lamb = inferenical_analysis.boxcox_transformation(path_grouped_window,path_destination=path_image)
		inferenical_analysis.test_normal_distribution(data_series=normalized_series,path_absolute_destination=os.path.join(path_analyses,"7.3_boxcox_transformed_data_gaussian_test_dist.json"))

		data_analysis.dataset_info(path_analyses)
if __name__ == '__main__':
	main()