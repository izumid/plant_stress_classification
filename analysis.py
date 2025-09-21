import os
from md_analysis import data as data_analysis
from md_analysis import inferencial as inferenical_analysis

def main():
	path_experiment = [
		r"data\03_experiment_result\01_value_duplicated\01_balanced\01_train_test"
		,r"data\03_experiment_result\01_value_duplicated\01_balanced\02_k_fold\another_model"
		,r"data\03_experiment_result\01_value_duplicated\01_balanced\02_k_fold\svm_partial"
		,r"data\03_experiment_result\02_value_unique\01_balanced"
		,r"data\03_experiment_result\02_value_unique\02_imbalanced"
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
		,"train_acc_ten_above": "sum"
		,"train_test_hundred": "sum"
		,"f1_hundred": "sum"
		,"below_dummy": "sum"
		,"invalid_window": "sum"
	}

	new_label_name = {
		"train_acc_ten_above": "Treino Acima Teste"
		,"train_test_hundred": "Treino & Teste 100%"
		,"f1_hundred": "F1 100%" 
		,"below_dummy": "Abaixo Dummy" 
		,"valid_window": "Janelas Válidas"
		,"invalid_window": "Janelas Invalidas" 
		,"total": "Total"
	}

	label_order = {
		"train_acc_ten_above": 1
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

	path_analyses = os.path.join(os.getcwd(),r"data\04_analyses")

	path_premisse_frequency = os.path.join(path_analyses,"02_premisse_frequency.feather")
	path_grouped_window = os.path.join(path_analyses,"04_grouped_window.feather")
	path_melted = os.path.join(path_analyses,"05_melted.feather")
	path_image = os.path.join(path_analyses,"image")
	path_premisse_boolean = os.path.join(path_analyses,"03_premisse_boolean.feather")

	data_analysis.join_result_data(path_experiment=path_experiment,dataset_structure=dataset_structure,path_destination=path_analyses)
	data_analysis.dataset_premisse(path_origin=path_analyses,path_destination_absolute=path_premisse_frequency)
	data_analysis.premisse_boolean(path_absolute_origin=path_premisse_frequency,path_absolute_destination=path_premisse_boolean)
	
	data_analysis.melt_data(
		#path_origin_absolute=path_grouped_window
		path_origin_absolute=path_premisse_boolean
		,path_destination_absolute=path_melted
		,label_order=label_order
		,column_select=column_select
	)
	
	data_analysis.dataset_info(path_analyses)
	
	data_analysis.chart_experiment_behaviour(path_origin_absolute=path_melted,path_destination=path_image,new_label_name=new_label_name,premisse=True)
	data_analysis.chart_experiment_behaviour(path_origin_absolute=path_melted,path_destination=path_image,new_label_name=new_label_name,premisse=False)
	
	data_analysis.chart_experiment_grid(path_origin_absolute=path_melted,path_destination=path_image,new_label_name=new_label_name)

	data_analysis.group_data(
		path_absolute_origin=path_premisse_frequency
		,path_absolute_destination=path_grouped_window
		,column_drop=["model"]
		,group_by=["experiment","window","samples_summarized"]
		,type_aggregation=type_aggregation
		,sort_ascending=[True, False, True]
	)
	
	data_analysis.chart_window_valid_distribution(path_dataset=path_grouped_window,path_destination=path_image)
	data_analysis.chart_window_valid_f1score(path_dataset=path_grouped_window,path_destination=path_image)
	data_analysis.chart_outlier(path_absolute_dataframe=path_grouped_window,path_destination=path_image)
		
	inferenical_analysis.test_normal_distribution(path_dataset=path_grouped_window,path_destination=path_analyses)
	inferenical_analysis.test_iqr_outlier(path_absolute_dataframe=path_grouped_window,path_destination=path_analyses)
	inferenical_analysis.test_random_permutation(path_absolute_dataframe=path_grouped_window,n_perm=10_000,alpha=0.05,path_destination=path_analyses)


if __name__ == '__main__':
	main()