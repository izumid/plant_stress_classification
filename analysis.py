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
	path_analyses = os.path.join(os.getcwd(),r"data\04_analyses")

	path_valid_window = os.path.join(path_analyses,"02_all_experiment_data_overfitting_premisses.feather")
	path_grouped_window = os.path.join(path_analyses,"04_grouped_window.feather")
	path_melted = os.path.join(path_analyses,"melted.feather")
	path_image = os.path.join(path_analyses,"image")

	data_analysis.join_result_data(path_experiment=path_experiment,dataset_structure=dataset_structure,path_destination=path_analyses)
	data_analysis.dataset_premisse(path_origin=path_analyses,path_destination=path_analyses)
	data_analysis.valid_window(path_absolute_origin=path_valid_window,path_absolute_destination=os.path.join(path_analyses,"03_valid_window.feather"))
	
	data_analysis.group_data(
		path_absolute_origin=path_valid_window
		,path_absolute_destination=path_grouped_window
		,column_drop=["model"]
		,group_by=["experiment","window","samples_summarized"]
		,sort_ascending=[True, False, True]
	)
	
	data_analysis.melt_data_to_chart(
		path_origin_absolute=path_grouped_window
		,path_destination_absolute=path_melted
		,column_select=["experiment","train_acc_ten_above","train_test_hundred","f1_hundred","invalid_window"]
	)
	
	data_analysis.chart_experiment_behaviour(path_origin_absolute=path_melted,path_destination=path_image)
	data_analysis.chart_window_valid_distribution(path_dataset=path_grouped_window,path_destination=path_image)
	data_analysis.chart_window_valid_f1score(path_dataset=path_grouped_window,path_destination=path_image)
	data_analysis.chart_outlier(path_absolute_dataframe=path_grouped_window,path_destination=path_image)
	
	inferenical_analysis.test_normal_distribution(path_dataset=path_grouped_window,path_destination=path_analyses)
	inferenical_analysis.test_iqr_outlier(path_absolute_dataframe=path_grouped_window,path_destination=path_analyses)
	inferenical_analysis.test_random_permutation(path_absolute_dataframe=path_grouped_window,n_perm=1000,alpha=0.05,path_destination=path_analyses)

	data_analysis.dataset_info(path_analyses)

if __name__ == '__main__':
	main()