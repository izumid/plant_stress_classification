import os

import pandas as pd
import numpy as np

import json 

from scipy.stats import shapiro
from statstests.tests import shapiro_francia

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import boxcox 


# MARK: Hyphoteses Test Message
def test_message(p_value,alpha,h0_greater):
	"""
		Description:
			Returns a message reporting the probability that the current data is consistent with the assumption of normality. 
		Arguments:
			p_value(float):
			alpha(float):
			h0_greater(boolean): Inverts the default hypothesis test messaging: 
				If p_value > alpha, outputs the “reject H₀” message; otherwise, outputs the “fail to reject H₀” message.
	"""

	if h0_greater:
		if p_value > alpha: 
			message = "No strong evidence against normality (probably is normally distributed)"
		else: 
			message = "High probability of non-normal distribution"
	else:
		if p_value > alpha: 
			message = "High probability of non-normal distribution"
		else: 
			message = "No strong evidence against normality (probably is normally distributed)"
			
	return message


# MARK: Norm. Dist. Test
def test_normal_distribution(data_series,path_absolute_destination,alpha=0.05):
	#df = pd.read_feather(path_dataset)
	result = {}
	#np_array = df["f1_score"]
	np_array = data_series
	np_array = np.array(np_array, dtype=float)
	
	statistic, p_value = shapiro(np_array)
	result["Shapiro-Wilk"] = {"Statistic": statistic, "p-value": p_value, "message":  test_message(p_value=p_value,alpha=alpha,h0_greater=True)}
	
	sf = shapiro_francia(np_array)
	message = test_message(p_value=sf["p-value"],alpha=alpha,h0_greater=True)
	sf["message"] = message
	result["Shapiro-Francia"] = sf
	
	with open(path_absolute_destination, "w", encoding="utf-8") as file_result:
		json.dump(result, file_result, ensure_ascii=False, indent=4)


# MARK: IQR Test
def test_iqr_outlier(path_absolute_dataframe,path_destination):
	"""
		Description:
			
		Arguments:
			
	"""
	df = pd.read_feather(path_absolute_dataframe)
	df.query("invalid_window == 0")
	numpy_array = np.array(df["f1_score"], dtype=float)
	result = {}

	Q1 = np.percentile(numpy_array, 25)
	Q3 = np.percentile(numpy_array, 75)
	IQR = Q3 - Q1
	upper_limit = Q3 + 1.5 * IQR
	outliers_iqr = numpy_array[numpy_array > upper_limit]
	outliers_iqr = outliers_iqr if len(outliers_iqr) >= 1  else None

	result["Q1"] = Q1
	result["Q3"] = Q3
	result["IQR"] = IQR
	result["upper_limit"] = upper_limit
	result["outliers"] = outliers_iqr

	with open(os.path.join(path_destination,"06_test_iqr_outlier.json"), "w", encoding="utf-8") as file_result:
		json.dump(result, file_result, ensure_ascii=False, indent=4)


def test_random_permutation_old(path_absolute_dataframe,n_perm,alpha,path_destination):
	"""
		Description:
			
		Arguments:
			
	"""

	df = pd.read_feather(path_absolute_dataframe)
	df.query("invalid_window == 0")

	df.sort_values(by="f1_score", inplace=True)
	window_name = np.array(df["window"], dtype=str)
	f1_score = np.array(df["f1_score"], dtype=float) #Already sorted by the previous dataframe sort
	f1_index_sorted_asc = np.argsort(f1_score)
	#equals = sorted_data==f1_score

	# print("sorted_data")
	# print(sorted_data)
	# print("\r\nf1_score")
	# print(f1_score)
	# print(f"same elements and order: {equals}")
	# print(f"total f1_score true: {np.sum(equals)}, f1_score length {len(equals)}")
	#print(f"valid f1 window average: {np.mean(f1_score)} \r\n F1 values: {f1_score}")

	permutated = np.array([np.random.permutation(f1_score) for _ in range(n_perm)])
	expected_value = np.mean(np.sort(permutated, axis=1), axis=0)

	p_value = []
	result={}

	for i in range(len(f1_score)):
		p = np.sum(permutated[:, i] >= f1_score[i]) / n_perm
		p_value.append(p)

	for i in range(len(f1_score)):
		if p_value[i] < alpha:
			idx = f1_index_sorted_asc[i]
			difference = f1_score[i] - expected_value[i]
			#print(window_name[i],{"f1_score": f1_score[i],"rank_position": idx, "difference": difference, "p-value": p_value[i]})
			result[window_name[i]] = {"f1_score": float(f1_score[i]),"rank_position": int(idx), "difference": float(difference), "p-value": float(p_value[i])}
		
	with open(os.path.join(path_destination,"test_random_permutation.json"), "w", encoding="utf-8") as file_result:
		json.dump(result, file_result, ensure_ascii=False, indent=4)


# MARK: Random Permutation Test
def test_random_permutation(path_absolute_dataframe,permutation_number,alpha,path_destination):
	"""
		Description:
			
		Arguments:
			
	"""

	df = pd.read_feather(path_absolute_dataframe)
	df.query("invalid_window == 0")
	df.sort_values(by="f1_score", inplace=True)
	experiment = np.array(df["experiment"], dtype=str)
	window_name = np.array(df["window"], dtype=str)
	f1_score = np.array(df["f1_score"], dtype=float) #Already sorted by the previous dataframe sort
	
	permutated = np.array([np.random.permutation(f1_score) for _ in range(permutation_number)])
	expected_value = np.mean(permutated, axis=0)
	p_value = []
	result={}

	for score in f1_score:
		p = np.sum(permutated >= score) / permutated.size
		p_value.append(p)

	for i, p_val in enumerate(p_value):
		if p_val < alpha:
			difference = f1_score[i] - expected_value[i]

			result.setdefault(f"{experiment[i].zfill(2)}_experiment_", {})[str(window_name[i])] = {
				"f1_score": float(f1_score[i]),
				"rank_position": int(i),
				"expected_value": float(expected_value[i]),
				"difference": float(difference),
				"p_value": float(p_val)
			}

	with open(os.path.join(path_destination,"test_random_permutation.json"), "w", encoding="utf-8") as file_result:
		json.dump(result, file_result, indent=4)


# MARK: Box-Cox Transformation
def boxcox_transformation(path_absolute_dataframe,path_destination):
	df = pd.read_feather(path_absolute_dataframe)
	df.query("invalid_window == 0")
	f1_series = df["f1_score"]
	
	normalized_series, lamb = boxcox(f1_series)

	plt.hist(f1_series, bins=8, color="skyblue", edgecolor="black")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.title("Histogram of Data")
	
	plt.savefig(os.path.join(path_destination, "7.1_original_data_series.svg"), format="svg", bbox_inches="tight")
	plt.close()

	plt.hist(normalized_series, bins=8, color="skyblue", edgecolor="black")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.title("Histogram of Data")

	plt.savefig(os.path.join(path_destination, "7.2_normalized_series.svg"), format="svg", bbox_inches="tight")
	plt.close()

	return(normalized_series,lamb)
