import os

import pandas as pd
import numpy as np

import json 

from scipy.stats import shapiro
from statstests.tests import shapiro_francia

import seaborn as sns
import matplotlib.pyplot as plt

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
def test_normal_distribution(path_dataset,path_destination,alpha=0.05):
	df = pd.read_feather(path_dataset)
	result = {}
	np_array = df["f1_score"]
	np_array = np.array(np_array, dtype=float)
	
	statistic, p_value = shapiro(np_array)
	result["Shapiro-Wilk"] = {"Statistic": statistic, "p-value": p_value, "message":  test_message(p_value=p_value,alpha=alpha,h0_greater=True)}
	
	sf = shapiro_francia(np_array)
	message = test_message(p_value=sf["p-value"],alpha=alpha,h0_greater=True)
	sf["message"] = message
	result["Shapiro-Francia"] = sf
	
	with open(os.path.join(path_destination,"test_normal_distribution.json"), "w", encoding="utf-8") as file_result:
		json.dump(result, file_result, ensure_ascii=False, indent=4)


def chart_outlier(path_absolute_dataframe,path_destination):
	"""
		Description:
			
		Arguments:
			
	"""

	pl_viridis = sns.color_palette("viridis", 20)
	pl_flare = sns.color_palette("flare",20)
	df = pd.read_feather(path_absolute_dataframe)
	df.query("invalid_window == 0", inplace=True)

	sns.set_style("darkgrid")
	# Sample category colors
	category_colors = {
		1: pl_flare[5],
		2: pl_flare[9],
		3: pl_viridis[8],
		4: pl_viridis[11],
	}

	# Determine number of rows (last row will have a single, full-width plot)
	#n_rows = (len(unique_experiments) - 1) // 2 + 1

	fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharex=False)
	axes = axes.flatten()
	unique_experiment = df["experiment"].unique()
	
	# Iterate through experiments and plot
	ax_position = 0
	for experiment in unique_experiment:
		subset = df[df["experiment"] == experiment]	
		ax = axes[ax_position]

		boxplot = sns.boxplot(x="experiment", y="f1_score", data=subset, ax=ax, patch_artist=True)
		for patch in boxplot.patches:
			patch.set_facecolor(category_colors[experiment])
			patch.set_alpha(0.5)  # Ensure partial opacity for better contrast
			
		ax.set_xticklabels([])
		#ax.set_xticklabels(["Média F1-Score"], fontsize=8)
		
		ax.set_xlabel(f"Experimento: {experiment}", fontsize=10, alpha=0.8)
		ax.set_ylabel("Média F1-Score", fontsize=10, alpha=0.8, labelpad=9)

		# Ensure consistent facecolor across resized section
		ax.set_facecolor((0.90, 0.93, 0.93, 0.6))
		
		ax_position+=1
		
	# Adjust layout and spacing
	fig.tight_layout()
	fig.subplots_adjust(top=0.9, hspace=0.4)  # Top margin & vertical spacing

	plt.savefig(os.path.join(path_destination, "outliers_analyses.png"), dpi=300, format="png", bbox_inches="tight")
	plt.savefig(os.path.join(path_destination, "outliers_analyses.svg"), format="svg", bbox_inches="tight")

	plt.close()

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

	with open(os.path.join(path_destination,"test_iqr_outlier.json"), "w", encoding="utf-8") as file_result:
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


def test_random_permutation(path_absolute_dataframe,n_perm,alpha,path_destination):
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


	permutated = np.array([np.random.permutation(f1_score) for _ in range(n_perm)])
	expected_value = np.mean(permutated, axis=0)

	p_value = []
	result={}
	
	#print(f"F1-Score values: {f1_score}")

	for score in f1_score:
		p = np.sum(permutated >= score) / permutated.size
		p_value.append(p)

	for i in range(len(f1_score)):
		if p_value[i] < alpha:
			idx = f1_index_sorted_asc[i]
			difference = f1_score[i] - expected_value[i]
			#print(window_name[i],{"f1_score": f1_score[i],"rank_position": idx, "difference": difference, "p-value": p_value[i]})
			result[window_name[i]] = {"f1_score": float(f1_score[i]),"rank_position": int(idx), "difference": float(difference), "p-value": float(p_value[i])}
	
	if 1==0:
		for i, p in enumerate(p_value):
			if p < alpha:
				diff = f1_score[i] - expected_value[i]
				print(f"Valor = {f1_score[i]:.3f} (posição {i}), "
					f"esperado = {expected_value[i]:.3f}, "
					f"diferença = {diff:.3f}, "
					f"p = {p:.4f}")

	with open(os.path.join(path_destination,"test_random_permutation.json"), "w", encoding="utf-8") as file_result:
		json.dump(result, file_result, ensure_ascii=False, indent=4)