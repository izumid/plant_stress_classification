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


def test_random_permutation(path_absolute_dataframe,n_perm,alpha,path_destination):
	"""
		Description:
			
		Arguments:
			
	"""

	df = pd.read_feather(path_absolute_dataframe)
	df.query("invalid_window == 0")
	numpy_array = np.array(df["f1_score"], dtype=float)
	sorted_data = np.sort(numpy_array)
	sorted_index = np.argsort(numpy_array)
	permutated = np.array([np.random.permutation(numpy_array) for _ in range(n_perm)])
	#expected_value = np.mean(np.sort(permutated, axis=1), axis=0)
	p_value = []
	result={}

	for i in range(19):
		p = np.sum(permutated[:, i] >= sorted_data[i]) / n_perm
		p_value.append(p)

	for i in range(19):
		if p_value[i] < alpha:
			idx = sorted_index[i]
			#result[sorted_data[i]] = {"rank_position": idx, "p-value": p_value[i]}
			print(f"Value: {sorted_data[i]:.3f}, rank position: {idx}, p-value: {p_value[i]:.4f}")

	
	with open(os.path.join(path_destination,"test_random_permutation.json"), "w", encoding="utf-8") as file_result:
		json.dump(result, file_result, ensure_ascii=False, indent=4)

def main():
	path_analyses = os.path.join(os.getcwd(),r"data\04_analyses")
	path_grouped_window = os.path.join(path_analyses,"04_grouped_window.feather")
	path_image = os.path.join(path_analyses,"image")

if __name__ == "__main__": 
	main()