import os
import sys

import pandas as pd
import numpy as np

import json 
import csv

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


#from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP,ZeroInflatedPoisson
#from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import md_util as ut


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


def dataset_info(path_origin):
	"""
		Description:
			
		Arguments:


	"""

	for file in os.listdir(path_origin):
		abs_path = os.path.join(path_origin,file)

		if os.path.splitext(file)[1] == ".feather":
			#print(f"============ {file} ============")
			df = pd.read_feather(abs_path)
			df.to_csv(abs_path.replace(".feather",".csv"),quotechar='"',quoting=csv.QUOTE_ALL,index=False)
			#print(df.info(),"\r\n"*2,df,"\r\n")
			
			try:
				x  = df.applied_stimulus
				print("dataset length: ",len(x), "\r\n"*5)
			except: continue
	

# MARK: Data Gather
def read_add_column(path_absolute):
	"""
		Description:
			
		Arguments:


	"""
	try:
		dataframe = pd.read_feather(path_absolute)
		column_name = "experiment"
		idx_column = 0

		match path_absolute:
			case _ if "01_train_test" in path_absolute:
				dataframe.insert(idx_column,column_name,1)
			case _ if "02_k_fold" in path_absolute:
				dataframe.insert(idx_column,column_name,4)
			case _ if r"02_value_unique\01_balanced" in path_absolute:
				dataframe.insert(idx_column,column_name,3)
			case _ if r"02_value_unique\02_imbalanced" in path_absolute:
				dataframe.insert(idx_column,column_name,2)
			case _:	
				print("error")
		
		return(dataframe)
	except Exception as error:
		ut.log_file(filename="log_file",header_message="[Window] read_add_column")


#def join_result_data(path_result_data,config):
def join_result_data(path_experiment,dataset_structure,path_destination):
	"""
		Description:
			
		Arguments:


	"""
	
	try:
		result_set = []

		for location in path_experiment:
			folder_experiment = os.path.join(os.getcwd(),location)
			
			for file_window_result in os.listdir(folder_experiment):
				path_absolute = os.path.join(folder_experiment,file_window_result)
				dataframe = read_add_column(path_absolute)
				result_set.append(dataframe)

		df_final = pd.concat(result_set, ignore_index=True)		
		#print("=="*38,"\r\n",df_final.info(),df_final.head())

		for col, dtype in dataset_structure.items():
			df_final[col] = df_final[col].astype(dtype)

		df_final.sort_values(by=["experiment","window","model"],ascending=[True, False, True], inplace=True)
		
		if not os.path.exists(path_destination): os.makedirs(path_destination)

		df_final.reset_index(drop=True, inplace=True)
		df_final.to_feather(os.path.join(path_destination,"01_all_experiment_data.feather"))
	except Exception as error:
		ut.log_file(filename="log_file",header_message="[Window] read_add_column")


#MARK: Premisse
def dataset_premisse(path_origin,path_destination):
	"""
		Description:
			
		Arguments:


	"""

	try:
		dataframe = pd.read_feather(os.path.join(path_origin,"01_all_experiment_data.feather"))

		dataframe["train_acc_ten_above"] = (dataframe["accuracy_train"] >= dataframe["accuracy_test"] * 1.10 ).astype(int)
		#dataframe["train_acc_ten_above"] = (dataframe["accuracy_train"] >= dataframe["accuracy_test"] * 1.10 and dataframe["accuracy_train"] <= dataframe["accuracy_test"] * 0.9).astype(int) #avoiding under and over fitting
		dataframe["train_test_hundred"] = ((dataframe["accuracy_train"] == 100) & (dataframe["accuracy_test"] == 100)).astype(int)
		dataframe["f1_hundred"] =  (dataframe["f1_score"] == 100).astype(int)
		dataframe["invalid_window"] = ((dataframe["train_acc_ten_above"] == 1) | (dataframe["train_test_hundred"] == 1) | (dataframe["f1_hundred"] == 1)).astype(int)
		
		dataframe.to_feather(os.path.join(path_destination,"02_all_experiment_data_overfitting_premisses.feather"))
	except Exception as error:
		ut.log_file(filename="log_file",header_message="[Window] premisse_dataset")


def valid_window(path_absolute_origin,path_absolute_destination):
	"""
		Description:
			
		Arguments:


	"""

	df = pd.read_feather(path_absolute_origin)
	df = df.query("model != 'DUM'").copy()

	df.drop(columns=df.columns.tolist()[df.columns.get_loc("accuracy_train"):df.columns.get_loc("train_acc_ten_above")], inplace=True)
	df.drop(columns=["model"], inplace=True)
	df = df.groupby(df.columns.to_list()[:3],as_index=False).sum()

	for col in (df.columns.to_list())[-4:]:
		df[col] = df[col].apply(lambda x: 1 if x != 0 else x)

	df.insert(
		loc=6,
		column="valid_window",
		value=df["invalid_window"].apply(lambda x: 1 if x == 0 else 0)
	)

	df.to_feather(path_absolute_destination)


# MARK: Data Group
def group_data(path_absolute_origin,path_absolute_destination,column_drop,group_by,sort_ascending):
	"""
		Description:
			
		Arguments:


	"""
	
	agg_dict = {
		"accuracy_train": "mean"
		,"accuracy_test": "mean"
		,"presicion": "mean"
		,"recall": "mean"
		,"f1_score": "mean"
		,"train_acc_ten_above": "sum"
		,"train_test_hundred": "sum"
		,"f1_hundred": "sum"
		,"invalid_window": "sum"
	}

	df = pd.read_feather(path_absolute_origin)
	df.drop(columns=column_drop,inplace=True)
	df = df.groupby(group_by,as_index=False).agg(agg_dict)
	df.reset_index(drop=True, inplace=True)
	df.sort_values(by=group_by,ascending=sort_ascending,inplace=True)
	df.rename(columns={"window": "window_size"}, inplace=True)

	window = df["window_size"].astype(str)+"x"+df["samples_summarized"].astype(str)
	df.insert(1,"window",window)
	
	df.to_feather(path_absolute_destination)


def melt_data_to_chart(path_origin_absolute,path_destination_absolute,column_select):
	"""
		Description:
			
		Arguments:


	"""
	df = pd.read_feather(path_origin_absolute)
	df = df[column_select]
	df["total"] = 1

	for col in (df.columns.to_list())[1:]:
		df[col] = df[col].apply(lambda x: 1 if x != 0 else x)

	df = df.groupby("experiment",as_index=True).sum()
	df.reset_index(inplace=True)
	
	df = df.melt(id_vars=["experiment"], var_name='premisses', value_name='quantity')

	category_to_order = {
		"train_acc_ten_above": 1
		,"train_test_hundred": 2
		,"f1_hundred": 3
		,"valid_window": 4 
		,"invalid_window": 5 
		,"total": 6
	}
	df["order"] = df["premisses"].map(category_to_order)


	df.rename(columns={"level_1": "premisses", 0: "quantity"},inplace=True)
	df.sort_values(["experiment", "order"], inplace=True)
	df.to_feather(path_destination_absolute)


# MARK: Chart Experiment Behaviour
def chart_experiment_behaviour(path_origin_absolute,path_destination):
	"""
		Description:
			
		Arguments:


	"""

	if not os.path.exists(path_destination): os.makedirs(path_destination)

	translate = {
		"train_acc_ten_above": "Treino Acima Teste"
		,"train_test_hundred": "Treino & Teste 100%"
		,"f1_hundred": "F1 100%" 
		,"valid_window": "Janelas Válidas"
		,"invalid_window": "Janelas Invalidas" 
		,"total": "Total"
	}
	df = pd.read_feather(path_origin_absolute)
	df["premisses"] = df['premisses'].map(translate)

	experiment_unique = df["experiment"].unique()

	for experiment in experiment_unique:

		df_filtered = df.query("experiment == @experiment").copy()

		sns.set_style("darkgrid")
		alpha = 0.7
		# Build plot
		g = sns.catplot(
			data=df_filtered,
			kind="bar",
			col="experiment",
			col_wrap=2,
			x="experiment",
			y="quantity",
			sharex=False,
			hue="premisses",
			height=4.2,
			aspect=0.9,
			palette="viridis",
			alpha=alpha
		)

		# Customize each subplot
		for ax in g.axes.flat:
			for bar in ax.patches:
				bar.set_width(bar.get_width() * 0.85)
			ax.set_title("")
			ax.set_ylabel("Quantidade")
			ax.set_ylim(0, 100)

			for container in ax.containers:
				ax.bar_label(container, fmt="%.0f", label_type="edge", padding=4, fontsize=9)

			ax.tick_params(axis="x", labelsize=9)

		g.set(xticklabels=[])
		plt.xlabel("Parâmetros de análise")
		g.figure.subplots_adjust(right=1.5)

		# Remove Seaborn’s default legend (doesn't handle styling well)
		g._legend.remove()

		# Create a new legend manually
		unique_labels = df_filtered["premisses"].unique()
		palette = sns.color_palette("viridis", n_colors=len(unique_labels))

		legend_handles = [
			Patch(facecolor=palette[i], edgecolor='black', label=str(label), alpha=alpha-0.15)
			for i, label in enumerate(unique_labels)
		]

		# Add custom legend
		#bbox_to_anchor=(0.62, 0.5)
		# 	0.965 shifts the box further right beyond the normal bounds (1.0 is fully right-aligned)
		#	0.5 centers it vertically
		legend = g.figure.legend(
			handles=legend_handles,
			title="Premissas",
			loc="center right",
			bbox_to_anchor=(0.965, 0.5), #shifts the box further right beyond the normal bounds (1.0 is fully right-aligned)
			frameon=True,
			fontsize=9,
			title_fontsize=12,
		)

		# Style it properly
		plt.draw()
		frame = legend.get_frame()
		frame.set_facecolor("#eee")
		frame.set_edgecolor("#ddd")
		frame.set_linewidth(1.5)

		plt.savefig(os.path.join(path_destination,f"{experiment}_experiment.svg"), format="svg")
		plt.close()


# MARK: Chart Distribution
def chart_window_valid_distribution(path_dataset,path_destination):
	"""
		Description:
			
		Arguments:


	"""
	if not os.path.exists(path_destination): os.makedirs(path_destination)

	df = pd.read_feather(path_dataset)

	np_array = df["f1_score"]
	np_array = np.array(np_array, dtype=float)

	pl_viridis = sns.color_palette("viridis", 20)
	sns.palplot(pl_viridis)
	pl_flare = sns.color_palette("flare",20)
	sns.palplot(pl_flare)

	fontsize = 11
	sns.set_style("darkgrid")
	#'ax' is shortcut to "axes object"
	#ax = sns.histplot(np_array, bins = 20, kde=True,color=(0.143343, 0.522773, 0.556295))
	plt.figure(figsize=(8, 5)) 
	ax = sns.histplot(np_array, color=pl_viridis[6], alpha=0.5, stat="density")  # Histograma
	ax = sns.kdeplot(np_array, color=pl_viridis[5], alpha=0.3,linewidth=1.5, clip=(None,99.5)) 

	#ax.lines[0].set_color(colors[-1])
	plt.xlabel("F1-Score", fontsize=fontsize, alpha=0.8,labelpad=7)
	plt.ylabel("Quantidade de Janelas", fontsize=fontsize,alpha=0.8, labelpad=7)

	#ax.set_xlabel("F1-Score", fontsize=fontsize)
	#ax.set_ylabel("Quantidade de Janelas", fontsize=fontsize)
	#plt.xlim(70, 100)
	plt.gca().set_facecolor((0.90,0.93,0.93,0.6)) 
	#plt.gca().set_facecolor((0.90,0.90,0.98,0.5)) 

	plt.savefig(os.path.join(path_destination,"valid_window_result_distribution.png"), dpi=300, format="png", bbox_inches="tight")
	plt.savefig(os.path.join(path_destination,"valid_window_result_distribution.svg"), format="svg", bbox_inches="tight")
	
	plt.close()


#MARK: Chart F1 Score Valid Window
def chart_window_valid_f1score(path_dataset,path_destination):
	"""
		Description:
			
		Arguments:


	"""
	if not os.path.exists(path_destination): os.makedirs(path_destination)
	
	df = pd.read_feather(path_dataset)
	df.query("invalid_window == 0 and f1_score > 70", inplace=True)
	df.sort_values(by=["experiment","f1_score"],ascending=[True,False],inplace=True)

	path_absolute = os.path.join(path_destination,"valid_window_f1_score.svg")
	sns.set_style("darkgrid")

	g = sns.catplot(
		data=df,
		x='f1_score', y='window',
		hue='window',
		col='experiment',
		kind='bar',
		col_wrap=2,
		height=4,
		aspect=1.5,
		palette="viridis",
		sharey=False
		#sharex=False
	)

	g.set_titles("Experimento: {col_name}")
	g.set_axis_labels("Janela [MxN]", "F1-Score")
	plt.tight_layout()

	for ax in g.axes.flatten():
		for container in ax.containers:
			# Custom formatting function
			def custom_formatter(x):
				return f'{x:.2f}'.replace('.', ',')

			ax.bar_label(container, labels=[custom_formatter(bar.get_width()) for bar in container],
						label_type='edge', padding=4, fontsize=9)

		ax.tick_params(axis='x', labelsize=9)
		ax.set_xlim(left=50)


	#plt.show()
	plt.savefig(path_absolute,dpi='figure')


# MARK: Chart Outlier
def chart_outlier(path_absolute_dataframe,path_destination):
	"""
		Description:
			
		Arguments:
			
	"""
	
	if not os.path.exists(path_destination): os.makedirs(path_destination)

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
