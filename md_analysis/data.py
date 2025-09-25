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
def dataset_premisse(path_origin,path_destination_absolute):
	"""
		Description:
			
		Arguments:


	"""

	try:
		df = pd.read_feather(os.path.join(path_origin,"01_all_experiment_data.feather"))

		df["train_acc_ten_difference"] = ((df["accuracy_train"] >= df["accuracy_test"] * 1.10 ) | (df["accuracy_train"] <= df["accuracy_test"] * 0.9)).astype(int) #avoiding under and over fitting
		df["train_test_hundred"] = ((df["accuracy_train"] == 100) & (df["accuracy_test"] == 100)).astype(int)
		df["f1_hundred"] =  (df["f1_score"] == 100).astype(int)
		df["below_dummy"] = 0
		model_name = list(df["model"].unique())
		model_name.remove("DUM")

		for experiment in list(df["experiment"].unique()):
			for model in model_name:
				for window in list(df["window"].unique()):
					minimum_limit = df.query("experiment == @experiment and model == 'DUM' and window == @window")["f1_score"].max()	
					condition = (
						(df["experiment"] == experiment) &
						(df["model"] == model) &
						(df["window"] ==  window) &
						(df["f1_score"] <= minimum_limit)
					)
					df.loc[condition, "below_dummy"] = 1

		#df["valid_window"] = ((df["train_acc_ten_difference"] == 0) & (df["train_test_hundred"] == 0) & (df["f1_hundred"] == 0) & (df["below_dummy"] == 0)).astype(int)
		df["invalid_window"] = ((df["train_acc_ten_difference"] == 1) | (df["train_test_hundred"] == 1) | (df["f1_hundred"] == 1) | (df["below_dummy"] == 1)).astype(int)

		df.to_feather(path_destination_absolute)
	except Exception as error:
		ut.log_file(filename="log_file",header_message="[Window] premisse_dataset")


def premisse_boolean(path_absolute_origin,path_absolute_destination):
	"""
		Description:
			
		Arguments:


	"""

	df = pd.read_feather(path_absolute_origin)
	df = df.query("model != 'DUM'").copy()
	category_until = 3

	df.drop(columns=df.columns.tolist()[df.columns.get_loc("accuracy_train"):df.columns.get_loc("train_acc_ten_difference")], inplace=True)
	df.drop(columns=["model"], inplace=True)
	df = df.groupby(df.columns.to_list()[:category_until],as_index=False).sum()

	for col in (df.columns.to_list())[category_until:]:
		df[col] = df[col].apply(lambda x: 1 if x != 0 else x)

	df.insert(loc=len(df.columns) - 1, column="valid_window", value=df["invalid_window"].apply(lambda x: 1 if x == 0 else 0))

	df.to_feather(path_absolute_destination)


# MARK: Data Group
def group_data(path_absolute_origin,path_absolute_destination,column_drop,group_by,type_aggregation,sort_ascending):
	"""
		Description:
			
		Arguments:


	"""
	
	df = pd.read_feather(path_absolute_origin)
	df.query("model != 'DUM'", inplace=True)
	df.drop(columns=column_drop,inplace=True)
	df = df.groupby(group_by,as_index=False).agg(type_aggregation)
	df.reset_index(drop=True, inplace=True)
	df.sort_values(by=group_by,ascending=sort_ascending,inplace=True)
	df.rename(columns={"window": "window_size"}, inplace=True)

	window = df["window_size"].astype(str)+"x"+df["samples_summarized"].astype(str)
	df.insert(1,"window",window)
	
	df.to_feather(path_absolute_destination)


def melt_data(path_origin_absolute,label_order,path_destination_absolute,column_select):
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

	df["order"] = df["premisses"].map(label_order)

	df.rename(columns={"level_1": "premisses", 0: "quantity"},inplace=True)
	df.sort_values(["experiment", "order"], inplace=True)
	df.to_feather(path_destination_absolute)



# MARK: Exp. Behaviour
def chart_experiment_behaviour(path_origin_absolute,path_destination,new_label_name,premisse):
	"""
		Description:
			
		Arguments:


	"""

	if not os.path.exists(path_destination): os.makedirs(path_destination)

	df = pd.read_feather(path_origin_absolute)
	df["premisses"] = df['premisses'].map(new_label_name)
	fontsize_label = 14
	fontsize_tick_label = fontsize_label-3
	labelpad=15
	experiment_unique = df["experiment"].unique()
	
	for experiment in experiment_unique:

		df_filtered = df.query("experiment == @experiment").copy()
		premisses_unique = df_filtered["premisses"].unique()
		
		sns.set_style("darkgrid")
		alpha = 0.7


		pl_flare = sns.color_palette("flare",20)
		pl_viridis = sns.color_palette("viridis", 20)
		palette = []
		filename_chart = ""

		if premisse: 
			x_label = premisses_unique[:4]
			palette = [pl_flare[8],pl_flare[11],pl_flare[14],pl_flare[17]]
			filename_chart = f"{str(experiment).zfill(2)}_experiment_premisse.svg"
		else: 
			x_label = premisses_unique[4:]
			palette = [pl_viridis[8],pl_flare[14],pl_flare[19]]
			filename_chart = f"{str(experiment).zfill(2)}_experiment_total.svg"
		
		# [fig] The entire canvas — everything in the plot window, including all subplots, titles, legends, etc.
		# Control overall size, background color, save the whole figure;
		fig = plt.figure(figsize=(9, 4))


		# [ax] A single plotting area inside the figure — where the actual bars, lines, etc. are drawn.
		# Add labels, set limits, customize ticks, plot data.
		# directly way to create both fig, ax = plt.subplots(figsize=(8, 5))

		ax = sns.barplot(
			data=df_filtered[df_filtered["premisses"].isin(x_label)]
			,x="premisses"
			,y="quantity"
			,hue="premisses"
			,palette=palette
			,alpha=alpha
			,width=0.6
		)
		
		ax.set_ylabel("Quantidade de Janelas", fontsize=fontsize_label)
		ax.set_ylim(0, 100)

		ax.tick_params(axis='x', labelsize=fontsize_tick_label)
		ax.tick_params(axis='y', labelsize=fontsize_tick_label)

		if premisse: plt.xlabel("Frequência de descumprimento (premissas de sub/sobreajuste)", labelpad=labelpad, fontsize=fontsize_label)
		else:  plt.xlabel("Resultados Gerais", labelpad=labelpad, fontsize=fontsize_label)

		# when use hue each bar is stored in a separate ax.containers entry. Insert and change style of data label
		for container in ax.containers:
			ax.bar_label(container, label_type="edge", padding=3, fontsize=fontsize_tick_label)

		plt.savefig(os.path.join(path_destination, filename_chart),format="svg",bbox_inches="tight")
		plt.close()


# MARK: Exp. Behaviour Legend
def chart_experiment_behaviour_legend(path_origin_absolute,path_destination,new_label_name,premisse):
	"""
		Description:
			
		Arguments:


	"""

	if not os.path.exists(path_destination): os.makedirs(path_destination)

	df = pd.read_feather(path_origin_absolute)
	df["premisses"] = df['premisses'].map(new_label_name)

	experiment_unique = df["experiment"].unique()
	
	for experiment in experiment_unique:

		df_filtered = df.query("experiment == @experiment").copy()
		premisses_unique = df_filtered["premisses"].unique()
		
		sns.set_style("darkgrid")
		alpha = 0.7
		

		pl_flare = sns.color_palette("flare",20)
		pl_viridis = sns.color_palette("viridis", 20)
		palette = []
		filename_chart = ""

		if premisse: 
			x_label = premisses_unique[:4]
			palette = [pl_flare[8],pl_flare[11],pl_flare[14],pl_flare[17]]
			filename_chart = f"{str(experiment).zfill(2)}_experiment_premisse.svg"
		else: 
			x_label = premisses_unique[4:]
			palette = [pl_viridis[8],pl_flare[14],pl_flare[19]]
			filename_chart = f"{str(experiment).zfill(2)}_experiment_total.svg"

		g = sns.catplot(
			data=df_filtered[df_filtered["premisses"].isin(x_label)],
			kind="bar",
			col="experiment",
			col_wrap=2,
			x="experiment",
			y="quantity",
			sharex=False,
			hue="premisses",
			height=4.2,
			aspect=0.92,
			palette=palette,
			alpha=alpha
		)

		# Customize each subplot
		for ax in g.axes.flat:
			for bar in ax.patches:
				bar.set_width(bar.get_width() * 0.85)
			ax.set_title("")
			ax.set_ylabel("Quantidade de Janelas",fontsize=14)
			ax.set_ylim(0, 100)
			ax.set_xlabel(ax.get_xlabel(), fontsize=14)

			for container in ax.containers:
				ax.bar_label(container, fmt="%.0f", label_type="edge", padding=4, fontsize=11) #change that font size to increase data labels

			ax.tick_params(axis="x", labelsize=11)

		g.set(xticklabels=[])
		if premisse: plt.xlabel("Parâmetros de análise", labelpad=15)
		else:  plt.xlabel("Resultados Gerais", labelpad=15)
		g.figure.subplots_adjust(right=1.43) #adjust "chart" wight

		# Remove Seaborn’s default legend (doesn't handle styling well)
		g._legend.remove()

		# Create a new legend manually
		#unique_labels = df_filtered["premisses"].unique()
		legend_palette = sns.color_palette(palette, n_colors=len(x_label))

		legend_handles = [
			Patch(facecolor=legend_palette[i], edgecolor='black', label=str(label), alpha=alpha-0.15)
			for i, label in enumerate(x_label)
		]

		# Add custom legend
		#bbox_to_anchor=(0.62, 0.5)
		# 	0.965 shifts the box further right beyond the normal bounds (1.0 is fully right-aligned)
		#	0.5 centers it vertically
		legend = g.figure.legend(
			handles=legend_handles,
			title="Legenda",
			loc="center right",
			bbox_to_anchor=(1.01, 0.5), #[0] increase or decrease X position, [1] increase or decrease Y position
			frameon=True,
			title_fontsize=14,
			prop={'size': 14}
		)
		
		for text in legend.get_texts():
			text.set_fontsize(12)

		# Style it properly
		plt.draw()
		frame = legend.get_frame()
		#frame.set_facecolor("#eee")
		frame.set_edgecolor("#eee")
		frame.set_linewidth(1.5)

		#plt.gca().set_facecolor((0.90,0.93,0.93,0.6))

		plt.savefig(os.path.join(path_destination,filename_chart), format="svg")
		plt.close()


#MARK: Exp. Behaviour Grid
def chart_experiment_grid(path_origin_absolute, path_destination, new_label_name):

	if not os.path.exists(path_destination):
		os.makedirs(path_destination)

	df = pd.read_feather(path_origin_absolute)
	df["premisses"] = df['premisses'].map(new_label_name)

	experiment_unique = df["experiment"].unique()
	
	sns.set_style("darkgrid")
	alpha = 0.7

	pl_viridis = sns.color_palette("viridis", 20)
	pl_flare = sns.color_palette("flare",20)

	premisses_palette = [pl_flare[8],pl_flare[11],pl_flare[14],pl_flare[17]]
	total_chart_pallete  = [pl_viridis[8],pl_flare[14],pl_flare[19]]

	for experiment in experiment_unique:
		df_filtered = df.query("experiment == @experiment").copy()

		# Divide premissas em dois grupos
		premisses_unique = df_filtered["premisses"].unique()
		first_group = premisses_unique[:4]
		second_group = premisses_unique[4:]

		fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

		# --- Gráfico da esquerda ---
		sns.barplot(
			data=df_filtered[df_filtered["premisses"].isin(first_group)],
			x="premisses",
			y="quantity",
			hue="premisses",
			palette=premisses_palette,
			alpha=alpha,
			ax=axes[0]
		)
		axes[0].set_title("Premissas")
		axes[0].set_ylabel("Quantidade")
		axes[0].set_xlabel("Parâmetros de análise", labelpad=15)
		axes[0].set_ylim(0, 100)

		# Adiciona rótulos numéricos
		for container in axes[0].containers:
			axes[0].bar_label(container, fmt="%.0f", label_type="edge", padding=3, fontsize=9)

		# --- Gráfico da direita ---
		sns.barplot(
			data=df_filtered[df_filtered["premisses"].isin(second_group)],
			x="premisses",
			y="quantity",
			hue="premisses",
			palette=total_chart_pallete,
			alpha=alpha,
			ax=axes[1]
		)
		axes[1].set_title("Totais")
		axes[1].set_ylabel("Quantidade")
		axes[1].set_xlabel("Resultados Gerais", labelpad=15)
		axes[1].set_ylim(0, 100)

		# Adiciona rótulos numéricos
		for container in axes[1].containers:
			axes[1].bar_label(container, fmt="%.0f", label_type="edge", padding=3, fontsize=9)

		# --- Legenda personalizada ---
		"""
		palette = sns.color_palette("viridis", n_colors=len(premisses_unique))
		
		legend_handles = [
			Patch(facecolor=palette[i], edgecolor='black', label=str(label), alpha=alpha-0.15)
			for i, label in enumerate(premisses_unique)
		]
		fig.legend(
			handles=legend_handles,
			title="Premissas",
			loc="center right",
			bbox_to_anchor=(1.15, 0.5),
			frameon=True,
			fontsize=9,
			title_fontsize=12
		)
		"""
		legend_handles = [
			Patch(facecolor=cor, edgecolor='black', label=str(label), alpha=alpha-0.15)
			for cor, label in zip(premisses_palette + total_chart_pallete,
								list(first_group) + list(second_group))
		]

		fig.legend(
			handles=legend_handles,
			title="Premissas",
			loc="center right",
			bbox_to_anchor=(1.15, 0.5),
			frameon=True,
			fontsize=9,
			title_fontsize=12
		)

		fig.tight_layout()
		plt.savefig(
			os.path.join(path_destination, f"{str(experiment).zfill(2)}_experiment_grid.svg"),
			format="svg",
			bbox_inches="tight"
		)
		plt.close()


# MARK: Chart Distribution
def chart_window_valid_distribution(data_series,path_destination):
	"""
		Description:
			
		Arguments:


	"""
	if not os.path.exists(path_destination): os.makedirs(path_destination)

	#df = pd.read_feather(path_dataset)
	#f1_series = df["f1_score"]

	pl_viridis = sns.color_palette("viridis", 20)
	sns.palplot(pl_viridis)
	pl_flare = sns.color_palette("flare",20)
	sns.palplot(pl_flare)

	fontsize = 14
	alpha = 0.8
	labelpad =7
	fontsize_tick_label = fontsize-3

	sns.set_style("darkgrid")
	#'ax' is shortcut to "axes object"
	#ax = sns.histplot(np_array, bins = 20, kde=True,color=(0.143343, 0.522773, 0.556295))
	
	#plt.figure(figsize=(8, 5))
	plt.figure(figsize=(9, 4))
	ax = sns.histplot(data=data_series, bins = 20, color=pl_viridis[6], alpha=0.5, stat="density")
	ax = sns.kdeplot(data=data_series, color=pl_viridis[5], alpha=0.3,linewidth=1.5, clip=(None,99.5)) 

	#ax.lines[0].set_color(colors[-1])

	plt.xlabel("F1-Score", fontsize=fontsize, alpha=alpha,labelpad=labelpad)
	ax.tick_params(axis='x', labelsize=fontsize_tick_label)

	plt.ylabel("Quantidade de Janelas", fontsize=fontsize,alpha=alpha, labelpad=labelpad)
	ax.tick_params(axis='y', labelsize=fontsize_tick_label)

	#plt.xlim(70, 100)
	#plt.gca().set_facecolor((0.90,0.93,0.93,0.6)) 
	#plt.gca().set_facecolor((0.90,0.90,0.98,0.5)) 

	#plt.savefig(os.path.join(path_destination,"08_valid_window_f1_distribution.png"), dpi=300, format="png", bbox_inches="tight")
	plt.savefig(os.path.join(path_destination,"08_valid_window_f1_distribution.svg"), format="svg", bbox_inches="tight")
	
	plt.close()


def custom_formatter(x):
	return f'{x:.2f}'.replace('.', ',')


#MARK: Chart F1 Score Valid Window
def chart_window_valid_f1score(path_dataset,path_destination,single_chart=True):
	"""
		Description:
			
		Arguments:


	"""
	if not os.path.exists(path_destination): os.makedirs(path_destination)
	
	df = pd.read_feather(path_dataset)
	#df.query("invalid_window == 0 and f1_score > 70", inplace=True)
	df.query("invalid_window == 0", inplace=True)
	df.sort_values(by=["experiment","f1_score"],ascending=[True,False],inplace=True)
	fontsize_label = 14
	fontsize_tick_label = fontsize_label-3

	sns.set_style("darkgrid")
	if single_chart:
		path_absolute = os.path.join(path_destination,"valid_window_f1_score.svg")
		g = sns.catplot(
			data=df
			,x='f1_score'
			,y='window'
			,hue='window'
			,col='experiment'
			,kind='bar'
			,col_wrap=2
			,height=4
			,aspect=1.5
			,palette="viridis"
			,sharey=False
			,sharex=True
		)

		g.set_titles("Experimento: {col_name}")
		g.set_axis_labels("F1-Score", "Janela [MxN]", fontsize=fontsize_label)
		plt.tight_layout()

		for ax in g.axes.flatten():
			for container in ax.containers:
				ax.bar_label(container, labels=[custom_formatter(bar.get_width()) for bar in container],
							label_type='edge', padding=4, fontsize=fontsize_tick_label)
	
				ax.tick_params(axis='x', labelsize=fontsize_tick_label, labelbottom=True)  # force labels on
				ax.tick_params(axis='y', labelsize=fontsize_tick_label, labelbottom=True)  # force labels on

			#ax.tick_params(axis='x', labelsize=10)
			ax.set_xlim(left=60)

		plt.savefig(path_absolute,dpi='figure')
	else:
		experiment_unique = df["experiment"].unique()
		sns.set_style("darkgrid")
		alpha = 0.7
		pl_viridis = sns.color_palette("viridis")
		filename_chart = ""
		#plt.figure(figsize=(9, 4))
		height = 4

		for experiment in experiment_unique:
			filename_chart =  f"{str(experiment).zfill(2)}_experiment_valid_window_f1_score.svg"

			dataset = df.query("experiment == @experiment").copy()
			dataset.sort_values(by="f1_score",ascending=False)
			
			if len(dataset) > 10: height = 6

			g = sns.catplot(
				data=dataset
				,x='f1_score'
				,y='window'
				,hue='window'
				,kind='bar'
				,height=height
				,aspect=1.5
				,palette="viridis"
			)

			g.set_titles("Experimento: {experiment}")
			g.set_axis_labels("F1-Score", "Janela [MxN]", fontsize=fontsize_label)
			#g.figure.subplots_adjust(right=1.43)
			
			if len(dataset) < 10: g.figure.set_size_inches(9, 4)
			
			plt.tight_layout()

			for ax in g.axes.flatten():
				for container in ax.containers:
					ax.bar_label(container, labels=[custom_formatter(bar.get_width()) for bar in container],
								label_type='edge', padding=4, fontsize=fontsize_tick_label)

				ax.tick_params(axis='x', labelsize=fontsize_tick_label, labelbottom=True)  # force labels on
				ax.tick_params(axis='y', labelsize=fontsize_tick_label, labelbottom=True)  # force labels on
				ax.set_xlim(left=60)


			#plt.show()
			plt.subplots_adjust(wspace=0.3)
			plt.savefig(os.path.join(path_destination,filename_chart),dpi='figure')


# MARK: Chart Outlier Grid
def chart_outlier_grid(path_absolute_dataframe,path_destination):
	"""
		Description:
			
		Arguments:
			
	"""
	
	if not os.path.exists(path_destination): os.makedirs(path_destination)

	pl_viridis = sns.color_palette("viridis", 20)
	pl_flare = sns.color_palette("flare",20)
	df = pd.read_feather(path_absolute_dataframe)
	df.query("invalid_window == 0", inplace=True)
	experiment_valid_window_ammount = len(df["experiment"].unique())
	fontsize_label = 14
	fontsize_tick_label = fontsize_label-3
	alpha=0.8

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

	fig, axes = plt.subplots(1, experiment_valid_window_ammount, figsize=(9, 4), sharex=False)
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
			patch.set_alpha(alpha-0.3)  # Ensure partial opacity for better contrast
			
		ax.set_xticklabels([])
			
		ax.set_xlabel(f"Experimento: {experiment}", fontsize=fontsize_label, alpha=alpha)
		ax.tick_params(axis='x', labelsize=fontsize_tick_label)

		ax.set_ylabel("Média F1-Score", fontsize=fontsize_label, alpha=alpha, labelpad=9)
		ax.tick_params(axis='y', labelsize=fontsize_tick_label)
	

		# Ensure consistent facecolor across resized section
		#ax.set_facecolor((0.90, 0.93, 0.93, 0.6))
		
		ax_position+=1
		
	# Adjust layout and spacing
	fig.tight_layout()
	fig.subplots_adjust(top=0.9, hspace=0.4)  # Top margin & vertical spacing

	#plt.savefig(os.path.join(path_destination, "outliers_analyses.png"), dpi=300, format="png", bbox_inches="tight")
	plt.savefig(os.path.join(path_destination, "09_outliers_analyses_grid.svg"), format="svg", bbox_inches="tight")

	plt.close()


# MARK: Chart Outlier
def chart_outlier(data_series,path_destination):
	"""
		Description:
			
		Arguments:
			
	"""

	if not os.path.exists(path_destination): os.makedirs(path_destination)

	fontsize_label = 16
	fontsize_tick_label = fontsize_label-3
	alpha=0.8
	sns.set_style("darkgrid")

	fig = plt.figure(figsize=(9, 4))
	ax = sns.boxplot(x=data_series, patch_artist=True)

	for patch in ax.patches:
		patch.set_alpha(alpha-0.3)  # Ensure partial opacity for better contrast

	ax.set_xlabel("Série médias de F1-Score", fontsize=fontsize_label, alpha=alpha, labelpad=9)
	ax.tick_params(axis='x', labelsize=fontsize_tick_label)
	ax.set_ylabel(f"Dados Unificados", fontsize=fontsize_label, alpha=alpha)
	ax.tick_params(axis='y', labelsize=fontsize_tick_label)


	#fig.tight_layout()

	#plt.savefig(os.path.join(path_destination, "outliers_analyses.png"), dpi=300, format="png", bbox_inches="tight")
	plt.savefig(os.path.join(path_destination, "10_outliers_analyses.svg"), format="svg", bbox_inches="tight")

	plt.close()
