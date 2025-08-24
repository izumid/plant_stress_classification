# Concepts
	- window: based on MxN configuration, where *M* the is window size and *N* samples ammount. 


# Json Parameters
	- "sample_size"(int): ;
	- "unique_sample_size(int)": ;
	- "_list_window"(list(list(tuple)) or empty list): ;
	- "pre_processing_data"(boolean): ;
	- "summarize"(boolean): ;
	- "unique_value"(boolean): ;
	- "balanced_sample"(boolean): ;
	- "classify"(boolean): ;
	- "list_classifier"(list): support models "DUM","DT","NB","KNN","XGB","RF","MLP";
	- "k_fold_split"(boolean): ;
	- "verbose"(boolean): ;
	- "random_state"(int): ;
	-"debug"(boolean): ;


# Json Template Configurations

## 1st Experiment
	"sample_size": 5240000
	,"unique_sample_size":
	,"list_window": [[26200,200], [13100,400], [6550,800], [3275,1600], [1310,4000], [200,26200], [400,13100], [800,6550], [1600,3275], [4000,1310]]
	,"pre_processing_data": true
	,"summarize": true
	,"unique_value": false
	,"balanced_sample": true


## 2nd Experiment
	"sample_size":
	,"unique_sample_size": 21888
	,"list_window": []
	,"pre_processing_data": true
	,"summarize": true
	,"unique_value": true
	,"balanced_sample": false


## 3th Experiment
	"sample_size":
	,"unique_sample_size": 21888
	,"list_window": []
	,"pre_processing_data": true
	,"summarize": true
	,"unique_value": true
	,"balanced_sample": true


## 4th Experiment
	"sample_size": 5240000
	,"unique_sample_size":
	,"list_window": []
	,"pre_processing_data": true
	,"summarize": true
	,"unique_value": false
	,"balanced_sample": true

