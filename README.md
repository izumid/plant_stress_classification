# Concepts
- The maximum data lenght equals to each stimuli and stage - normal(0) and stressed(1) - is 5.240.000;
- Each stimuli must collect data from original dataset, because volume of 5.240.000 limit collect 3648 observations of 2nd experiment; 
- Fixed windows are based on MxN shape, where __M__ the is window size and __N__ samples ammount;
- Given stimuli used are Cold (Cl), Low light (Lw) and Manitol Mn) a quimic agent to conduct the plant to osmotic stresse state;
- A window of [10, 176] will have 10 window of each stimuli and stage, e.g 10 results in  $$Cl(10 \cdot 2) + Ll(10 \cdot 2) + Mn(10 \cdot 2) = Observations(60)$$


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
- "show_debug_message"(boolean): ;
- "show_debug_message"(dictionary): ;


# Json Template Configurations

## 1st Experiment
- "sample_size": 5240000,
- "unique_sample_size":,
- "list_window": [[26200,200], [13100,400], [6550,800], [3275,1600], [1310,4000], [200,26200], [400,13100], [800,6550], [1600,3275], [4000,1310]],
- "pre_processing_data": true,
- "summarize": true,
- "unique_value": false,
- "balanced_sample": true ,

## 2nd Experiment
- "sample_size":,
- "unique_sample_size": 21888,
- "list_window": [],
- "pre_processing_data": true,
- "summarize": true,
- "unique_value": true,
- "balanced_sample": false,

## 3th Experiment []
- "sample_size":,
- "unique_sample_size": 21888,
- "list_window": [],
- "pre_processing_data": true,
- "summarize": true,
- "unique_value": true,
- "balanced_sample": true,

## 4th Experiment []
- "sample_size": 5240000,
- "unique_sample_size":,
- "list_window": [],
- "pre_processing_data": true,
- "summarize": true,
- "unique_value": false,
- "balanced_sample": true,


## Validated
- [] 1st Experimento
- [] 2nd Experimento
- [] 3th Experimento
- [] 4th Experimento