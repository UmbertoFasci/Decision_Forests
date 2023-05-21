# TensorFlow Decision Forests

![Python](https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=flat-square&logo=tensorflow&logoColor=white)
![Emacs](https://img.shields.io/badge/Emacs-7F5AB6.svg?style=flat-square&logo=gnuemacs&logoColor=white)
![Org](https://img.shields.io/badge/Org%20Mode-77AA99.svg?style=flat-square&logo=org&logoColor=white)

![Status](https://img.shields.io/badge/Status-Complete-88CE02.svg?style=flat-square)

<img src="https://github.com/UmbertoFasci/Decision_Forests/blob/main/Forest.jpeg" width=400 align="right" />



The following document will contain the basic instructions for creating a decision tree model with tensorflow. In this document I will:

1.  Train a binary classification Random Forest on a dataset containing numerical, categorical, and missing data.
2.  Evaluate the model on the test set.
3.  Prepare the model for TensorFlow Serving
4.  Examine the overall of the model and the importance of each feature.
5.  Re-train the model with a different learning algorithm (Gradient Boost Decision Trees).
6.  Use a different set of input features.
7.  Change the hyperparameters of the model.
8.  Preprocess the features.
9.  Train the model for regression.

***

- [Importing Libraries](#org2c18f3b)
- [Training a Random Forest model](#org6063c7e)
- [Evaluate the model](#org12a2516)
- [TensorFlow Serving](#org4cffe8b)
- [Model structure and feature importance](#orgb4ddae8)
- [Using make<sub>inspector</sub>](#org427d85d)
- [Model self evaluation](#org6137bd8)
- [Plotting the training logs](#orgbac479b)
- [Retrain model with different learning algorithm](#org3498bd1)
- [Using a subset of features](#org3ba03d9)
- [Hyper-parameters](#org2aabab3)
- [Feature Preprocessing](#org105f10b)
- [Training a regression model](#org9e65e84)
- [Conclusion](#orgbedf592)


<a id="org2c18f3b"></a>

# Importing Libraries

```python
import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
```

```python
print("Found TensorFlow Decision Forests v" + tfdf.__version__)
```

    Found TensorFlow Decision Forests v1.3.0


<a id="org6063c7e"></a>

# Training a Random Forest model

```python
# Download the dataset
!wget -q https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv -O /tmp/penguins.csv

# Load the dataset into Pandas DataFrame
dataset_df = pd.read_csv("/tmp/penguins.csv")

# Display the first 3 examples
dataset_df.head(3)
```

      species     island  bill_length_mm  bill_depth_mm  flipper_length_mm   
    0  Adelie  Torgersen            39.1           18.7              181.0  \
    1  Adelie  Torgersen            39.5           17.4              186.0   
    2  Adelie  Torgersen            40.3           18.0              195.0   
    
       body_mass_g     sex  year  
    0       3750.0    male  2007  
    1       3800.0  female  2007  
    2       3250.0  female  2007  

```python
label = "species"

classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)
```

    Label classes: ['Adelie', 'Gentoo', 'Chinstrap']

```python
def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))
```

    243 examples in training, 101 examples for testing.

```python
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)
```

    Metal device set to: Apple M1

```python
# Specify the model
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model
model_1.fit(train_ds)
```

```
Use 8 thread(s) for training
Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp9_im9_11 as temporary training directory
Reading training dataset...
Training tensor examples:
Features: {'island': <tf.Tensor 'data:0' shape=(None,) dtype=string>, 'bill_length_mm': <tf.Tensor 'data_1:0' shape=(None,) dtype=float64>, 'bill_depth_mm': <tf.Tensor 'data_2:0' shape=(None,) dtype=float64>, 'flipper_length_mm': <tf.Tensor 'data_3:0' shape=(None,) dtype=float64>, 'body_mass_g': <tf.Tensor 'data_4:0' shape=(None,) dtype=float64>, 'sex': <tf.Tensor 'data_5:0' shape=(None,) dtype=string>, 'year': <tf.Tensor 'data_6:0' shape=(None,) dtype=int64>}
Label: Tensor("data_7:0", shape=(None,), dtype=int64)
Weights: None
Normalized tensor features:
 {'island': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data:0' shape=(None,) dtype=string>), 'bill_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast:0' shape=(None,) dtype=float32>), 'bill_depth_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_1:0' shape=(None,) dtype=float32>), 'flipper_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_2:0' shape=(None,) dtype=float32>), 'body_mass_g': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_3:0' shape=(None,) dtype=float32>), 'sex': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data_5:0' shape=(None,) dtype=string>), 'year': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_4:0' shape=(None,) dtype=float32>)}
Training dataset read in 0:00:01.794301. Found 243 examples.
Training model...
Standard output detected as not visible to the user e.g. running in a notebook. Creating a training log redirection. If training gets stuck, try calling tfdf.keras.set_training_logs_redirection(False).
2023-05-21 18:08:01.281734: W tensorflow/tsl/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz

systemMemory: 8.00 GB
maxCacheSize: 2.67 GB
[INFO 23-05-21 18:08:01.3264 CDT kernel.cc:773] Start Yggdrasil model training
[INFO 23-05-21 18:08:01.3271 CDT kernel.cc:774] Collect training examples
[INFO 23-05-21 18:08:01.3272 CDT kernel.cc:787] Dataspec guide:
column_guides {
  column_name_pattern: "^__LABEL$"
  type: CATEGORICAL
  categorial {
    min_vocab_frequency: 0
    max_vocab_count: -1
  }
}
default_column_guide {
  categorial {
    max_vocab_count: 2000
  }
  discretized_numerical {
    maximum_num_bins: 255
  }
}
ignore_columns_without_guides: false
detect_numerical_as_discretized_numerical: false

[INFO 23-05-21 18:08:01.3275 CDT kernel.cc:393] Number of batches: 1
[INFO 23-05-21 18:08:01.3275 CDT kernel.cc:394] Number of examples: 243
[INFO 23-05-21 18:08:01.3276 CDT kernel.cc:794] Training dataset:
Number of records: 243
Number of columns: 8

Number of columns by type:
	NUMERICAL: 5 (62.5%)
	CATEGORICAL: 3 (37.5%)

Columns:

NUMERICAL: 5 (62.5%)
	1: "bill_depth_mm" NUMERICAL num-nas:2 (0.823045%) mean:17.2581 min:13.2 max:21.5 sd:1.93037
	2: "bill_length_mm" NUMERICAL num-nas:2 (0.823045%) mean:43.9954 min:33.1 max:59.6 sd:5.6213
	3: "body_mass_g" NUMERICAL num-nas:2 (0.823045%) mean:4180.71 min:2850 max:6300 sd:800.39
	4: "flipper_length_mm" NUMERICAL num-nas:2 (0.823045%) mean:200.51 min:172 max:230 sd:13.8075
	7: "year" NUMERICAL mean:2008.05 min:2007 max:2009 sd:0.827764

CATEGORICAL: 3 (37.5%)
	0: "__LABEL" CATEGORICAL integerized vocab-size:4 no-ood-item
	5: "island" CATEGORICAL has-dict vocab-size:4 zero-ood-items most-frequent:"Biscoe" 120 (49.3827%)
	6: "sex" CATEGORICAL num-nas:9 (3.7037%) has-dict vocab-size:3 zero-ood-items most-frequent:"male" 119 (50.8547%)

Terminology:
	nas: Number of non-available (i.e. missing) values.
	ood: Out of dictionary.
	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
	tokenized: The attribute value is obtained through tokenization.
	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
	vocab-size: Number of unique values.

[INFO 23-05-21 18:08:01.3276 CDT kernel.cc:810] Configure learner
[INFO 23-05-21 18:08:01.3277 CDT kernel.cc:824] Training config:
learner: "RANDOM_FOREST"
features: "^bill_depth_mm$"
features: "^bill_length_mm$"
features: "^body_mass_g$"
features: "^flipper_length_mm$"
features: "^island$"
features: "^sex$"
features: "^year$"
label: "^__LABEL$"
task: CLASSIFICATION
random_seed: 123456
metadata {
  framework: "TF Keras"
}
pure_serving_model: false
[yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
  num_trees: 300
  decision_tree {
    max_depth: 16
    min_examples: 5
    in_split_min_examples_check: true
    keep_non_leaf_label_distribution: true
    num_candidate_attributes: 0
    missing_value_policy: GLOBAL_IMPUTATION
    allow_na_conditions: false
    categorical_set_greedy_forward {
      sampling: 0.1
      max_num_items: -1
      min_item_frequency: 1
    }
    growing_strategy_local {
    }
    categorical {
      cart {
      }
    }
    axis_aligned_split {
    }
    internal {
      sorting_strategy: PRESORTED
    }
    uplift {
      min_examples_in_treatment: 5
      split_score: KULLBACK_LEIBLER
    }
  }
  winner_take_all_inference: true
  compute_oob_performances: true
  compute_oob_variable_importances: false
  num_oob_variable_importances_permutations: 1
  bootstrap_training_dataset: true
  bootstrap_size_ratio: 1
  adapt_bootstrap_size_ratio_for_maximum_training_duration: false
  sampling_with_replacement: true
}

[INFO 23-05-21 18:08:01.3278 CDT kernel.cc:827] Deployment config:
cache_path: "/var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp9_im9_11/working_cache"
num_threads: 8
try_resume_training: true

[INFO 23-05-21 18:08:01.3279 CDT kernel.cc:889] Train model
[INFO 23-05-21 18:08:01.3279 CDT random_forest.cc:416] Training random forest on 243 example(s) and 7 feature(s).
[INFO 23-05-21 18:08:01.3284 CDT random_forest.cc:805] Training of tree  1/300 (tree index:0) done accuracy:0.989796 logloss:0.367792
[INFO 23-05-21 18:08:01.3286 CDT random_forest.cc:805] Training of tree  11/300 (tree index:11) done accuracy:0.941667 logloss:0.946494
[INFO 23-05-21 18:08:01.3289 CDT random_forest.cc:805] Training of tree  22/300 (tree index:18) done accuracy:0.954733 logloss:0.380512
[INFO 23-05-21 18:08:01.3291 CDT random_forest.cc:805] Training of tree  32/300 (tree index:32) done accuracy:0.967078 logloss:0.376725
[INFO 23-05-21 18:08:01.3293 CDT random_forest.cc:805] Training of tree  42/300 (tree index:37) done accuracy:0.962963 logloss:0.22888
[INFO 23-05-21 18:08:01.3295 CDT random_forest.cc:805] Training of tree  52/300 (tree index:51) done accuracy:0.967078 logloss:0.226351
[INFO 23-05-21 18:08:01.3297 CDT random_forest.cc:805] Training of tree  62/300 (tree index:61) done accuracy:0.967078 logloss:0.0921431
[INFO 23-05-21 18:08:01.3300 CDT random_forest.cc:805] Training of tree  72/300 (tree index:73) done accuracy:0.967078 logloss:0.0966992
[INFO 23-05-21 18:08:01.3302 CDT random_forest.cc:805] Training of tree  82/300 (tree index:82) done accuracy:0.971193 logloss:0.0993379
[INFO 23-05-21 18:08:01.3305 CDT random_forest.cc:805] Training of tree  92/300 (tree index:89) done accuracy:0.967078 logloss:0.0991124
[INFO 23-05-21 18:08:01.3308 CDT random_forest.cc:805] Training of tree  106/300 (tree index:105) done accuracy:0.971193 logloss:0.097703
[INFO 23-05-21 18:08:01.3311 CDT random_forest.cc:805] Training of tree  116/300 (tree index:117) done accuracy:0.971193 logloss:0.0939756
[INFO 23-05-21 18:08:01.3314 CDT random_forest.cc:805] Training of tree  126/300 (tree index:126) done accuracy:0.967078 logloss:0.0927071
[INFO 23-05-21 18:08:01.3316 CDT random_forest.cc:805] Training of tree  136/300 (tree index:136) done accuracy:0.967078 logloss:0.0923017
[INFO 23-05-21 18:08:01.3319 CDT random_forest.cc:805] Training of tree  147/300 (tree index:147) done accuracy:0.967078 logloss:0.0919531
[INFO 23-05-21 18:08:01.3322 CDT random_forest.cc:805] Training of tree  157/300 (tree index:155) done accuracy:0.967078 logloss:0.0907799
[INFO 23-05-21 18:08:01.3324 CDT random_forest.cc:805] Training of tree  167/300 (tree index:165) done accuracy:0.967078 logloss:0.091866
[INFO 23-05-21 18:08:01.3326 CDT random_forest.cc:805] Training of tree  177/300 (tree index:176) done accuracy:0.967078 logloss:0.0921001
[INFO 23-05-21 18:08:01.3328 CDT random_forest.cc:805] Training of tree  188/300 (tree index:188) done accuracy:0.967078 logloss:0.0913054
[INFO 23-05-21 18:08:01.3330 CDT random_forest.cc:805] Training of tree  198/300 (tree index:197) done accuracy:0.967078 logloss:0.0914426
[INFO 23-05-21 18:08:01.3334 CDT random_forest.cc:805] Training of tree  208/300 (tree index:208) done accuracy:0.971193 logloss:0.0922682
[INFO 23-05-21 18:08:01.3336 CDT random_forest.cc:805] Training of tree  218/300 (tree index:215) done accuracy:0.971193 logloss:0.0923452
[INFO 23-05-21 18:08:01.3339 CDT random_forest.cc:805] Training of tree  228/300 (tree index:227) done accuracy:0.971193 logloss:0.0924834
[INFO 23-05-21 18:08:01.3341 CDT random_forest.cc:805] Training of tree  238/300 (tree index:236) done accuracy:0.971193 logloss:0.0926265
[INFO 23-05-21 18:08:01.3343 CDT random_forest.cc:805] Training of tree  250/300 (tree index:247) done accuracy:0.971193 logloss:0.093243
[INFO 23-05-21 18:08:01.3346 CDT random_forest.cc:805] Training of tree  260/300 (tree index:260) done accuracy:0.971193 logloss:0.0928761
[INFO 23-05-21 18:08:01.3349 CDT random_forest.cc:805] Training of tree  271/300 (tree index:267) done accuracy:0.975309 logloss:0.0923707
[INFO 23-05-21 18:08:01.3351 CDT random_forest.cc:805] Training of tree  281/300 (tree index:282) done accuracy:0.975309 logloss:0.0924931
[INFO 23-05-21 18:08:01.3353 CDT random_forest.cc:805] Training of tree  291/300 (tree index:290) done accuracy:0.975309 logloss:0.0928786
[INFO 23-05-21 18:08:01.3355 CDT random_forest.cc:805] Training of tree  300/300 (tree index:299) done accuracy:0.975309 logloss:0.0933225
[INFO 23-05-21 18:08:01.3356 CDT random_forest.cc:885] Final OOB metrics: accuracy:0.975309 logloss:0.0933225
[INFO 23-05-21 18:08:01.3358 CDT kernel.cc:926] Export model in log directory: /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp9_im9_11 with prefix 45128972badc466c
[INFO 23-05-21 18:08:01.3380 CDT kernel.cc:944] Save model in resources
[INFO 23-05-21 18:08:01.3400 CDT abstract_model.cc:849] Model self evaluation:
Number of predictions (without weights): 243
Number of predictions (with weights): 243
Task: CLASSIFICATION
Label: __LABEL

Accuracy: 0.975309  CI95[W][0.95185 0.989193]
LogLoss: : 0.0933225
ErrorRate: : 0.0246913

Default Accuracy: : 0.444444
Default LogLoss: : 1.05726
Default ErrorRate: : 0.555556

Confusion Table:
truth\prediction
   0    1   2   3
0  0    0   0   0
1  0  107   0   1
2  0    1  82   0
3  0    3   1  48
Total: 243

One vs other classes:
[INFO 23-05-21 18:08:01.3450 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp9_im9_11/model/ with prefix 45128972badc466c
[INFO 23-05-21 18:08:01.3507 CDT decision_forest.cc:660] Model loaded with 300 root(s), 4296 node(s), and 7 input feature(s).
[INFO 23-05-21 18:08:01.3507 CDT abstract_model.cc:1312] Engine "RandomForestGeneric" built
[INFO 23-05-21 18:08:01.3507 CDT kernel.cc:1074] Use fast generic engine
Model trained in 0:00:00.029421
Compiling model...
WARNING:tensorflow:AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x156fbcee0> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING:tensorflow:AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x156fbcee0> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING: AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x156fbcee0> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
Model compiled.
```

    <keras.callbacks.History at 0x1578aa430>


<a id="org12a2516"></a>

# Evaluate the model

```python
model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")
```

    1/1 [==============================] - 0s 177ms/step - loss: 0.0000e+00 - accuracy: 0.9802
    
    
    loss: 0.0000
    accuracy: 0.9802


<a id="org4cffe8b"></a>

# TensorFlow Serving

```python
model_1.save("/tmp/my_saved_model")
```

    WARNING:absl:Found untraced functions such as call_get_leaves while saving (showing 1 of 1). These functions will not be directly callable after loading.
    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets
    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets


<a id="orgb4ddae8"></a>

# Model structure and feature importance

```python
model_1.summary()
```

```
Model: "random_forest_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
=================================================================
Total params: 1
Trainable params: 0
Non-trainable params: 1
_________________________________________________________________
Type: "RANDOM_FOREST"
Task: CLASSIFICATION
Label: "__LABEL"

Input Features (7):
	bill_depth_mm
	bill_length_mm
	body_mass_g
	flipper_length_mm
	island
	sex
	year

No weights

Variable Importance: INV_MEAN_MIN_DEPTH:
    1.    "bill_length_mm"  0.448902 ################
    2. "flipper_length_mm"  0.436019 ###############
    3.     "bill_depth_mm"  0.312390 #####
    4.            "island"  0.311290 #####
    5.       "body_mass_g"  0.269659 ##
    6.               "sex"  0.242070 
    7.              "year"  0.240360 

Variable Importance: NUM_AS_ROOT:
    1. "flipper_length_mm" 145.000000 ################
    2.    "bill_length_mm" 93.000000 #########
    3.     "bill_depth_mm" 47.000000 ####
    4.            "island"  8.000000 
    5.       "body_mass_g"  7.000000 

Variable Importance: NUM_NODES:
    1.    "bill_length_mm" 674.000000 ################
    2.     "bill_depth_mm" 396.000000 #########
    3. "flipper_length_mm" 355.000000 ########
    4.            "island" 279.000000 ######
    5.       "body_mass_g" 253.000000 #####
    6.               "sex" 34.000000 
    7.              "year"  7.000000 

Variable Importance: SUM_SCORE:
    1.    "bill_length_mm" 27202.614958 ################
    2. "flipper_length_mm" 23134.120365 #############
    3.            "island" 10959.965564 ######
    4.     "bill_depth_mm" 9478.776921 #####
    5.       "body_mass_g" 2730.325157 #
    6.               "sex" 306.791798 
    7.              "year" 13.697044 



Winner takes all: true
Out-of-bag evaluation: accuracy:0.975309 logloss:0.0933225
Number of trees: 300
Total number of nodes: 4296

Number of nodes by tree:
Count: 300 Average: 14.32 StdDev: 2.72108
Min: 7 Max: 25 Ignored: 0
----------------------------------------------
[  7,  8)  2   0.67%   0.67%
[  8,  9)  0   0.00%   0.67%
[  9, 10)  9   3.00%   3.67% #
[ 10, 11)  0   0.00%   3.67%
[ 11, 12) 42  14.00%  17.67% #####
[ 12, 13)  0   0.00%  17.67%
[ 13, 14) 88  29.33%  47.00% ##########
[ 14, 15)  0   0.00%  47.00%
[ 15, 16) 92  30.67%  77.67% ##########
[ 16, 17)  0   0.00%  77.67%
[ 17, 18) 43  14.33%  92.00% #####
[ 18, 19)  0   0.00%  92.00%
[ 19, 20) 15   5.00%  97.00% ##
[ 20, 21)  0   0.00%  97.00%
[ 21, 22)  5   1.67%  98.67% #
[ 22, 23)  0   0.00%  98.67%
[ 23, 24)  3   1.00%  99.67%
[ 24, 25)  0   0.00%  99.67%
[ 25, 25]  1   0.33% 100.00%

Depth by leafs:
Count: 2298 Average: 3.22846 StdDev: 0.970644
Min: 1 Max: 7 Ignored: 0
----------------------------------------------
[ 1, 2)   9   0.39%   0.39%
[ 2, 3) 556  24.19%  24.59% ######
[ 3, 4) 888  38.64%  63.23% ##########
[ 4, 5) 628  27.33%  90.56% #######
[ 5, 6) 186   8.09%  98.65% ##
[ 6, 7)  25   1.09%  99.74%
[ 7, 7]   6   0.26% 100.00%

Number of training obs by leaf:
Count: 2298 Average: 31.7232 StdDev: 31.8759
Min: 5 Max: 120 Ignored: 0
----------------------------------------------
[   5,  10) 1122  48.83%  48.83% ##########
[  10,  16)  116   5.05%  53.87% #
[  16,  22)   75   3.26%  57.14% #
[  22,  28)   58   2.52%  59.66% #
[  28,  34)   61   2.65%  62.32% #
[  34,  39)   48   2.09%  64.40%
[  39,  45)   79   3.44%  67.84% #
[  45,  51)   82   3.57%  71.41% #
[  51,  57)   48   2.09%  73.50%
[  57,  63)   46   2.00%  75.50%
[  63,  68)   40   1.74%  77.24%
[  68,  74)   92   4.00%  81.24% #
[  74,  80)  156   6.79%  88.03% #
[  80,  86)  118   5.13%  93.17% #
[  86,  92)   57   2.48%  95.65% #
[  92,  97)   43   1.87%  97.52%
[  97, 103)   31   1.35%  98.87%
[ 103, 109)   16   0.70%  99.56%
[ 109, 115)    6   0.26%  99.83%
[ 115, 120]    4   0.17% 100.00%

Attribute in nodes:
	674 : bill_length_mm [NUMERICAL]
	396 : bill_depth_mm [NUMERICAL]
	355 : flipper_length_mm [NUMERICAL]
	279 : island [CATEGORICAL]
	253 : body_mass_g [NUMERICAL]
	34 : sex [CATEGORICAL]
	7 : year [NUMERICAL]

Attribute in nodes with depth <= 0:
	145 : flipper_length_mm [NUMERICAL]
	93 : bill_length_mm [NUMERICAL]
	47 : bill_depth_mm [NUMERICAL]
	8 : island [CATEGORICAL]
	7 : body_mass_g [NUMERICAL]

Attribute in nodes with depth <= 1:
	256 : bill_length_mm [NUMERICAL]
	240 : flipper_length_mm [NUMERICAL]
	180 : bill_depth_mm [NUMERICAL]
	142 : island [CATEGORICAL]
	73 : body_mass_g [NUMERICAL]

Attribute in nodes with depth <= 2:
	483 : bill_length_mm [NUMERICAL]
	314 : flipper_length_mm [NUMERICAL]
	312 : bill_depth_mm [NUMERICAL]
	239 : island [CATEGORICAL]
	161 : body_mass_g [NUMERICAL]
	6 : sex [CATEGORICAL]
	2 : year [NUMERICAL]

Attribute in nodes with depth <= 3:
	629 : bill_length_mm [NUMERICAL]
	372 : bill_depth_mm [NUMERICAL]
	343 : flipper_length_mm [NUMERICAL]
	272 : island [CATEGORICAL]
	232 : body_mass_g [NUMERICAL]
	29 : sex [CATEGORICAL]
	4 : year [NUMERICAL]

Attribute in nodes with depth <= 5:
	673 : bill_length_mm [NUMERICAL]
	396 : bill_depth_mm [NUMERICAL]
	354 : flipper_length_mm [NUMERICAL]
	279 : island [CATEGORICAL]
	252 : body_mass_g [NUMERICAL]
	34 : sex [CATEGORICAL]
	7 : year [NUMERICAL]

Condition type in nodes:
	1685 : HigherCondition
	313 : ContainsBitmapCondition
Condition type in nodes with depth <= 0:
	292 : HigherCondition
	8 : ContainsBitmapCondition
Condition type in nodes with depth <= 1:
	749 : HigherCondition
	142 : ContainsBitmapCondition
Condition type in nodes with depth <= 2:
	1272 : HigherCondition
	245 : ContainsBitmapCondition
Condition type in nodes with depth <= 3:
	1580 : HigherCondition
	301 : ContainsBitmapCondition
Condition type in nodes with depth <= 5:
	1682 : HigherCondition
	313 : ContainsBitmapCondition
Node format: NOT_SET

Training OOB:
	trees: 1, Out-of-bag evaluation: accuracy:0.989796 logloss:0.367792
	trees: 11, Out-of-bag evaluation: accuracy:0.941667 logloss:0.946494
	trees: 22, Out-of-bag evaluation: accuracy:0.954733 logloss:0.380512
	trees: 32, Out-of-bag evaluation: accuracy:0.967078 logloss:0.376725
	trees: 42, Out-of-bag evaluation: accuracy:0.962963 logloss:0.22888
	trees: 52, Out-of-bag evaluation: accuracy:0.967078 logloss:0.226351
	trees: 62, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0921431
	trees: 72, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0966992
	trees: 82, Out-of-bag evaluation: accuracy:0.971193 logloss:0.0993379
	trees: 92, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0991124
	trees: 106, Out-of-bag evaluation: accuracy:0.971193 logloss:0.097703
	trees: 116, Out-of-bag evaluation: accuracy:0.971193 logloss:0.0939756
	trees: 126, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0927071
	trees: 136, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0923017
	trees: 147, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0919531
	trees: 157, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0907799
	trees: 167, Out-of-bag evaluation: accuracy:0.967078 logloss:0.091866
	trees: 177, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0921001
	trees: 188, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0913054
	trees: 198, Out-of-bag evaluation: accuracy:0.967078 logloss:0.0914426
	trees: 208, Out-of-bag evaluation: accuracy:0.971193 logloss:0.0922682
	trees: 218, Out-of-bag evaluation: accuracy:0.971193 logloss:0.0923452
	trees: 228, Out-of-bag evaluation: accuracy:0.971193 logloss:0.0924834
	trees: 238, Out-of-bag evaluation: accuracy:0.971193 logloss:0.0926265
	trees: 250, Out-of-bag evaluation: accuracy:0.971193 logloss:0.093243
	trees: 260, Out-of-bag evaluation: accuracy:0.971193 logloss:0.0928761
	trees: 271, Out-of-bag evaluation: accuracy:0.975309 logloss:0.0923707
	trees: 281, Out-of-bag evaluation: accuracy:0.975309 logloss:0.0924931
	trees: 291, Out-of-bag evaluation: accuracy:0.975309 logloss:0.0928786
	trees: 300, Out-of-bag evaluation: accuracy:0.975309 logloss:0.0933225
```


<a id="org427d85d"></a>

# Using make<sub>inspector</sub>

```python
model_1.make_inspector().features()
```

    '("bill_depth_mm" (1; #1) 
     "bill_length_mm" (1; #2) 
     "body_mass_g" (1; #3) 
     "flipper_length_mm" (1; #4) 
     "island" (4; #5) 
     "sex" (4; #6) 
     "year" (1; #7))

```python
model_1.make_inspector().variable_importances()
```

```
'("NUM_AS_ROOT": (("flipper_length_mm" (1; #4)  145.0) 
  ("bill_length_mm" (1; #2)  93.0) 
  ("bill_depth_mm" (1; #1)  47.0) 
  ("island" (4; #5)  8.0) 
  ("body_mass_g" (1; #3)  7.0)) 
 "INV_MEAN_MIN_DEPTH": (("bill_length_mm" (1; #2)  0.4489024611371831) 
  ("flipper_length_mm" (1; #4)  0.4360192595096871) 
  ("bill_depth_mm" (1; #1)  0.3123900322532565) 
  ("island" (4; #5)  0.3112903191624733) 
  ("body_mass_g" (1; #3)  0.26965874160940095) 
  ("sex" (4; #6)  0.24206985281762014) 
  ("year" (1; #7)  0.24036022658293554)) 
 "SUM_SCORE": (("bill_length_mm" (1; #2)  27202.61495835334) 
  ("flipper_length_mm" (1; #4)  23134.120364524424) 
  ("island" (4; #5)  10959.965564111248) 
  ("bill_depth_mm" (1; #1)  9478.776920754462) 
  ("body_mass_g" (1; #3)  2730.325157149695) 
  ("sex" (4; #6)  306.7917976118624) 
  ("year" (1; #7)  13.697043802589178)) 
 "NUM_NODES": (("bill_length_mm" (1; #2)  674.0) 
  ("bill_depth_mm" (1; #1)  396.0) 
  ("flipper_length_mm" (1; #4)  355.0) 
  ("island" (4; #5)  279.0) 
  ("body_mass_g" (1; #3)  253.0) 
  ("sex" (4; #6)  34.0) 
  ("year" (1; #7)  7.0)))
```


<a id="org6137bd8"></a>

# Model self evaluation

```python
model_1.make_inspector().evaluation()
```

    Evaluation(num_examples=243, accuracy=0.9753086419753086, loss=0.09332247295344072, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)


<a id="orgbac479b"></a>

# Plotting the training logs

```python
model_1.make_inspector().training_logs()
```

|         |                                                                                                                                                                                 |         |                                                                                                                                                                                  |         |                                                                                                                                                                                   |         |                                                                                                                                                                                   |         |                                                                                                                                                                                   |         |                                                                                                                                                                                   |         |                                                                                                                                                                                   |         |                                                                                                                                                                                   |         |                                                                                                                                                                                   |         |                                                                                                                                                                                 |         |                                                                                                                                                                                    |         |                                                                                                                                                                                   |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                   |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                   |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |         |                                                                                                                                                                                    |
|-------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |-------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TrainLog | (num<sub>trees</sub>=1 evaluation=Evaluation (num<sub>examples</sub>=98 accuracy=0.9897959183673469 loss=0.36779236306949536 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=11 evaluation=Evaluation (num<sub>examples</sub>=240 accuracy=0.9416666666666667 loss=0.9464943699538708 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=22 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9547325102880658 loss=0.38051191389315414 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=32 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.37672530123848974 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=42 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9629629629629629 loss=0.22888033251458234 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=52 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.22635050861494532 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=62 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09214309737883478 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=72 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09669924490613702 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=82 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.09933792135904355 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=92 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.099112403736193 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=106 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.09770301330849958 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=116 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.0939756027986238 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=126 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09270708282083395 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=136 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09230172781296718 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=147 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09195313022046546 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=157 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09077992822606991 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=167 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09186600034849511 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=177 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09210005546670882 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=188 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09130537166900228 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=198 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9670781893004116 loss=0.09144260634672004 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=208 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.0922682062979527 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=218 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.09234516689009627 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=228 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.09248343468816192 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=238 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.09262648260857097 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=250 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.09324301321274091 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=260 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.0928760742684328 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=271 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.09237068120405507 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=281 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.09249307419108266 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=291 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.09287857651373241 rmse=None ndcg=None aucs=None auuc=None qini=None)) | TrainLog | (num<sub>trees</sub>=300 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.09332247295344072 rmse=None ndcg=None aucs=None auuc=None qini=None)) |

```python
import matplotlib.pyplot as plt

logs = model_1.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")

plt.show()
```

![img](./.ob-jupyter/8b245f59727761f8589999ae808dfcaf9dfd12da.png)


<a id="org3498bd1"></a>

# Retrain model with different learning algorithm

```python
tfdf.keras.get_all_models()
```

|                                                                        |                                                                                |                                                                |                                                                                           |
|----------------------------------------------------------------------- |------------------------------------------------------------------------------- |--------------------------------------------------------------- |------------------------------------------------------------------------------------------ |
| tensorflow<sub>decision</sub><sub>forests.keras.RandomForestModel</sub> | tensorflow<sub>decision</sub><sub>forests.keras.GradientBoostedTreesModel</sub> | tensorflow<sub>decision</sub><sub>forests.keras.CartModel</sub> | tensorflow<sub>decision</sub><sub>forests.keras.DistributedGradientBoostedTreesModel</sub> |


<a id="org3ba03d9"></a>

# Using a subset of features

```python
feature_1 = tfdf.keras.FeatureUsage(name="bill_length_mm")
feature_2 = tfdf.keras.FeatureUsage(name="island")

all_features = [feature_1, feature_2]

# This model is only being trained on two features.
# It will NOT be as good as the previous model trained on all features.

model_2 = tfdf.keras.GradientBoostedTreesModel(
    features=all_features, exclude_non_specified_features=True)

model_2.compile(metrics=["accuracy"])
model_2.fit(train_ds, validation_data=test_ds)

print(model_2.evaluate(test_ds, return_dict=True))
```

```
Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp338u5rqh as temporary training directory
Reading training dataset...
Training dataset read in 0:00:00.075797. Found 243 examples.
Reading validation dataset...
Num validation examples: tf.Tensor(101, shape=(), dtype=int32)
Validation dataset read in 0:00:00.086017. Found 101 examples.
Training model...
[WARNING 23-05-21 18:08:04.4900 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:04.4900 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:04.4900 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
Model trained in 0:00:00.060420
Compiling model...
Model compiled.
1/1 [==============================] - 0s 48ms/step - loss: 0.0000e+00 - accuracy: 0.9208
{'loss': 0.0, 'accuracy': 0.9207921028137207}
[INFO 23-05-21 18:08:04.7178 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp338u5rqh/model/ with prefix 9de16e8fb0c84799
[INFO 23-05-21 18:08:04.7220 CDT decision_forest.cc:660] Model loaded with 102 root(s), 3268 node(s), and 2 input feature(s).
[INFO 23-05-21 18:08:04.7220 CDT kernel.cc:1074] Use fast generic engine
```

**TF-DF** attaches a **semantics** to each feature. This semantics controls how the feature is used by the model. The following semantics are currently supported.

-   **Numerical**: Generally for quantities or counts with full ordering. For example, the age of a person, or the number of items in a bag. Can be a float or an integer. Missing values are represented with a float(Nan) or with an empty sparse tensor.
-   **Categorical**: Generally for a type/class in finite set of possible values without ordering. For example, the color RED in the set {RED, BLUE, GREEN}. Can be a string or an integer. Missing values are represented as &ldquo;&rdquo; (empty string), value -2 or with an empty sparse tensor.
-   **Categorical-Set**: A set of categorical values. Great to represent tokenized text. Can be a string or an integer in a sparse tensor or a ragged tensor (recommended). The order/index of each item doesnt matter.
    
    If not specified, the semantics is inferred from the representation type and shown in the training logs:
    
    -   int, float (dense or sparse) -> Numerical semantics
    
    -   str, (dense or sparse) -> Categorical semantics
    
    -   int, str (ragged) -> Categorical-Set semantics

In some cases, the inferred semantics is incorrect. For example: An Enum stored as an integer is semantically categorical, but it will be detected as numerical. In this case, you should specify the semantic argument in the input. The education<sub>num</sub> field of the Adult dataset is a classic example.

```python
feature_1 = tfdf.keras.FeatureUsage(name="year", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
feature_2 = tfdf.keras.FeatureUsage(name="bill_length_mm")
feature_3 = tfdf.keras.FeatureUsage(name="sex")
all_features = [feature_1, feature_2, feature_3]

model_3 = tfdf.keras.GradientBoostedTreesModel(features=all_features, exclude_non_specified_features=True)
model_3.compile(metrics=["accuracy"])

model_3.fit(train_ds, validation_data=test_ds)
```

```
Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp85w5v3lt as temporary training directory
Reading training dataset...
Training dataset read in 0:00:00.071150. Found 243 examples.
Reading validation dataset...
Num validation examples: tf.Tensor(101, shape=(), dtype=int32)
Validation dataset read in 0:00:00.063014. Found 101 examples.
Training model...
Model trained in 0:00:00.045273
Compiling model...
[WARNING 23-05-21 18:08:05.0026 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:05.0026 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:05.0026 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
[INFO 23-05-21 18:08:05.1870 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp85w5v3lt/model/ with prefix 95340771a14d499e
[INFO 23-05-21 18:08:05.1886 CDT decision_forest.cc:660] Model loaded with 33 root(s), 1247 node(s), and 3 input feature(s).
[INFO 23-05-21 18:08:05.1886 CDT kernel.cc:1074] Use fast generic engine
Model compiled.
```

    <keras.callbacks.History at 0x16cca8dc0>

Note that `year` is in the list of CATEGORICAL features (unlike the first run)


<a id="org2aabab3"></a>

# Hyper-parameters

**Hyper-parameters** are paramters of the training algorithm that impact the quality of the final model. They are specified in the model class constructor. The list of hyper-parameters is visible with the *question mark* colab command.

**I will figure out how to obtain that list without the question mark command.**

```python
# A classical but slightly more complex model.
model_6 = tfdf.keras.GradientBoostedTreesModel(
    num_trees=500, growing_strategy="BEST_FIRST_GLOBAL", max_depth=8)

model_6.fit(train_ds)
```

```
Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpujxude3v as temporary training directory
Reading training dataset...
Training dataset read in 0:00:00.084673. Found 243 examples.
Training model...
[WARNING 23-05-21 18:08:05.3840 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:05.3840 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:05.3840 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
Model trained in 0:00:02.150270
Compiling model...
Model compiled.
[INFO 23-05-21 18:08:07.5151 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpujxude3v/model/ with prefix edd51640dcbf46ec
[INFO 23-05-21 18:08:07.6191 CDT decision_forest.cc:660] Model loaded with 1500 root(s), 88026 node(s), and 7 input feature(s).
[INFO 23-05-21 18:08:07.6191 CDT kernel.cc:1074] Use fast generic engine
```

    <keras.callbacks.History at 0x16dc91040>

```python
model_6.summary()
```

```
Model: "gradient_boosted_trees_model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
=================================================================
Total params: 1
Trainable params: 0
Non-trainable params: 1
_________________________________________________________________
Type: "GRADIENT_BOOSTED_TREES"
Task: CLASSIFICATION
Label: "__LABEL"

Input Features (7):
	bill_depth_mm
	bill_length_mm
	body_mass_g
	flipper_length_mm
	island
	sex
	year

No weights

Variable Importance: INV_MEAN_MIN_DEPTH:
    1.    "bill_length_mm"  0.375446 ################
    2.     "bill_depth_mm"  0.318281 ###########
    3.            "island"  0.271386 ########
    4. "flipper_length_mm"  0.258080 #######
    5.       "body_mass_g"  0.197906 ###
    6.              "year"  0.157296 
    7.               "sex"  0.154175 

Variable Importance: NUM_AS_ROOT:
    1.            "island" 523.000000 ################
    2.    "bill_length_mm" 499.000000 ###############
    3. "flipper_length_mm" 455.000000 #############
    4.     "bill_depth_mm" 23.000000 

Variable Importance: NUM_NODES:
    1.    "bill_length_mm" 11824.000000 ################
    2.     "bill_depth_mm" 10596.000000 ##############
    3.       "body_mass_g" 7330.000000 #########
    4. "flipper_length_mm" 7296.000000 #########
    5.            "island" 2801.000000 ##
    6.              "year" 2234.000000 #
    7.               "sex" 1182.000000 

Variable Importance: SUM_SCORE:
    1.    "bill_length_mm" 290.119603 ################
    2. "flipper_length_mm" 205.296322 ###########
    3.            "island" 105.308220 #####
    4.     "bill_depth_mm" 32.283865 #
    5.       "body_mass_g"  3.782699 
    6.              "year"  0.485703 
    7.               "sex"  0.016964 



Loss: MULTINOMIAL_LOG_LIKELIHOOD
Validation loss value: 7.6024e-05
Number of trees per iteration: 3
Node format: NOT_SET
Number of trees: 1500
Total number of nodes: 88026

Number of nodes by tree:
Count: 1500 Average: 58.684 StdDev: 4.58157
Min: 13 Max: 61 Ignored: 0
----------------------------------------------
[ 13, 15)    2   0.13%   0.13%
[ 15, 17)    1   0.07%   0.20%
[ 17, 20)    0   0.00%   0.20%
[ 20, 22)    0   0.00%   0.20%
[ 22, 25)    0   0.00%   0.20%
[ 25, 27)    0   0.00%   0.20%
[ 27, 30)    1   0.07%   0.27%
[ 30, 32)    0   0.00%   0.27%
[ 32, 35)    2   0.13%   0.40%
[ 35, 37)    0   0.00%   0.40%
[ 37, 39)    1   0.07%   0.47%
[ 39, 42)    5   0.33%   0.80%
[ 42, 44)   10   0.67%   1.47%
[ 44, 47)   18   1.20%   2.67%
[ 47, 49)   22   1.47%   4.13%
[ 49, 52)   60   4.00%   8.13% #
[ 52, 54)   48   3.20%  11.33%
[ 54, 57)   86   5.73%  17.07% #
[ 57, 59)  103   6.87%  23.93% #
[ 59, 61] 1141  76.07% 100.00% ##########

Depth by leafs:
Count: 44763 Average: 5.8448 StdDev: 1.60806
Min: 2 Max: 8 Ignored: 0
----------------------------------------------
[ 2, 3) 1291   2.88%   2.88% #
[ 3, 4) 2088   4.66%   7.55% ##
[ 4, 5) 6778  15.14%  22.69% #######
[ 5, 6) 7646  17.08%  39.77% ########
[ 6, 7) 9747  21.77%  61.55% ##########
[ 7, 8) 8743  19.53%  81.08% #########
[ 8, 8] 8470  18.92% 100.00% #########

Number of training obs by leaf:
Count: 44763 Average: 0 StdDev: 0
Min: 0 Max: 0 Ignored: 0
----------------------------------------------
[ 0, 0] 44763 100.00% 100.00% ##########

Attribute in nodes:
	11824 : bill_length_mm [NUMERICAL]
	10596 : bill_depth_mm [NUMERICAL]
	7330 : body_mass_g [NUMERICAL]
	7296 : flipper_length_mm [NUMERICAL]
	2801 : island [CATEGORICAL]
	2234 : year [NUMERICAL]
	1182 : sex [CATEGORICAL]

Attribute in nodes with depth <= 0:
	523 : island [CATEGORICAL]
	499 : bill_length_mm [NUMERICAL]
	455 : flipper_length_mm [NUMERICAL]
	23 : bill_depth_mm [NUMERICAL]

Attribute in nodes with depth <= 1:
	1408 : bill_length_mm [NUMERICAL]
	1350 : bill_depth_mm [NUMERICAL]
	924 : island [CATEGORICAL]
	626 : flipper_length_mm [NUMERICAL]
	180 : body_mass_g [NUMERICAL]
	12 : sex [CATEGORICAL]

Attribute in nodes with depth <= 2:
	2819 : bill_depth_mm [NUMERICAL]
	2349 : bill_length_mm [NUMERICAL]
	1318 : body_mass_g [NUMERICAL]
	1123 : flipper_length_mm [NUMERICAL]
	1077 : island [CATEGORICAL]
	319 : year [NUMERICAL]
	204 : sex [CATEGORICAL]

Attribute in nodes with depth <= 3:
	5859 : bill_length_mm [NUMERICAL]
	4207 : bill_depth_mm [NUMERICAL]
	2165 : body_mass_g [NUMERICAL]
	1979 : flipper_length_mm [NUMERICAL]
	1387 : island [CATEGORICAL]
	706 : year [NUMERICAL]
	236 : sex [CATEGORICAL]

Attribute in nodes with depth <= 5:
	9209 : bill_length_mm [NUMERICAL]
	8111 : bill_depth_mm [NUMERICAL]
	5473 : body_mass_g [NUMERICAL]
	4885 : flipper_length_mm [NUMERICAL]
	2575 : island [CATEGORICAL]
	1513 : year [NUMERICAL]
	773 : sex [CATEGORICAL]

Condition type in nodes:
	39280 : HigherCondition
	3983 : ContainsBitmapCondition
Condition type in nodes with depth <= 0:
	977 : HigherCondition
	523 : ContainsBitmapCondition
Condition type in nodes with depth <= 1:
	3564 : HigherCondition
	936 : ContainsBitmapCondition
Condition type in nodes with depth <= 2:
	7928 : HigherCondition
	1281 : ContainsBitmapCondition
Condition type in nodes with depth <= 3:
	14916 : HigherCondition
	1623 : ContainsBitmapCondition
Condition type in nodes with depth <= 5:
	29191 : HigherCondition
	3348 : ContainsBitmapCondition

Training logs:
Number of iteration to final model: 500
	Iter:1 train-loss:0.919744 valid-loss:0.914596  train-accuracy:0.990654 valid-accuracy:1.000000
	Iter:2 train-loss:0.780398 valid-loss:0.772710  train-accuracy:0.990654 valid-accuracy:1.000000
	Iter:3 train-loss:0.668444 valid-loss:0.660671  train-accuracy:0.990654 valid-accuracy:1.000000
	Iter:4 train-loss:0.576662 valid-loss:0.569235  train-accuracy:0.990654 valid-accuracy:1.000000
	Iter:5 train-loss:0.500238 valid-loss:0.493459  train-accuracy:0.990654 valid-accuracy:1.000000
	Iter:6 train-loss:0.435140 valid-loss:0.426582  train-accuracy:0.990654 valid-accuracy:1.000000
	Iter:16 train-loss:0.120116 valid-loss:0.118244  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:26 train-loss:0.037248 valid-loss:0.040347  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:36 train-loss:0.011058 valid-loss:0.013689  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:46 train-loss:0.003254 valid-loss:0.004820  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:56 train-loss:0.000971 valid-loss:0.001859  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:66 train-loss:0.000295 valid-loss:0.000899  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:76 train-loss:0.000117 valid-loss:0.000499  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:86 train-loss:0.000063 valid-loss:0.000327  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:96 train-loss:0.000043 valid-loss:0.000277  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:106 train-loss:0.000033 valid-loss:0.000245  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:116 train-loss:0.000027 valid-loss:0.000222  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:126 train-loss:0.000023 valid-loss:0.000207  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:136 train-loss:0.000020 valid-loss:0.000190  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:146 train-loss:0.000018 valid-loss:0.000178  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:156 train-loss:0.000016 valid-loss:0.000163  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:166 train-loss:0.000015 valid-loss:0.000150  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:176 train-loss:0.000013 valid-loss:0.000141  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:186 train-loss:0.000012 valid-loss:0.000134  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:196 train-loss:0.000011 valid-loss:0.000128  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:206 train-loss:0.000011 valid-loss:0.000124  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:216 train-loss:0.000010 valid-loss:0.000120  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:226 train-loss:0.000009 valid-loss:0.000118  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:236 train-loss:0.000009 valid-loss:0.000115  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:246 train-loss:0.000008 valid-loss:0.000113  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:256 train-loss:0.000008 valid-loss:0.000111  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:266 train-loss:0.000008 valid-loss:0.000109  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:276 train-loss:0.000007 valid-loss:0.000108  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:286 train-loss:0.000007 valid-loss:0.000106  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:296 train-loss:0.000007 valid-loss:0.000104  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:306 train-loss:0.000006 valid-loss:0.000103  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:316 train-loss:0.000006 valid-loss:0.000101  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:326 train-loss:0.000006 valid-loss:0.000100  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:336 train-loss:0.000006 valid-loss:0.000098  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:346 train-loss:0.000005 valid-loss:0.000096  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:356 train-loss:0.000005 valid-loss:0.000094  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:366 train-loss:0.000005 valid-loss:0.000092  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:376 train-loss:0.000005 valid-loss:0.000091  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:386 train-loss:0.000005 valid-loss:0.000089  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:396 train-loss:0.000005 valid-loss:0.000088  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:406 train-loss:0.000005 valid-loss:0.000086  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:416 train-loss:0.000004 valid-loss:0.000085  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:426 train-loss:0.000004 valid-loss:0.000083  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:436 train-loss:0.000004 valid-loss:0.000082  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:446 train-loss:0.000004 valid-loss:0.000081  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:456 train-loss:0.000004 valid-loss:0.000081  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:466 train-loss:0.000004 valid-loss:0.000079  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:476 train-loss:0.000004 valid-loss:0.000078  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:486 train-loss:0.000004 valid-loss:0.000077  train-accuracy:1.000000 valid-accuracy:1.000000
	Iter:496 train-loss:0.000004 valid-loss:0.000076  train-accuracy:1.000000 valid-accuracy:1.000000
```

```python
# A more complex, but possibly, more accurate model.
model_7 = tfdf.keras.GradientBoostedTreesModel(
    num_trees=500,
    growing_strategy="BEST_FIRST_GLOBAL",
    max_depth=8,
    split_axis="SPARSE_OBLIQUE",
    categorical_algorithm="RANDOM",
    )

model_7.fit(train_ds)
```

```
Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmppyi9uibs as temporary training directory
Reading training dataset...
WARNING:tensorflow:5 out of the last 5 calls to <function CoreModel._consumes_training_examples_until_eof at 0x156ff38b0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
[WARNING 23-05-21 18:08:07.9705 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:07.9706 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:07.9706 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
WARNING:tensorflow:5 out of the last 5 calls to <function CoreModel._consumes_training_examples_until_eof at 0x156ff38b0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
Training dataset read in 0:00:00.150575. Found 243 examples.
Training model...
Model trained in 0:00:03.457530
Compiling model...
WARNING:tensorflow:5 out of the last 5 calls to <function InferenceCoreModel.make_predict_function.<locals>.predict_function_trained at 0x16e274f70> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
[INFO 23-05-21 18:08:11.4695 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmppyi9uibs/model/ with prefix 07ba26a767d543e5
[INFO 23-05-21 18:08:11.5781 CDT decision_forest.cc:660] Model loaded with 1500 root(s), 86172 node(s), and 7 input feature(s).
[INFO 23-05-21 18:08:11.5781 CDT abstract_model.cc:1312] Engine "GradientBoostedTreesGeneric" built
[INFO 23-05-21 18:08:11.5782 CDT kernel.cc:1074] Use fast generic engine
WARNING:tensorflow:5 out of the last 5 calls to <function InferenceCoreModel.make_predict_function.<locals>.predict_function_trained at 0x16e274f70> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
Model compiled.
```

    <keras.callbacks.History at 0x16e2a3100>

As new training methods are published and implemented, combinations of hyper-parameters can emerge as good or almost-always-better than the default parameters. To avoid changing the default hyper-parameter values these good combinations are indexed and availale as hyper-parameter templates.

For example, the benchmark<sub>rank1</sub> template is the best combination on our internal benchmarks. Those templates are versioned to allow training configuration stability e.g. benchmark<sub>rank1</sub>@v1.

```python
# A good template of hyper-parameters.
model_8 = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1")
model_8.fit(train_ds)
```

```
Resolve hyper-parameter template "benchmark_rank1" to "benchmark_rank1@v1" -> {'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}.
Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpiwy100sd as temporary training directory
Reading training dataset...
WARNING:tensorflow:6 out of the last 6 calls to <function CoreModel._consumes_training_examples_until_eof at 0x156ff38b0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
[WARNING 23-05-21 18:08:11.7852 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:11.7852 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
[WARNING 23-05-21 18:08:11.7852 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
WARNING:tensorflow:6 out of the last 6 calls to <function CoreModel._consumes_training_examples_until_eof at 0x156ff38b0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
Training dataset read in 0:00:00.081700. Found 243 examples.
Training model...
Model trained in 0:00:01.061980
Compiling model...
WARNING:tensorflow:6 out of the last 6 calls to <function InferenceCoreModel.make_predict_function.<locals>.predict_function_trained at 0x16bebbdc0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
[INFO 23-05-21 18:08:12.8867 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpiwy100sd/model/ with prefix a218cf53b23f437e
[INFO 23-05-21 18:08:12.9315 CDT decision_forest.cc:660] Model loaded with 900 root(s), 35792 node(s), and 7 input feature(s).
[INFO 23-05-21 18:08:12.9315 CDT kernel.cc:1074] Use fast generic engine
WARNING:tensorflow:6 out of the last 6 calls to <function InferenceCoreModel.make_predict_function.<locals>.predict_function_trained at 0x16bebbdc0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
Model compiled.
```

    <keras.callbacks.History at 0x16cad8670>

The available templates are available with `predefined_hyperparameters`. Note that different learning algorithms have different templates, even if the name is similar.

```python
print(tfdf.keras.GradientBoostedTreesModel.predefined_hyperparameters())
```

    [HyperParameterTemplate(name='better_default', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL'}, description='A configuration that is generally better than the default parameters without being more expensive.'), HyperParameterTemplate(name='benchmark_rank1', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}, description='Top ranking hyper-parameters on our benchmark slightly modified to run in reasonable time.')]

What is returned are the predefined hyper-parameters of the Gradient Boosted Tree model.


<a id="org105f10b"></a>

# Feature Preprocessing

Pre-processing features is sometimes necessary to consume signals with complex structures, to regularize the model or to apply transfer learning. Pre-processing can be done in one of three ways:

1.  **Preprocessing on the pandas dataframe**: This solution is easy tto implement and generally suitable for experiementation. However, the pre-processing logic will not be exported in the model by model.save()
2.  **Keras Preprocessing**: While more complex than the previous solution, Keras Preprocessing is packaged in the model.
3.  **TensorFlow Feature Columns**: This API is part of the TF Estimator library (!= Keras) and planned for deprecation. This solution is interesting when using existing preprocessing code.

**Note**: Using **TensorFlow Hub** pre-trained embedding is often, a great way to consume text and image with TF-DF.

In the next example, pre-process the body<sub>mass</sub><sub>g</sub> feature into body<sub>mass</sub><sub>kg</sub> = body<sub>mass</sub><sub>g</sub> / 1000. The bill<sub>length</sub><sub>mm</sub> is consumed without preprocessing. Note that such monotonic transformations have generally no impact on decision forest models.

```python
body_mass_g = tf.keras.layers.Input(shape=(1,), name="body_mass_g")
body_mass_kg = body_mass_g / 1000.0

bill_length_mm = tf.keras.layers.Input(shape=(1,), name="bill_length_mm")

raw_inputs = {"body_mass_g": body_mass_g, "bill_length_mm": bill_length_mm}
processed_inputs = {"body_mass_kg": body_mass_kg, "bill_length_mm": bill_length_mm}

# "preprocessor" contains the preprocessing logic.
preprocessor = tf.keras.Model(inputs=raw_inputs, outputs=processed_inputs)

# "model_4" contains both the pre-processing logic and the decision forest.
model_4 = tfdf.keras.RandomForestModel(preprocessing=preprocessor)
model_4.fit(train_ds)

model_4.summary()
```

```
Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpmftmu7ng as temporary training directory
Reading training dataset...
Training dataset read in 0:00:00.129735. Found 243 examples.
Training model...
Model trained in 0:00:00.021975
Compiling model...
Model compiled.
WARNING:tensorflow:5 out of the last 10 calls to <function InferenceCoreModel.yggdrasil_model_path_tensor at 0x16e2f5160> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
/Users/umbertofasci/miniforge3/envs/tensorflow-metal/lib/python3.9/site-packages/keras/engine/functional.py:639: UserWarning: Input dict contained keys ['island', 'bill_depth_mm', 'flipper_length_mm', 'sex', 'year'] which did not match any model input. They will be ignored by the model.
  inputs = self._flatten_to_reference_inputs(inputs)
[INFO 23-05-21 18:08:13.3931 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpmftmu7ng/model/ with prefix 31397880dd3241e7
[INFO 23-05-21 18:08:13.4002 CDT decision_forest.cc:660] Model loaded with 300 root(s), 6000 node(s), and 2 input feature(s).
[INFO 23-05-21 18:08:13.4003 CDT kernel.cc:1074] Use fast generic engine
WARNING:tensorflow:5 out of the last 10 calls to <function InferenceCoreModel.yggdrasil_model_path_tensor at 0x16e2f5160> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
Model: "random_forest_model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 model (Functional)          {'body_mass_kg': (None,   0         
                             1),                                 
                              'bill_length_mm': (None            
                             , 1)}                               
                                                                 
=================================================================
Total params: 1
Trainable params: 0
Non-trainable params: 1
_________________________________________________________________
Type: "RANDOM_FOREST"
Task: CLASSIFICATION
Label: "__LABEL"

Input Features (2):
	bill_length_mm
	body_mass_kg

No weights

Variable Importance: INV_MEAN_MIN_DEPTH:
    1. "bill_length_mm"  1.000000 ################
    2.   "body_mass_kg"  0.425244 

Variable Importance: NUM_AS_ROOT:
    1. "bill_length_mm" 300.000000 

Variable Importance: NUM_NODES:
    1. "bill_length_mm" 1542.000000 ################
    2.   "body_mass_kg" 1308.000000 

Variable Importance: SUM_SCORE:
    1. "bill_length_mm" 46779.241968 ################
    2.   "body_mass_kg" 24649.079174 



Winner takes all: true
Out-of-bag evaluation: accuracy:0.91358 logloss:0.480882
Number of trees: 300
Total number of nodes: 6000

Number of nodes by tree:
Count: 300 Average: 20 StdDev: 2.82607
Min: 13 Max: 29 Ignored: 0
----------------------------------------------
[ 13, 14)  3   1.00%   1.00%
[ 14, 15)  0   0.00%   1.00%
[ 15, 16) 24   8.00%   9.00% ###
[ 16, 17)  0   0.00%   9.00%
[ 17, 18) 41  13.67%  22.67% #####
[ 18, 19)  0   0.00%  22.67%
[ 19, 20) 84  28.00%  50.67% ##########
[ 20, 21)  0   0.00%  50.67%
[ 21, 22) 72  24.00%  74.67% #########
[ 22, 23)  0   0.00%  74.67%
[ 23, 24) 58  19.33%  94.00% #######
[ 24, 25)  0   0.00%  94.00%
[ 25, 26) 13   4.33%  98.33% ##
[ 26, 27)  0   0.00%  98.33%
[ 27, 28)  4   1.33%  99.67%
[ 28, 29)  0   0.00%  99.67%
[ 29, 29]  1   0.33% 100.00%

Depth by leafs:
Count: 3150 Average: 3.91333 StdDev: 1.34085
Min: 1 Max: 9 Ignored: 0
----------------------------------------------
[ 1, 2)  18   0.57%   0.57%
[ 2, 3) 382  12.13%  12.70% ####
[ 3, 4) 977  31.02%  43.71% ##########
[ 4, 5) 812  25.78%  69.49% ########
[ 5, 6) 561  17.81%  87.30% ######
[ 6, 7) 264   8.38%  95.68% ###
[ 7, 8) 113   3.59%  99.27% #
[ 8, 9)  21   0.67%  99.94%
[ 9, 9]   2   0.06% 100.00%

Number of training obs by leaf:
Count: 3150 Average: 23.1429 StdDev: 27.8301
Min: 5 Max: 120 Ignored: 0
----------------------------------------------
[   5,  10) 1990  63.17%  63.17% ##########
[  10,  16)  227   7.21%  70.38% #
[  16,  22)   30   0.95%  71.33%
[  22,  28)    5   0.16%  71.49%
[  28,  34)   32   1.02%  72.51%
[  34,  39)   61   1.94%  74.44%
[  39,  45)  105   3.33%  77.78% #
[  45,  51)  104   3.30%  81.08% #
[  51,  57)   87   2.76%  83.84%
[  57,  63)   98   3.11%  86.95%
[  63,  68)   64   2.03%  88.98%
[  68,  74)   55   1.75%  90.73%
[  74,  80)   36   1.14%  91.87%
[  80,  86)   81   2.57%  94.44%
[  86,  92)   70   2.22%  96.67%
[  92,  97)   52   1.65%  98.32%
[  97, 103)   30   0.95%  99.27%
[ 103, 109)   18   0.57%  99.84%
[ 109, 115)    3   0.10%  99.94%
[ 115, 120]    2   0.06% 100.00%

Attribute in nodes:
	1542 : bill_length_mm [NUMERICAL]
	1308 : body_mass_kg [NUMERICAL]

Attribute in nodes with depth <= 0:
	300 : bill_length_mm [NUMERICAL]

Attribute in nodes with depth <= 1:
	542 : bill_length_mm [NUMERICAL]
	340 : body_mass_kg [NUMERICAL]

Attribute in nodes with depth <= 2:
	879 : bill_length_mm [NUMERICAL]
	785 : body_mass_kg [NUMERICAL]

Attribute in nodes with depth <= 3:
	1207 : bill_length_mm [NUMERICAL]
	1044 : body_mass_kg [NUMERICAL]

Attribute in nodes with depth <= 5:
	1494 : bill_length_mm [NUMERICAL]
	1282 : body_mass_kg [NUMERICAL]

Condition type in nodes:
	2850 : HigherCondition
Condition type in nodes with depth <= 0:
	300 : HigherCondition
Condition type in nodes with depth <= 1:
	882 : HigherCondition
Condition type in nodes with depth <= 2:
	1664 : HigherCondition
Condition type in nodes with depth <= 3:
	2251 : HigherCondition
Condition type in nodes with depth <= 5:
	2776 : HigherCondition
Node format: NOT_SET

Training OOB:
	trees: 1, Out-of-bag evaluation: accuracy:0.90625 logloss:3.37909
	trees: 15, Out-of-bag evaluation: accuracy:0.908714 logloss:1.88241
	trees: 25, Out-of-bag evaluation: accuracy:0.91358 logloss:0.884956
	trees: 36, Out-of-bag evaluation: accuracy:0.917695 logloss:0.881573
	trees: 46, Out-of-bag evaluation: accuracy:0.91358 logloss:0.878336
	trees: 58, Out-of-bag evaluation: accuracy:0.917695 logloss:0.877692
	trees: 70, Out-of-bag evaluation: accuracy:0.921811 logloss:0.876194
	trees: 81, Out-of-bag evaluation: accuracy:0.909465 logloss:0.877944
	trees: 93, Out-of-bag evaluation: accuracy:0.90535 logloss:0.879283
	trees: 103, Out-of-bag evaluation: accuracy:0.909465 logloss:0.878753
	trees: 113, Out-of-bag evaluation: accuracy:0.91358 logloss:0.87967
	trees: 124, Out-of-bag evaluation: accuracy:0.91358 logloss:0.877961
	trees: 137, Out-of-bag evaluation: accuracy:0.91358 logloss:0.478915
	trees: 147, Out-of-bag evaluation: accuracy:0.917695 logloss:0.47581
	trees: 161, Out-of-bag evaluation: accuracy:0.917695 logloss:0.476437
	trees: 173, Out-of-bag evaluation: accuracy:0.917695 logloss:0.475837
	trees: 183, Out-of-bag evaluation: accuracy:0.91358 logloss:0.476234
	trees: 194, Out-of-bag evaluation: accuracy:0.925926 logloss:0.476402
	trees: 205, Out-of-bag evaluation: accuracy:0.925926 logloss:0.477254
	trees: 215, Out-of-bag evaluation: accuracy:0.921811 logloss:0.477153
	trees: 225, Out-of-bag evaluation: accuracy:0.921811 logloss:0.478058
	trees: 237, Out-of-bag evaluation: accuracy:0.921811 logloss:0.477183
	trees: 250, Out-of-bag evaluation: accuracy:0.917695 logloss:0.478852
	trees: 260, Out-of-bag evaluation: accuracy:0.91358 logloss:0.478021
	trees: 270, Out-of-bag evaluation: accuracy:0.921811 logloss:0.478902
	trees: 283, Out-of-bag evaluation: accuracy:0.917695 logloss:0.480639
	trees: 293, Out-of-bag evaluation: accuracy:0.91358 logloss:0.480454
	trees: 300, Out-of-bag evaluation: accuracy:0.91358 logloss:0.480882
```

The following example re-implements the same logic using TensorFlow Feature Columns.

```python
def g_to_kg(x):
    return x / 1000

feature_columns = [
    tf.feature_column.numeric_column("body_mass_g", normalizer_fn=g_to_kg),
    tf.feature_column.numeric_column("bill_length_mm"),
]

preprocessing = tf.keras.layers.DenseFeatures(feature_columns)

model_5 = tfdf.keras.RandomForestModel(preprocessing=preprocessing)
model_5.fit(train_ds)
```

```
WARNING:tensorflow:From /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/ipykernel_37562/3447023075.py:5: numeric_column (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be removed in a future version.
Instructions for updating:
Use Keras preprocessing layers instead, either directly or via the `tf.keras.utils.FeatureSpace` utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.
WARNING:tensorflow:From /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/ipykernel_37562/3447023075.py:5: numeric_column (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be removed in a future version.
Instructions for updating:
Use Keras preprocessing layers instead, either directly or via the `tf.keras.utils.FeatureSpace` utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.
Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpngu4fv5k as temporary training directory
Reading training dataset...
Training dataset read in 0:00:00.093859. Found 243 examples.
Training model...
Model trained in 0:00:00.021618
Compiling model...
Model compiled.
WARNING:tensorflow:6 out of the last 11 calls to <function InferenceCoreModel.yggdrasil_model_path_tensor at 0x17babbaf0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
[INFO 23-05-21 18:08:13.7193 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpngu4fv5k/model/ with prefix 5fc95a09a1a842fe
[INFO 23-05-21 18:08:13.7267 CDT decision_forest.cc:660] Model loaded with 300 root(s), 6000 node(s), and 2 input feature(s).
[INFO 23-05-21 18:08:13.7267 CDT kernel.cc:1074] Use fast generic engine
WARNING:tensorflow:6 out of the last 11 calls to <function InferenceCoreModel.yggdrasil_model_path_tensor at 0x17babbaf0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
```

    <keras.callbacks.History at 0x16e7abb50>


<a id="org9e65e84"></a>

# Training a regression model

The previous example trains a classification model(TF-DF does not differentiate between binary classification and multi-class classification). In the next example, train a regression model on the Abalone dataset. The objective of this dataset is to predict the number of rings on a shell of a abalone.

**Note**: The csv file is assembled by appending UCI&rsquo;s header and data files. No preprocessing was applied.

```python
!wget -q https://storage.googleapis.com/download.tensorflow.org/data/abalone_raw.csv -O /tmp/abalone.csv

dataset_df = pd.read_csv("/tmp/abalone.csv")
print(dataset_df.head(3))
```

      Type  LongestShell  Diameter  Height  WholeWeight  ShuckedWeight   
    0    M         0.455     0.365   0.095       0.5140         0.2245  \
    1    M         0.350     0.265   0.090       0.2255         0.0995   
    2    F         0.530     0.420   0.135       0.6770         0.2565   
    
       VisceraWeight  ShellWeight  Rings  
    0         0.1010         0.15     15  
    1         0.0485         0.07      7  
    2         0.1415         0.21      9  

```python
# Split the dataset into a training and testing dataset.
train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

# Name of the label column.
label = "Rings"

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
```

    2898 examples in training, 1279 examples for testing.

```python
# Configure the model
model_7 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)

# Train the model
model_7.fit(train_ds)
```

```
Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpe3d1txkk as temporary training directory
Reading training dataset...
Training dataset read in 0:00:00.097030. Found 2898 examples.
Training model...
[INFO 23-05-21 18:08:15.2297 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpe3d1txkk/model/ with prefix 397ac8325947412a
Model trained in 0:00:00.685941
Compiling model...
Model compiled.
[INFO 23-05-21 18:08:15.5284 CDT decision_forest.cc:660] Model loaded with 300 root(s), 259488 node(s), and 8 input feature(s).
[INFO 23-05-21 18:08:15.5284 CDT kernel.cc:1074] Use fast generic engine
```

    <keras.callbacks.History at 0x1578aa8e0>

```python
# Evaluate the model on the test dataset
model_7.compile(metrics=["mse"])
evaluation = model_7.evaluate(test_ds, return_dict=True)

print(evaluation)
print()
print(f"MSE: {evaluation['mse']}")
print(f"RMSE: {math.sqrt(evaluation['mse'])}")
```

    WARNING:tensorflow:5 out of the last 5 calls to <function InferenceCoreModel.make_test_function.<locals>.test_function at 0x17c6a1820> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    WARNING:tensorflow:5 out of the last 5 calls to <function InferenceCoreModel.make_test_function.<locals>.test_function at 0x17c6a1820> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    2/2 [==============================] - 0s 16ms/step - loss: 0.0000e+00 - mse: 4.4014
    
    {'loss': 0.0, 'mse': 4.401405334472656}
    
    MSE: 4.401405334472656
    RMSE: 2.0979526530578942


<a id="orgbedf592"></a>

# Conclusion

This concludes the basic overview of TensorFlow Decision Forest utility.
