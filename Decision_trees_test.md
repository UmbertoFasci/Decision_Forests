
# Table of Contents

1.  [Importing Libraries](#orgfdc5346)
2.  [Training a Random Forest model](#org1909609)
3.  [Evaluate the model](#org75ffd61)
4.  [TensorFlow Serving](#org8872b20)
5.  [Model structure and feature importance](#org0d6f1ed)
6.  [Using make<sub>inspector</sub>](#org83ba3ba)
7.  [Model self evaluation](#org692b22c)
8.  [Plotting the training logs](#orge4e78ec)
9.  [Retrain model with different learning algorithm](#org76f07c5)
10. [Using a subset of features](#orgcfc9a6d)
11. [Hyper-parameters](#org690b29e)
12. [Feature Preprocessing](#orgb96aed6)
13. [Training a regression model](#orgccccc74)
14. [Conclusion](#orgb84ff1d)

The following document will contain the basic instructions for creating a decision tree model with tensorflow.
In this document I will:

1.  Train a binary classification Random Forest on a dataset containing numerical, categorical, and missing data.
2.  Evaluate the model on the test set.
3.  Prepare the model for TensorFlow Serving
4.  Examine the overall of the model and the importance of each feature.
5.  Re-train the model with a different learning algorithm (Gradient Boost Decision Trees).
6.  Use a different set of input features.
7.  Change the hyperparameters of the model.
8.  Preprocess the features.
9.  Train the model for regression.


<a id="orgfdc5346"></a>

# Importing Libraries

    import tensorflow_decision_forests as tfdf
    
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import math

    Found TensorFlow Decision Forests v1.3.0


<a id="org1909609"></a>

# Training a Random Forest model

      species     island  bill_length_mm  bill_depth_mm  flipper_length_mm   
    0  Adelie  Torgersen            39.1           18.7              181.0  \
    1  Adelie  Torgersen            39.5           17.4              186.0   
    2  Adelie  Torgersen            40.3           18.0              195.0   
    
       body_mass_g     sex  year  
    0       3750.0    male  2007  
    1       3800.0  female  2007  
    2       3250.0  female  2007  

    Label classes: ['Adelie', 'Gentoo', 'Chinstrap']

    247 examples in training, 97 examples for testing.

    Use 8 thread(s) for training
    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp06vmo__w as temporary training directory
    Reading training dataset...
    Training tensor examples:
    Features: {'island': <tf.Tensor 'data:0' shape=(None,) dtype=string>, 'bill_length_mm': <tf.Tensor 'data_1:0' shape=(None,) dtype=float64>, 'bill_depth_mm': <tf.Tensor 'data_2:0' shape=(None,) dtype=float64>, 'flipper_length_mm': <tf.Tensor 'data_3:0' shape=(None,) dtype=float64>, 'body_mass_g': <tf.Tensor 'data_4:0' shape=(None,) dtype=float64>, 'sex': <tf.Tensor 'data_5:0' shape=(None,) dtype=string>, 'year': <tf.Tensor 'data_6:0' shape=(None,) dtype=int64>}
    Label: Tensor("data_7:0", shape=(None,), dtype=int64)
    Weights: None
    Normalized tensor features:
     {'island': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data:0' shape=(None,) dtype=string>), 'bill_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast:0' shape=(None,) dtype=float32>), 'bill_depth_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_1:0' shape=(None,) dtype=float32>), 'flipper_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_2:0' shape=(None,) dtype=float32>), 'body_mass_g': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_3:0' shape=(None,) dtype=float32>), 'sex': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data_5:0' shape=(None,) dtype=string>), 'year': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_4:0' shape=(None,) dtype=float32>)}
    Training dataset read in 0:00:00.100529. Found 247 examples.
    Training model...
    [INFO 23-05-21 17:26:56.5263 CDT kernel.cc:773] Start Yggdrasil model training
    [INFO 23-05-21 17:26:56.5263 CDT kernel.cc:774] Collect training examples
    [INFO 23-05-21 17:26:56.5263 CDT kernel.cc:787] Dataspec guide:
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
    
    [INFO 23-05-21 17:26:56.5264 CDT kernel.cc:393] Number of batches: 1
    [INFO 23-05-21 17:26:56.5264 CDT kernel.cc:394] Number of examples: 247
    [INFO 23-05-21 17:26:56.5264 CDT kernel.cc:794] Training dataset:
    Number of records: 247
    Number of columns: 8
    
    Number of columns by type:
    	NUMERICAL: 5 (62.5%)
    	CATEGORICAL: 3 (37.5%)
    
    Columns:
    
    NUMERICAL: 5 (62.5%)
    	1: "bill_depth_mm" NUMERICAL num-nas:2 (0.809717%) mean:17.1037 min:13.1 max:21.5 sd:1.95651
    	2: "bill_length_mm" NUMERICAL num-nas:2 (0.809717%) mean:43.7812 min:32.1 max:58 sd:5.42317
    	3: "body_mass_g" NUMERICAL num-nas:2 (0.809717%) mean:4204.29 min:2850 max:6300 sd:803.355
    	4: "flipper_length_mm" NUMERICAL num-nas:2 (0.809717%) mean:200.714 min:172 max:230 sd:14.0999
    	7: "year" NUMERICAL mean:2007.99 min:2007 max:2009 sd:0.82468
    
    CATEGORICAL: 3 (37.5%)
    	0: "__LABEL" CATEGORICAL integerized vocab-size:4 no-ood-item
    	5: "island" CATEGORICAL has-dict vocab-size:4 zero-ood-items most-frequent:"Biscoe" 120 (48.583%)
    	6: "sex" CATEGORICAL num-nas:9 (3.64372%) has-dict vocab-size:3 zero-ood-items most-frequent:"female" 123 (51.6807%)
    
    Terminology:
    	nas: Number of non-available (i.e. missing) values.
    	ood: Out of dictionary.
    	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
    	tokenized: The attribute value is obtained through tokenization.
    	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
    	vocab-size: Number of unique values.
    
    [INFO 23-05-21 17:26:56.5264 CDT kernel.cc:810] Configure learner
    [INFO 23-05-21 17:26:56.5265 CDT kernel.cc:824] Training config:
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
    
    [INFO 23-05-21 17:26:56.5266 CDT kernel.cc:827] Deployment config:
    cache_path: "/var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp06vmo__w/working_cache"
    num_threads: 8
    try_resume_training: true
    [INFO 23-05-21 17:26:56.5267 CDT kernel.cc:889] Train model
    [INFO 23-05-21 17:26:56.5267 CDT random_forest.cc:416] Training random forest on 247 example(s) and 7 feature(s).
    [INFO 23-05-21 17:26:56.5271 CDT random_forest.cc:805] Training of tree  1/300 (tree index:2) done accuracy:0.965909 logloss:1.22876
    [INFO 23-05-21 17:26:56.5273 CDT random_forest.cc:805] Training of tree  11/300 (tree index:11) done accuracy:0.959016 logloss:0.0722883
    [INFO 23-05-21 17:26:56.5277 CDT random_forest.cc:805] Training of tree  21/300 (tree index:19) done accuracy:0.975708 logloss:0.0607441
    [INFO 23-05-21 17:26:56.5279 CDT random_forest.cc:805] Training of tree  31/300 (tree index:32) done accuracy:0.97166 logloss:0.068951
    [INFO 23-05-21 17:26:56.5282 CDT random_forest.cc:805] Training of tree  42/300 (tree index:45) done accuracy:0.975708 logloss:0.0707371
    [INFO 23-05-21 17:26:56.5284 CDT random_forest.cc:805] Training of tree  52/300 (tree index:49) done accuracy:0.975708 logloss:0.068778
    [INFO 23-05-21 17:26:56.5287 CDT random_forest.cc:805] Training of tree  62/300 (tree index:63) done accuracy:0.975708 logloss:0.0695978
    [INFO 23-05-21 17:26:56.5289 CDT random_forest.cc:805] Training of tree  72/300 (tree index:72) done accuracy:0.975708 logloss:0.0719713
    [INFO 23-05-21 17:26:56.5291 CDT random_forest.cc:805] Training of tree  82/300 (tree index:82) done accuracy:0.975708 logloss:0.0710908
    [INFO 23-05-21 17:26:56.5294 CDT random_forest.cc:805] Training of tree  92/300 (tree index:93) done accuracy:0.975708 logloss:0.0712041
    [INFO 23-05-21 17:26:56.5297 CDT random_forest.cc:805] Training of tree  102/300 (tree index:98) done accuracy:0.975708 logloss:0.073906
    [INFO 23-05-21 17:26:56.5300 CDT random_forest.cc:805] Training of tree  112/300 (tree index:111) done accuracy:0.975708 logloss:0.073205
    [INFO 23-05-21 17:26:56.5302 CDT random_forest.cc:805] Training of tree  122/300 (tree index:123) done accuracy:0.97166 logloss:0.0745004
    [INFO 23-05-21 17:26:56.5305 CDT random_forest.cc:805] Training of tree  132/300 (tree index:134) done accuracy:0.97166 logloss:0.0732437
    [INFO 23-05-21 17:26:56.5308 CDT random_forest.cc:805] Training of tree  143/300 (tree index:137) done accuracy:0.975708 logloss:0.0737169
    [INFO 23-05-21 17:26:56.5311 CDT random_forest.cc:805] Training of tree  154/300 (tree index:157) done accuracy:0.967611 logloss:0.0738033
    [INFO 23-05-21 17:26:56.5313 CDT random_forest.cc:805] Training of tree  164/300 (tree index:164) done accuracy:0.963563 logloss:0.0735375
    [INFO 23-05-21 17:26:56.5316 CDT random_forest.cc:805] Training of tree  174/300 (tree index:175) done accuracy:0.967611 logloss:0.0741928
    [INFO 23-05-21 17:26:56.5318 CDT random_forest.cc:805] Training of tree  184/300 (tree index:184) done accuracy:0.97166 logloss:0.073243
    [INFO 23-05-21 17:26:56.5321 CDT random_forest.cc:805] Training of tree  194/300 (tree index:194) done accuracy:0.97166 logloss:0.0741288
    [INFO 23-05-21 17:26:56.5324 CDT random_forest.cc:805] Training of tree  204/300 (tree index:203) done accuracy:0.97166 logloss:0.0739353
    [INFO 23-05-21 17:26:56.5326 CDT random_forest.cc:805] Training of tree  215/300 (tree index:215) done accuracy:0.97166 logloss:0.0732281
    [INFO 23-05-21 17:26:56.5330 CDT random_forest.cc:805] Training of tree  225/300 (tree index:221) done accuracy:0.975708 logloss:0.0738094
    [INFO 23-05-21 17:26:56.5332 CDT random_forest.cc:805] Training of tree  235/300 (tree index:235) done accuracy:0.975708 logloss:0.0738642
    [INFO 23-05-21 17:26:56.5335 CDT random_forest.cc:805] Training of tree  246/300 (tree index:241) done accuracy:0.975708 logloss:0.0747816
    [INFO 23-05-21 17:26:56.5337 CDT random_forest.cc:805] Training of tree  256/300 (tree index:256) done accuracy:0.975708 logloss:0.0747483
    [INFO 23-05-21 17:26:56.5340 CDT random_forest.cc:805] Training of tree  266/300 (tree index:261) done accuracy:0.975708 logloss:0.0755402
    [INFO 23-05-21 17:26:56.5343 CDT random_forest.cc:805] Training of tree  276/300 (tree index:276) done accuracy:0.975708 logloss:0.076221
    [INFO 23-05-21 17:26:56.5345 CDT random_forest.cc:805] Training of tree  286/300 (tree index:285) done accuracy:0.975708 logloss:0.0755566
    [INFO 23-05-21 17:26:56.5348 CDT random_forest.cc:805] Training of tree  296/300 (tree index:290) done accuracy:0.975708 logloss:0.0755158
    [INFO 23-05-21 17:26:56.5349 CDT random_forest.cc:805] Training of tree  300/300 (tree index:294) done accuracy:0.975708 logloss:0.0753986
    [INFO 23-05-21 17:26:56.5350 CDT random_forest.cc:885] Final OOB metrics: accuracy:0.975708 logloss:0.0753986
    [INFO 23-05-21 17:26:56.5353 CDT kernel.cc:926] Export model in log directory: /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp06vmo__w with prefix 89cc770bc6994b6f
    [INFO 23-05-21 17:26:56.5375 CDT kernel.cc:944] Save model in resources
    [INFO 23-05-21 17:26:56.5389 CDT abstract_model.cc:849] Model self evaluation:
    Number of predictions (without weights): 247
    Number of predictions (with weights): 247
    Task: CLASSIFICATION
    Label: __LABEL
    
    Accuracy: 0.975708  CI95[W][0.952621 0.989369]
    LogLoss: : 0.0753986
    ErrorRate: : 0.0242915
    
    Default Accuracy: : 0.445344
    Default LogLoss: : 1.04383
    Default ErrorRate: : 0.554656
    
    Confusion Table:
    truth\prediction
       0    1   2   3
    0  0    0   0   0
    1  0  108   0   2
    2  0    2  88   0
    3  0    2   0  45
    Total: 247
    
    One vs other classes:
    [INFO 23-05-21 17:26:56.5434 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp06vmo__w/model/ with prefix 89cc770bc6994b6f
    [INFO 23-05-21 17:26:56.5487 CDT decision_forest.cc:660] Model loaded with 300 root(s), 4148 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:26:56.5487 CDT abstract_model.cc:1312] Engine "RandomForestGeneric" built
    [INFO 23-05-21 17:26:56.5487 CDT kernel.cc:1074] Use fast generic engine
    Model trained in 0:00:00.025489
    Compiling model...
    Model compiled.

    <keras.callbacks.History at 0x2c68bc610>


<a id="org75ffd61"></a>

# Evaluate the model

    1/1 [==============================] - 0s 77ms/step - loss: 0.0000e+00 - accuracy: 0.9794
    
    
    loss: 0.0000
    accuracy: 0.9794


<a id="org8872b20"></a>

# TensorFlow Serving

    WARNING:absl:Found untraced functions such as call_get_leaves while saving (showing 1 of 1). These functions will not be directly callable after loading.
    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets
    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets


<a id="org0d6f1ed"></a>

# Model structure and feature importance

    Model: "random_forest_model_12"
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
        1. "flipper_length_mm"  0.457659 ################
        2.    "bill_length_mm"  0.434472 ##############
        3.            "island"  0.324085 ######
        4.     "bill_depth_mm"  0.296402 ####
        5.       "body_mass_g"  0.273813 ##
        6.              "year"  0.241458 
        7.               "sex"  0.241434 
    
    Variable Importance: NUM_AS_ROOT:
        1. "flipper_length_mm" 163.000000 ################
        2.    "bill_length_mm" 83.000000 #######
        3.     "bill_depth_mm" 32.000000 ##
        4.            "island" 15.000000 
        5.       "body_mass_g"  7.000000 
    
    Variable Importance: NUM_NODES:
        1.    "bill_length_mm" 634.000000 ################
        2. "flipper_length_mm" 372.000000 #########
        3.     "bill_depth_mm" 338.000000 ########
        4.            "island" 284.000000 ######
        5.       "body_mass_g" 253.000000 ######
        6.               "sex" 25.000000 
        7.              "year" 18.000000 
    
    Variable Importance: SUM_SCORE:
        1. "flipper_length_mm" 26397.211392 ################
        2.    "bill_length_mm" 26229.497206 ###############
        3.            "island" 11178.969668 ######
        4.     "bill_depth_mm" 7427.787270 ####
        5.       "body_mass_g" 3074.528762 #
        6.               "sex" 168.326006 
        7.              "year" 64.973497 
    
    
    
    Winner takes all: true
    Out-of-bag evaluation: accuracy:0.975708 logloss:0.0753986
    Number of trees: 300
    Total number of nodes: 4148
    
    Number of nodes by tree:
    Count: 300 Average: 13.8267 StdDev: 2.932
    Min: 7 Max: 25 Ignored: 0
    ----------------------------------------------
    [  7,  8)   2   0.67%   0.67%
    [  8,  9)   0   0.00%   0.67%
    [  9, 10)   9   3.00%   3.67% #
    [ 10, 11)   0   0.00%   3.67%
    [ 11, 12)  68  22.67%  26.33% ######
    [ 12, 13)   0   0.00%  26.33%
    [ 13, 14) 108  36.00%  62.33% ##########
    [ 14, 15)   0   0.00%  62.33%
    [ 15, 16)  56  18.67%  81.00% #####
    [ 16, 17)   0   0.00%  81.00%
    [ 17, 18)  31  10.33%  91.33% ###
    [ 18, 19)   0   0.00%  91.33%
    [ 19, 20)  13   4.33%  95.67% #
    [ 20, 21)   0   0.00%  95.67%
    [ 21, 22)   9   3.00%  98.67% #
    [ 22, 23)   0   0.00%  98.67%
    [ 23, 24)   1   0.33%  99.00%
    [ 24, 25)   0   0.00%  99.00%
    [ 25, 25]   3   1.00% 100.00%
    
    Depth by leafs:
    Count: 2224 Average: 3.2473 StdDev: 1.06404
    Min: 1 Max: 7 Ignored: 0
    ----------------------------------------------
    [ 1, 2)  45   2.02%   2.02% #
    [ 2, 3) 557  25.04%  27.07% ########
    [ 3, 4) 728  32.73%  59.80% ##########
    [ 4, 5) 642  28.87%  88.67% #########
    [ 5, 6) 207   9.31%  97.98% ###
    [ 6, 7)  39   1.75%  99.73% #
    [ 7, 7]   6   0.27% 100.00%
    
    Number of training obs by leaf:
    Count: 2224 Average: 33.3183 StdDev: 34.0985
    Min: 5 Max: 121 Ignored: 0
    ----------------------------------------------
    [   5,  10) 1067  47.98%  47.98% ##########
    [  10,  16)  103   4.63%  52.61% #
    [  16,  22)   53   2.38%  54.99%
    [  22,  28)   68   3.06%  58.05% #
    [  28,  34)   90   4.05%  62.10% #
    [  34,  40)  104   4.68%  66.77% #
    [  40,  45)   75   3.37%  70.14% #
    [  45,  51)   59   2.65%  72.80% #
    [  51,  57)   32   1.44%  74.24%
    [  57,  63)   23   1.03%  75.27%
    [  63,  69)   37   1.66%  76.93%
    [  69,  75)   50   2.25%  79.18%
    [  75,  81)   69   3.10%  82.28% #
    [  81,  86)   74   3.33%  85.61% #
    [  86,  92)  118   5.31%  90.92% #
    [  92,  98)   90   4.05%  94.96% #
    [  98, 104)   68   3.06%  98.02% #
    [ 104, 110)   28   1.26%  99.28%
    [ 110, 116)   12   0.54%  99.82%
    [ 116, 121]    4   0.18% 100.00%
    
    Attribute in nodes:
    	634 : bill_length_mm [NUMERICAL]
    	372 : flipper_length_mm [NUMERICAL]
    	338 : bill_depth_mm [NUMERICAL]
    	284 : island [CATEGORICAL]
    	253 : body_mass_g [NUMERICAL]
    	25 : sex [CATEGORICAL]
    	18 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 0:
    	163 : flipper_length_mm [NUMERICAL]
    	83 : bill_length_mm [NUMERICAL]
    	32 : bill_depth_mm [NUMERICAL]
    	15 : island [CATEGORICAL]
    	7 : body_mass_g [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	245 : flipper_length_mm [NUMERICAL]
    	238 : bill_length_mm [NUMERICAL]
    	161 : island [CATEGORICAL]
    	156 : bill_depth_mm [NUMERICAL]
    	54 : body_mass_g [NUMERICAL]
    	1 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	429 : bill_length_mm [NUMERICAL]
    	319 : flipper_length_mm [NUMERICAL]
    	253 : bill_depth_mm [NUMERICAL]
    	248 : island [CATEGORICAL]
    	147 : body_mass_g [NUMERICAL]
    	9 : sex [CATEGORICAL]
    	3 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 3:
    	574 : bill_length_mm [NUMERICAL]
    	355 : flipper_length_mm [NUMERICAL]
    	316 : bill_depth_mm [NUMERICAL]
    	278 : island [CATEGORICAL]
    	228 : body_mass_g [NUMERICAL]
    	24 : sex [CATEGORICAL]
    	11 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 5:
    	631 : bill_length_mm [NUMERICAL]
    	372 : flipper_length_mm [NUMERICAL]
    	338 : bill_depth_mm [NUMERICAL]
    	284 : island [CATEGORICAL]
    	253 : body_mass_g [NUMERICAL]
    	25 : sex [CATEGORICAL]
    	18 : year [NUMERICAL]
    
    Condition type in nodes:
    	1615 : HigherCondition
    	309 : ContainsBitmapCondition
    Condition type in nodes with depth <= 0:
    	285 : HigherCondition
    	15 : ContainsBitmapCondition
    Condition type in nodes with depth <= 1:
    	694 : HigherCondition
    	161 : ContainsBitmapCondition
    Condition type in nodes with depth <= 2:
    	1151 : HigherCondition
    	257 : ContainsBitmapCondition
    Condition type in nodes with depth <= 3:
    	1484 : HigherCondition
    	302 : ContainsBitmapCondition
    Condition type in nodes with depth <= 5:
    	1612 : HigherCondition
    	309 : ContainsBitmapCondition
    Node format: NOT_SET
    
    Training OOB:
    	trees: 1, Out-of-bag evaluation: accuracy:0.965909 logloss:1.22876
    	trees: 11, Out-of-bag evaluation: accuracy:0.959016 logloss:0.0722883
    	trees: 21, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0607441
    	trees: 31, Out-of-bag evaluation: accuracy:0.97166 logloss:0.068951
    	trees: 42, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0707371
    	trees: 52, Out-of-bag evaluation: accuracy:0.975708 logloss:0.068778
    	trees: 62, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0695978
    	trees: 72, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0719713
    	trees: 82, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0710908
    	trees: 92, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0712041
    	trees: 102, Out-of-bag evaluation: accuracy:0.975708 logloss:0.073906
    	trees: 112, Out-of-bag evaluation: accuracy:0.975708 logloss:0.073205
    	trees: 122, Out-of-bag evaluation: accuracy:0.97166 logloss:0.0745004
    	trees: 132, Out-of-bag evaluation: accuracy:0.97166 logloss:0.0732437
    	trees: 143, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0737169
    	trees: 154, Out-of-bag evaluation: accuracy:0.967611 logloss:0.0738033
    	trees: 164, Out-of-bag evaluation: accuracy:0.963563 logloss:0.0735375
    	trees: 174, Out-of-bag evaluation: accuracy:0.967611 logloss:0.0741928
    	trees: 184, Out-of-bag evaluation: accuracy:0.97166 logloss:0.073243
    	trees: 194, Out-of-bag evaluation: accuracy:0.97166 logloss:0.0741288
    	trees: 204, Out-of-bag evaluation: accuracy:0.97166 logloss:0.0739353
    	trees: 215, Out-of-bag evaluation: accuracy:0.97166 logloss:0.0732281
    	trees: 225, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0738094
    	trees: 235, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0738642
    	trees: 246, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0747816
    	trees: 256, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0747483
    	trees: 266, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0755402
    	trees: 276, Out-of-bag evaluation: accuracy:0.975708 logloss:0.076221
    	trees: 286, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0755566
    	trees: 296, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0755158
    	trees: 300, Out-of-bag evaluation: accuracy:0.975708 logloss:0.0753986


<a id="org83ba3ba"></a>

# Using make<sub>inspector</sub>

    '("bill_depth_mm" (1; #1) 
     "bill_length_mm" (1; #2) 
     "body_mass_g" (1; #3) 
     "flipper_length_mm" (1; #4) 
     "island" (4; #5) 
     "sex" (4; #6) 
     "year" (1; #7))

    '("INV_MEAN_MIN_DEPTH": (("flipper_length_mm" (1; #4)  0.4576590394689772) 
      ("bill_length_mm" (1; #2)  0.4344717370564895) 
      ("island" (4; #5)  0.32408522308201926) 
      ("bill_depth_mm" (1; #1)  0.29640216741411735) 
      ("body_mass_g" (1; #3)  0.27381252067752093) 
      ("year" (1; #7)  0.24145786518039727) 
      ("sex" (4; #6)  0.241434060971869)) 
     "NUM_AS_ROOT": (("flipper_length_mm" (1; #4)  163.0) 
      ("bill_length_mm" (1; #2)  83.0) 
      ("bill_depth_mm" (1; #1)  32.0) 
      ("island" (4; #5)  15.0) 
      ("body_mass_g" (1; #3)  7.0)) 
     "SUM_SCORE": (("flipper_length_mm" (1; #4)  26397.21139154397) 
      ("bill_length_mm" (1; #2)  26229.49720552191) 
      ("island" (4; #5)  11178.969668364152) 
      ("bill_depth_mm" (1; #1)  7427.787269951776) 
      ("body_mass_g" (1; #3)  3074.5287622250617) 
      ("sex" (4; #6)  168.3260055705905) 
      ("year" (1; #7)  64.97349715605378)) 
     "NUM_NODES": (("bill_length_mm" (1; #2)  634.0) 
      ("flipper_length_mm" (1; #4)  372.0) 
      ("bill_depth_mm" (1; #1)  338.0) 
      ("island" (4; #5)  284.0) 
      ("body_mass_g" (1; #3)  253.0) 
      ("sex" (4; #6)  25.0) 
      ("year" (1; #7)  18.0)))


<a id="org692b22c"></a>

# Model self evaluation

    Evaluation(num_examples=247, accuracy=0.9757085020242915, loss=0.07539860509818623, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)


<a id="orge4e78ec"></a>

# Plotting the training logs

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=1 evaluation=Evaluation (num<sub>examples</sub>=88 accuracy=0.9659090909090909 loss=1.2287608493458142 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=11 evaluation=Evaluation (num<sub>examples</sub>=244 accuracy=0.9590163934426229 loss=0.07228826939082536 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=21 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.0607441041452682 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=31 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.97165991902834 loss=0.06895102862163112 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=42 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07073711700405669 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=52 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.06877801988107955 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=62 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.06959785142468537 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=72 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07197134801552363 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=82 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07109075393301514 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=92 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07120410751114008 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=102 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07390605097991011 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=112 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07320502791025861 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=122 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.97165991902834 loss=0.07450044588732575 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=132 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.97165991902834 loss=0.0732436667381209 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=143 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07371688680911836 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=154 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9676113360323887 loss=0.07380332788716444 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=164 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9635627530364372 loss=0.07353750742247954 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=174 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9676113360323887 loss=0.07419281175005653 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=184 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.97165991902834 loss=0.07324301531002951 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=194 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.97165991902834 loss=0.07412883250216241 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=204 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.97165991902834 loss=0.07393531512227738 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=215 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.97165991902834 loss=0.07322807853611615 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=225 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07380938008880085 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=235 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.0738641716057231 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=246 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07478160741041426 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=256 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07474832479589381 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=266 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07554018897143935 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=276 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07622103342552658 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=286 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07555657345801592 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=296 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07551582798980146 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=300 evaluation=Evaluation (num<sub>examples</sub>=247 accuracy=0.9757085020242915 loss=0.07539860509818623 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
</tr>
</tbody>
</table>

![img](./.ob-jupyter/4b20350eca12d59627286ccb3c305707be9dfdb8.png)


<a id="org76f07c5"></a>

# Retrain model with different learning algorithm

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">tensorflow<sub>decision</sub><sub>forests.keras.RandomForestModel</sub></td>
<td class="org-left">tensorflow<sub>decision</sub><sub>forests.keras.GradientBoostedTreesModel</sub></td>
<td class="org-left">tensorflow<sub>decision</sub><sub>forests.keras.CartModel</sub></td>
<td class="org-left">tensorflow<sub>decision</sub><sub>forests.keras.DistributedGradientBoostedTreesModel</sub></td>
</tr>
</tbody>
</table>


<a id="orgcfc9a6d"></a>

# Using a subset of features

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpanre1cxb as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.078216. Found 247 examples.
    Reading validation dataset...
    [WARNING 23-05-21 17:26:58.5505 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:26:58.5505 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:26:58.5505 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Num validation examples: tf.Tensor(97, shape=(), dtype=int32)
    Validation dataset read in 0:00:00.174504. Found 97 examples.
    Training model...
    Model trained in 0:00:00.053095
    Compiling model...
    Model compiled.
    1/1 [==============================] - 0s 47ms/step - loss: 0.0000e+00 - accuracy: 0.9485
    {'loss': 0.0, 'accuracy': 0.9484536051750183}
    [INFO 23-05-21 17:26:58.8620 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpanre1cxb/model/ with prefix 1047d116e8ed41f8
    [INFO 23-05-21 17:26:58.8650 CDT decision_forest.cc:660] Model loaded with 81 root(s), 2417 node(s), and 2 input feature(s).
    [INFO 23-05-21 17:26:58.8650 CDT kernel.cc:1074] Use fast generic engine

**TF-DF** attaches a **semantics** to each feature. This semantics controls how the feature is used by the model. The following semantics are currently supported.

-   **Numerical**: Generally for quantities or counts with full ordering. For example, the age of a person, or the number of items in a bag. Can be a float or an integer. Missing values are represented with a float(Nan) or with an empty sparse tensor.
-   **Categorical**: Generally for a type/class in finite set of possible values without ordering. For example, the color RED in the set {RED, BLUE, GREEN}. Can be a string or an integer. Missing values are represented as &ldquo;&rdquo; (empty string), value -2 or with an empty sparse tensor.
-   **Categorical-Set**: A set of categorical values. Great to represent tokenized text. Can be a string or an integer in a sparse tensor or a ragged tensor (recommended). The order/index of each item doesnt matter.
    
    If not specified, the semantics is inferred from the representation type and shown in the training logs:
    
    -   int, float (dense or sparse) -> Numerical semantics
    
    -   str, (dense or sparse) -> Categorical semantics
    
    -   int, str (ragged) -> Categorical-Set semantics

In some cases, the inferred semantics is incorrect. For example: An Enum stored as an integer is semantically categorical, but it will be detected as numerical. In this case, you should specify the semantic argument in the input. The education<sub>num</sub> field of the Adult dataset is a classic example.

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpo_i4bm_e as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.076355. Found 247 examples.
    Reading validation dataset...
    Num validation examples: tf.Tensor(97, shape=(), dtype=int32)
    Validation dataset read in 0:00:00.062893. Found 97 examples.
    Training model...
    Model trained in 0:00:00.047812
    Compiling model...
    [WARNING 23-05-21 17:26:59.1559 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:26:59.1560 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:26:59.1560 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    [INFO 23-05-21 17:26:59.3480 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpo_i4bm_e/model/ with prefix c6f3952553e94287
    [INFO 23-05-21 17:26:59.3494 CDT decision_forest.cc:660] Model loaded with 42 root(s), 1108 node(s), and 3 input feature(s).
    [INFO 23-05-21 17:26:59.3495 CDT kernel.cc:1074] Use fast generic engine
    Model compiled.

    <keras.callbacks.History at 0x2c3423ca0>

Note that `year` is in the list of CATEGORICAL features (unlike the first run)


<a id="org690b29e"></a>

# Hyper-parameters

**Hyper-parameters** are paramters of the training algorithm that impact the quality of the final model. They are specified in the model class constructor. The list of hyper-parameters is visible with the *question mark* colab command.

**I will figure out how to obtain that list without the question mark command.**

    Model: "gradient_boosted_trees_model_17"
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
        1.    "bill_length_mm"  0.340541 ################
        2. "flipper_length_mm"  0.334855 ###############
        3.     "bill_depth_mm"  0.289173 ###########
        4.            "island"  0.247452 ########
        5.       "body_mass_g"  0.246817 ########
        6.              "year"  0.150871 
        7.               "sex"  0.150066 
    
    Variable Importance: NUM_AS_ROOT:
        1.    "bill_length_mm" 25.000000 
        2. "flipper_length_mm" 25.000000 
        3.            "island" 25.000000 
    
    Variable Importance: NUM_NODES:
        1.    "bill_length_mm" 453.000000 ################
        2.     "bill_depth_mm" 432.000000 ###############
        3. "flipper_length_mm" 367.000000 ############
        4.       "body_mass_g" 316.000000 ###########
        5.            "island" 63.000000 ##
        6.              "year" 23.000000 
        7.               "sex"  6.000000 
    
    Variable Importance: SUM_SCORE:
        1.    "bill_length_mm" 312.318797 ################
        2. "flipper_length_mm" 222.262171 ###########
        3.            "island" 95.441697 ####
        4.     "bill_depth_mm" 13.794965 
        5.               "sex"  3.582136 
        6.       "body_mass_g"  1.231969 
        7.              "year"  0.000086 
    
    
    
    Loss: MULTINOMIAL_LOG_LIKELIHOOD
    Validation loss value: 0.170693
    Number of trees per iteration: 3
    Node format: NOT_SET
    Number of trees: 75
    Total number of nodes: 3395
    
    Number of nodes by tree:
    Count: 75 Average: 45.2667 StdDev: 9.64065
    Min: 11 Max: 55 Ignored: 0
    ----------------------------------------------
    [ 11, 13)  1   1.33%   1.33% #
    [ 13, 15)  2   2.67%   4.00% #
    [ 15, 17)  0   0.00%   4.00%
    [ 17, 20)  1   1.33%   5.33% #
    [ 20, 22)  0   0.00%   5.33%
    [ 22, 24)  1   1.33%   6.67% #
    [ 24, 26)  1   1.33%   8.00% #
    [ 26, 29)  0   0.00%   8.00%
    [ 29, 31)  0   0.00%   8.00%
    [ 31, 33)  1   1.33%   9.33% #
    [ 33, 35)  2   2.67%  12.00% #
    [ 35, 38)  0   0.00%  12.00%
    [ 38, 40)  1   1.33%  13.33% #
    [ 40, 42)  5   6.67%  20.00% ###
    [ 42, 44)  5   6.67%  26.67% ###
    [ 44, 47)  7   9.33%  36.00% ####
    [ 47, 49)  5   6.67%  42.67% ###
    [ 49, 51) 16  21.33%  64.00% ##########
    [ 51, 53) 16  21.33%  85.33% ##########
    [ 53, 55] 11  14.67% 100.00% #######
    
    Depth by leafs:
    Count: 1735 Average: 5.79885 StdDev: 1.85168
    Min: 1 Max: 8 Ignored: 0
    ----------------------------------------------
    [ 1, 2)   1   0.06%   0.06%
    [ 2, 3) 119   6.86%   6.92% ###
    [ 3, 4) 138   7.95%  14.87% ###
    [ 4, 5) 162   9.34%  24.21% ####
    [ 5, 6) 263  15.16%  39.37% #######
    [ 6, 7) 321  18.50%  57.87% ########
    [ 7, 8) 329  18.96%  76.83% ########
    [ 8, 8] 402  23.17% 100.00% ##########
    
    Number of training obs by leaf:
    Count: 1735 Average: 0 StdDev: 0
    Min: 0 Max: 0 Ignored: 0
    ----------------------------------------------
    [ 0, 0] 1735 100.00% 100.00% ##########
    
    Attribute in nodes:
    	453 : bill_length_mm [NUMERICAL]
    	432 : bill_depth_mm [NUMERICAL]
    	367 : flipper_length_mm [NUMERICAL]
    	316 : body_mass_g [NUMERICAL]
    	63 : island [CATEGORICAL]
    	23 : year [NUMERICAL]
    	6 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 0:
    	25 : island [CATEGORICAL]
    	25 : flipper_length_mm [NUMERICAL]
    	25 : bill_length_mm [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	55 : bill_length_mm [NUMERICAL]
    	52 : flipper_length_mm [NUMERICAL]
    	43 : bill_depth_mm [NUMERICAL]
    	40 : island [CATEGORICAL]
    	34 : body_mass_g [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	106 : bill_depth_mm [NUMERICAL]
    	88 : flipper_length_mm [NUMERICAL]
    	84 : body_mass_g [NUMERICAL]
    	84 : bill_length_mm [NUMERICAL]
    	40 : island [CATEGORICAL]
    	1 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 3:
    	168 : bill_depth_mm [NUMERICAL]
    	156 : bill_length_mm [NUMERICAL]
    	142 : flipper_length_mm [NUMERICAL]
    	106 : body_mass_g [NUMERICAL]
    	49 : island [CATEGORICAL]
    	2 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 5:
    	341 : bill_length_mm [NUMERICAL]
    	291 : bill_depth_mm [NUMERICAL]
    	266 : flipper_length_mm [NUMERICAL]
    	221 : body_mass_g [NUMERICAL]
    	59 : island [CATEGORICAL]
    	10 : year [NUMERICAL]
    	6 : sex [CATEGORICAL]
    
    Condition type in nodes:
    	1591 : HigherCondition
    	69 : ContainsBitmapCondition
    Condition type in nodes with depth <= 0:
    	50 : HigherCondition
    	25 : ContainsBitmapCondition
    Condition type in nodes with depth <= 1:
    	184 : HigherCondition
    	40 : ContainsBitmapCondition
    Condition type in nodes with depth <= 2:
    	362 : HigherCondition
    	41 : ContainsBitmapCondition
    Condition type in nodes with depth <= 3:
    	572 : HigherCondition
    	51 : ContainsBitmapCondition
    Condition type in nodes with depth <= 5:
    	1129 : HigherCondition
    	65 : ContainsBitmapCondition
    
    Training logs:
    Number of iteration to final model: 25
    	Iter:1 train-loss:0.913222 valid-loss:0.929066  train-accuracy:1.000000 valid-accuracy:0.896552
    	Iter:2 train-loss:0.768881 valid-loss:0.796872  train-accuracy:1.000000 valid-accuracy:0.931035
    	Iter:3 train-loss:0.653148 valid-loss:0.691194  train-accuracy:1.000000 valid-accuracy:0.931035
    	Iter:4 train-loss:0.558519 valid-loss:0.605474  train-accuracy:1.000000 valid-accuracy:0.931035
    	Iter:5 train-loss:0.480033 valid-loss:0.534116  train-accuracy:1.000000 valid-accuracy:0.931035
    	Iter:6 train-loss:0.414281 valid-loss:0.475021  train-accuracy:1.000000 valid-accuracy:0.896552
    	Iter:16 train-loss:0.107564 valid-loss:0.206812  train-accuracy:1.000000 valid-accuracy:0.896552
    	Iter:26 train-loss:0.030898 valid-loss:0.170750  train-accuracy:1.000000 valid-accuracy:0.896552

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp9rub46v9 as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.083896. Found 247 examples.
    Training model...
    [WARNING 23-05-21 17:26:59.5442 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:26:59.5442 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:26:59.5442 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:00.133133
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:26:59.7623 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp9rub46v9/model/ with prefix d4f45eacce7c441c
    [INFO 23-05-21 17:26:59.7665 CDT decision_forest.cc:660] Model loaded with 75 root(s), 3395 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:26:59.7665 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x29f1c58e0>

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpyt3_qjez as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.091578. Found 247 examples.
    Training model...
    [WARNING 23-05-21 17:27:00.0616 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:27:00.0617 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:27:00.0617 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:00.271622
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:27:00.4269 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpyt3_qjez/model/ with prefix d0c447ce391e4f08
    [INFO 23-05-21 17:27:00.4330 CDT decision_forest.cc:660] Model loaded with 102 root(s), 4654 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:27:00.4330 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x29f183880>

As new training methods are published and implemented, combinations of hyper-parameters can emerge as good or almost-always-better than the default parameters. To avoid changing the default hyper-parameter values these good combinations are indexed and availale as hyper-parameter templates.

For example, the benchmark<sub>rank1</sub> template is the best combination on our internal benchmarks. Those templates are versioned to allow training configuration stability e.g. benchmark<sub>rank1</sub>@v1.

    Resolve hyper-parameter template "benchmark_rank1" to "benchmark_rank1@v1" -> {'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}.
    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp8hejwd1a as temporary training directory
    Reading training dataset...
    [WARNING 23-05-21 17:27:00.5716 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:27:00.5716 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:27:00.5716 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Training dataset read in 0:00:00.082458. Found 247 examples.
    Training model...
    Model trained in 0:00:00.132112
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:27:00.7868 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp8hejwd1a/model/ with prefix c7d5226b21884084
    [INFO 23-05-21 17:27:00.7911 CDT decision_forest.cc:660] Model loaded with 84 root(s), 3280 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:27:00.7912 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2b1152c10>

The available templates are available with `predefined_hyperparameters`. Note that different learning algorithms have different templates, even if the name is similar.

    [HyperParameterTemplate(name='better_default', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL'}, description='A configuration that is generally better than the default parameters without being more expensive.'), HyperParameterTemplate(name='benchmark_rank1', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}, description='Top ranking hyper-parameters on our benchmark slightly modified to run in reasonable time.')]

What is returned are the predefined hyper-parameters of the Gradient Boosted Tree model.


<a id="orgb96aed6"></a>

# Feature Preprocessing

Pre-processing features is sometimes necessary to consume signals with complex structures, to regularize the model or to apply transfer learning. Pre-processing can be done in one of three ways:

1.  **Preprocessing on the pandas dataframe**: This solution is easy tto implement and generally suitable for experiementation. However, the pre-processing logic will not be exported in the model by model.save()
2.  **Keras Preprocessing**: While more complex than the previous solution, Keras Preprocessing is packaged in the model.
3.  **TensorFlow Feature Columns**: This API is part of the TF Estimator library (!= Keras) and planned for deprecation. This solution is interesting when using existing preprocessing code.

**Note**: Using **TensorFlow Hub** pre-trained embedding is often, a great way to consume text and image with TF-DF.

In the next example, pre-process the body<sub>mass</sub><sub>g</sub> feature into body<sub>mass</sub><sub>kg</sub> = body<sub>mass</sub><sub>g</sub> / 1000. The bill<sub>length</sub><sub>mm</sub> is consumed without preprocessing. Note that such monotonic transformations have generally no impact on decision forest models.

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmplzn43s6f as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.088356. Found 247 examples.
    Training model...
    Model trained in 0:00:00.021911
    Compiling model...
    Model compiled.
    Model: "random_forest_model_13"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model_3 (Functional)        {'body_mass_kg': (None,   0         
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
        1. "bill_length_mm"  0.991080 ################
        2.   "body_mass_kg"  0.474044 
    
    Variable Importance: NUM_AS_ROOT:
        1. "bill_length_mm" 298.000000 ################
        2.   "body_mass_kg"  2.000000 
    
    Variable Importance: NUM_NODES:
        1. "bill_length_mm" 1530.000000 ################
        2.   "body_mass_kg" 1149.000000 
    
    Variable Importance: SUM_SCORE:
        1. "bill_length_mm" 46282.464302 ################
        2.   "body_mass_kg" 25498.658782 
    
    
    
    Winner takes all: true
    Out-of-bag evaluation: accuracy:0.91498 logloss:0.213831
    Number of trees: 300
    Total number of nodes: 5658
    
    Number of nodes by tree:
    Count: 300 Average: 18.86 StdDev: 2.56523
    Min: 13 Max: 25 Ignored: 0
    ----------------------------------------------
    [ 13, 14)  6   2.00%   2.00% #
    [ 14, 15)  0   0.00%   2.00%
    [ 15, 16) 35  11.67%  13.67% ####
    [ 16, 17)  0   0.00%  13.67%
    [ 17, 18) 67  22.33%  36.00% #######
    [ 18, 19)  0   0.00%  36.00%
    [ 19, 20) 98  32.67%  68.67% ##########
    [ 20, 21)  0   0.00%  68.67%
    [ 21, 22) 64  21.33%  90.00% #######
    [ 22, 23)  0   0.00%  90.00%
    [ 23, 24) 20   6.67%  96.67% ##
    [ 24, 25)  0   0.00%  96.67%
    [ 25, 25] 10   3.33% 100.00% #
    
    Depth by leafs:
    Count: 2979 Average: 4.17086 StdDev: 1.66697
    Min: 1 Max: 9 Ignored: 0
    ----------------------------------------------
    [ 1, 2) 165   5.54%   5.54% ##
    [ 2, 3) 274   9.20%  14.74% ####
    [ 3, 4) 633  21.25%  35.99% #########
    [ 4, 5) 728  24.44%  60.42% ##########
    [ 5, 6) 536  17.99%  78.42% #######
    [ 6, 7) 378  12.69%  91.10% #####
    [ 7, 8) 183   6.14%  97.25% ###
    [ 8, 9)  66   2.22%  99.46% #
    [ 9, 9]  16   0.54% 100.00%
    
    Number of training obs by leaf:
    Count: 2979 Average: 24.8741 StdDev: 30.7881
    Min: 5 Max: 119 Ignored: 0
    ----------------------------------------------
    [   5,  10) 1868  62.71%  62.71% ##########
    [  10,  16)  185   6.21%  68.92% #
    [  16,  22)   22   0.74%  69.65%
    [  22,  28)   26   0.87%  70.53%
    [  28,  33)   56   1.88%  72.41%
    [  33,  39)  102   3.42%  75.83% #
    [  39,  45)   88   2.95%  78.78%
    [  45,  51)   42   1.41%  80.19%
    [  51,  56)   22   0.74%  80.93%
    [  56,  62)   65   2.18%  83.12%
    [  62,  68)   83   2.79%  85.90%
    [  68,  74)   69   2.32%  88.22%
    [  74,  79)   36   1.21%  89.43%
    [  79,  85)   27   0.91%  90.33%
    [  85,  91)   43   1.44%  91.78%
    [  91,  97)   88   2.95%  94.73%
    [  97, 102)   75   2.52%  97.25%
    [ 102, 108)   57   1.91%  99.16%
    [ 108, 114)   20   0.67%  99.83%
    [ 114, 119]    5   0.17% 100.00%
    
    Attribute in nodes:
    	1530 : bill_length_mm [NUMERICAL]
    	1149 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 0:
    	298 : bill_length_mm [NUMERICAL]
    	2 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	403 : bill_length_mm [NUMERICAL]
    	332 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	689 : body_mass_kg [NUMERICAL]
    	642 : bill_length_mm [NUMERICAL]
    
    Attribute in nodes with depth <= 3:
    	1032 : bill_length_mm [NUMERICAL]
    	858 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 5:
    	1425 : bill_length_mm [NUMERICAL]
    	1099 : body_mass_kg [NUMERICAL]
    
    Condition type in nodes:
    	2679 : HigherCondition
    Condition type in nodes with depth <= 0:
    	300 : HigherCondition
    Condition type in nodes with depth <= 1:
    	735 : HigherCondition
    Condition type in nodes with depth <= 2:
    	1331 : HigherCondition
    Condition type in nodes with depth <= 3:
    	1890 : HigherCondition
    Condition type in nodes with depth <= 5:
    	2524 : HigherCondition
    Node format: NOT_SET
    
    Training OOB:
    	trees: 1, Out-of-bag evaluation: accuracy:0.838384 logloss:5.82524
    	trees: 11, Out-of-bag evaluation: accuracy:0.894958 logloss:1.89779
    	trees: 24, Out-of-bag evaluation: accuracy:0.910931 logloss:1.15036
    	trees: 35, Out-of-bag evaluation: accuracy:0.919028 logloss:0.73776
    	trees: 49, Out-of-bag evaluation: accuracy:0.919028 logloss:0.72829
    	trees: 61, Out-of-bag evaluation: accuracy:0.919028 logloss:0.461881
    	trees: 74, Out-of-bag evaluation: accuracy:0.91498 logloss:0.461553
    	trees: 86, Out-of-bag evaluation: accuracy:0.919028 logloss:0.462821
    	trees: 96, Out-of-bag evaluation: accuracy:0.919028 logloss:0.464891
    	trees: 109, Out-of-bag evaluation: accuracy:0.919028 logloss:0.337525
    	trees: 119, Out-of-bag evaluation: accuracy:0.919028 logloss:0.337174
    	trees: 130, Out-of-bag evaluation: accuracy:0.919028 logloss:0.336495
    	trees: 143, Out-of-bag evaluation: accuracy:0.919028 logloss:0.335129
    	trees: 155, Out-of-bag evaluation: accuracy:0.91498 logloss:0.337151
    	trees: 167, Out-of-bag evaluation: accuracy:0.919028 logloss:0.339358
    	trees: 177, Out-of-bag evaluation: accuracy:0.91498 logloss:0.338962
    	trees: 189, Out-of-bag evaluation: accuracy:0.919028 logloss:0.338844
    	trees: 201, Out-of-bag evaluation: accuracy:0.919028 logloss:0.339388
    	trees: 212, Out-of-bag evaluation: accuracy:0.919028 logloss:0.338146
    	trees: 222, Out-of-bag evaluation: accuracy:0.919028 logloss:0.336514
    	trees: 232, Out-of-bag evaluation: accuracy:0.919028 logloss:0.337595
    	trees: 244, Out-of-bag evaluation: accuracy:0.919028 logloss:0.337128
    	trees: 256, Out-of-bag evaluation: accuracy:0.91498 logloss:0.338151
    	trees: 267, Out-of-bag evaluation: accuracy:0.91498 logloss:0.340114
    	trees: 277, Out-of-bag evaluation: accuracy:0.91498 logloss:0.340115
    	trees: 287, Out-of-bag evaluation: accuracy:0.91498 logloss:0.340192
    	trees: 297, Out-of-bag evaluation: accuracy:0.91498 logloss:0.214053
    	trees: 300, Out-of-bag evaluation: accuracy:0.91498 logloss:0.213831
    /Users/umbertofasci/miniforge3/envs/tensorflow-metal/lib/python3.9/site-packages/keras/engine/functional.py:639: UserWarning: Input dict contained keys ['island', 'bill_depth_mm', 'flipper_length_mm', 'sex', 'year'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)
    [INFO 23-05-21 17:27:01.1862 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmplzn43s6f/model/ with prefix cc1ad616e3db407b
    [INFO 23-05-21 17:27:01.1930 CDT decision_forest.cc:660] Model loaded with 300 root(s), 5658 node(s), and 2 input feature(s).
    [INFO 23-05-21 17:27:01.1930 CDT kernel.cc:1074] Use fast generic engine

The following example re-implements the same logic using TensorFlow Feature Columns.

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpl7lemwn1 as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.084385. Found 247 examples.
    Training model...
    Model trained in 0:00:00.020940
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:27:01.4409 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpl7lemwn1/model/ with prefix 768acd454e1a43f0
    [INFO 23-05-21 17:27:01.4480 CDT decision_forest.cc:660] Model loaded with 300 root(s), 5658 node(s), and 2 input feature(s).
    [INFO 23-05-21 17:27:01.4480 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2a9e0e0a0>


<a id="orgccccc74"></a>

# Training a regression model

The previous example trains a classification model(TF-DF does not differentiate between binary classification and multi-class classification). In the next example, train a regression model on the Abalone dataset. The objective of this dataset is to predict the number of rings on a shell of a abalone.

**Note**: The csv file is assembled by appending UCI&rsquo;s header and data files. No preprocessing was applied.

      Type  LongestShell  Diameter  Height  WholeWeight  ShuckedWeight   
    0    M         0.455     0.365   0.095       0.5140         0.2245  \
    1    M         0.350     0.265   0.090       0.2255         0.0995   
    2    F         0.530     0.420   0.135       0.6770         0.2565   
    
       VisceraWeight  ShellWeight  Rings  
    0         0.1010         0.15     15  
    1         0.0485         0.07      7  
    2         0.1415         0.21      9  

    2918 examples in training, 1259 examples for testing.

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpkzmqfatu as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.102875. Found 2918 examples.
    Training model...
    [INFO 23-05-21 17:27:02.8524 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpkzmqfatu/model/ with prefix 7326197700f546af
    Model trained in 0:00:00.701618
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:27:03.1595 CDT decision_forest.cc:660] Model loaded with 300 root(s), 260236 node(s), and 8 input feature(s).
    [INFO 23-05-21 17:27:03.1595 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2c342a070>

    2/2 [==============================] - 0s 16ms/step - loss: 0.0000e+00 - mse: 4.1789
    
    {'loss': 0.0, 'mse': 4.1789445877075195}
    
    MSE: 4.1789445877075195
    RMSE: 2.0442467042183337


<a id="orgb84ff1d"></a>

# Conclusion

This concludes the basic overview of TensorFlow Decision Forest utility.

