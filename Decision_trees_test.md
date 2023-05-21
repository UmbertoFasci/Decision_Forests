
# Table of Contents

1.  [Importing Libraries](#org5d705cd)
2.  [Training a Random Forest model](#org30ebaa5)
3.  [Evaluate the model](#org762da3d)
4.  [TensorFlow Serving](#org7191025)
5.  [Model structure and feature importance](#orgdf60811)
6.  [Using make<sub>inspector</sub>](#orgdfdf6a6)
7.  [Model self evaluation](#org87e5eeb)
8.  [Plotting the training logs](#orgbf7d12b)
9.  [Retrain model with different learning algorithm](#org0cc95b6)
10. [Using a subset of features](#org0ca1421)
11. [Hyper-parameters](#org8ee6c44)
12. [Feature Preprocessing](#org2ebf064)
13. [Training a regression model](#org5e31bf5)
14. [Conclusion](#org224545f)

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


<a id="org5d705cd"></a>

# Importing Libraries

    import tensorflow_decision_forests as tfdf
    
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import math

    Found TensorFlow Decision Forests v1.3.0


<a id="org30ebaa5"></a>

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

    243 examples in training, 101 examples for testing.

    Use 8 thread(s) for training
    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpzc6rc0x8 as temporary training directory
    Reading training dataset...
    Training tensor examples:
    Features: {'island': <tf.Tensor 'data:0' shape=(None,) dtype=string>, 'bill_length_mm': <tf.Tensor 'data_1:0' shape=(None,) dtype=float64>, 'bill_depth_mm': <tf.Tensor 'data_2:0' shape=(None,) dtype=float64>, 'flipper_length_mm': <tf.Tensor 'data_3:0' shape=(None,) dtype=float64>, 'body_mass_g': <tf.Tensor 'data_4:0' shape=(None,) dtype=float64>, 'sex': <tf.Tensor 'data_5:0' shape=(None,) dtype=string>, 'year': <tf.Tensor 'data_6:0' shape=(None,) dtype=int64>}
    Label: Tensor("data_7:0", shape=(None,), dtype=int64)
    Weights: None
    Normalized tensor features:
     {'island': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data:0' shape=(None,) dtype=string>), 'bill_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast:0' shape=(None,) dtype=float32>), 'bill_depth_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_1:0' shape=(None,) dtype=float32>), 'flipper_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_2:0' shape=(None,) dtype=float32>), 'body_mass_g': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_3:0' shape=(None,) dtype=float32>), 'sex': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data_5:0' shape=(None,) dtype=string>), 'year': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_4:0' shape=(None,) dtype=float32>)}
    Training dataset read in 0:00:00.099806. Found 243 examples.
    Training model...
    [INFO 23-05-21 17:18:02.2464 CDT kernel.cc:773] Start Yggdrasil model training
    [INFO 23-05-21 17:18:02.2464 CDT kernel.cc:774] Collect training examples
    [INFO 23-05-21 17:18:02.2464 CDT kernel.cc:787] Dataspec guide:
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
    
    [INFO 23-05-21 17:18:02.2464 CDT kernel.cc:393] Number of batches: 1
    [INFO 23-05-21 17:18:02.2464 CDT kernel.cc:394] Number of examples: 243
    [INFO 23-05-21 17:18:02.2465 CDT kernel.cc:794] Training dataset:
    Number of records: 243
    Number of columns: 8
    
    Number of columns by type:
    	NUMERICAL: 5 (62.5%)
    	CATEGORICAL: 3 (37.5%)
    
    Columns:
    
    NUMERICAL: 5 (62.5%)
    	1: "bill_depth_mm" NUMERICAL num-nas:2 (0.823045%) mean:17.0859 min:13.1 max:21.5 sd:1.94967
    	2: "bill_length_mm" NUMERICAL num-nas:2 (0.823045%) mean:44.1822 min:32.1 max:59.6 sd:5.27779
    	3: "body_mass_g" NUMERICAL num-nas:2 (0.823045%) mean:4227.07 min:2850 max:6300 sd:813.69
    	4: "flipper_length_mm" NUMERICAL num-nas:2 (0.823045%) mean:201.411 min:174 max:231 sd:14.0663
    	7: "year" NUMERICAL mean:2008 min:2007 max:2009 sd:0.796081
    
    CATEGORICAL: 3 (37.5%)
    	0: "__LABEL" CATEGORICAL integerized vocab-size:4 no-ood-item
    	5: "island" CATEGORICAL has-dict vocab-size:4 zero-ood-items most-frequent:"Biscoe" 117 (48.1481%)
    	6: "sex" CATEGORICAL num-nas:9 (3.7037%) has-dict vocab-size:3 zero-ood-items most-frequent:"male" 121 (51.7094%)
    
    Terminology:
    	nas: Number of non-available (i.e. missing) values.
    	ood: Out of dictionary.
    	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
    	tokenized: The attribute value is obtained through tokenization.
    	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
    	vocab-size: Number of unique values.
    
    [INFO 23-05-21 17:18:02.2465 CDT kernel.cc:810] Configure learner
    [INFO 23-05-21 17:18:02.2466 CDT kernel.cc:824] Training config:
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
    
    [INFO 23-05-21 17:18:02.2466 CDT kernel.cc:827] Deployment config:
    cache_path: "/var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpzc6rc0x8/working_cache"
    num_threads: 8
    try_resume_training: true
    [INFO 23-05-21 17:18:02.2467 CDT kernel.cc:889] Train model
    [INFO 23-05-21 17:18:02.2467 CDT random_forest.cc:416] Training random forest on 243 example(s) and 7 feature(s).
    [INFO 23-05-21 17:18:02.2471 CDT random_forest.cc:805] Training of tree  1/300 (tree index:0) done accuracy:0.908163 logloss:3.31013
    [INFO 23-05-21 17:18:02.2473 CDT random_forest.cc:805] Training of tree  11/300 (tree index:10) done accuracy:0.941176 logloss:0.981108
    [INFO 23-05-21 17:18:02.2475 CDT random_forest.cc:805] Training of tree  21/300 (tree index:20) done accuracy:0.950617 logloss:0.385867
    [INFO 23-05-21 17:18:02.2478 CDT random_forest.cc:805] Training of tree  31/300 (tree index:31) done accuracy:0.962963 logloss:0.379752
    [INFO 23-05-21 17:18:02.2480 CDT random_forest.cc:805] Training of tree  41/300 (tree index:42) done accuracy:0.962963 logloss:0.235278
    [INFO 23-05-21 17:18:02.2483 CDT random_forest.cc:805] Training of tree  52/300 (tree index:51) done accuracy:0.971193 logloss:0.233288
    [INFO 23-05-21 17:18:02.2485 CDT random_forest.cc:805] Training of tree  62/300 (tree index:60) done accuracy:0.971193 logloss:0.234777
    [INFO 23-05-21 17:18:02.2488 CDT random_forest.cc:805] Training of tree  74/300 (tree index:77) done accuracy:0.975309 logloss:0.236124
    [INFO 23-05-21 17:18:02.2491 CDT random_forest.cc:805] Training of tree  84/300 (tree index:84) done accuracy:0.971193 logloss:0.234085
    [INFO 23-05-21 17:18:02.2493 CDT random_forest.cc:805] Training of tree  95/300 (tree index:94) done accuracy:0.971193 logloss:0.232744
    [INFO 23-05-21 17:18:02.2497 CDT random_forest.cc:805] Training of tree  106/300 (tree index:106) done accuracy:0.975309 logloss:0.231187
    [INFO 23-05-21 17:18:02.2499 CDT random_forest.cc:805] Training of tree  117/300 (tree index:118) done accuracy:0.975309 logloss:0.230766
    [INFO 23-05-21 17:18:02.2501 CDT random_forest.cc:805] Training of tree  127/300 (tree index:127) done accuracy:0.975309 logloss:0.231985
    [INFO 23-05-21 17:18:02.2504 CDT random_forest.cc:805] Training of tree  137/300 (tree index:138) done accuracy:0.975309 logloss:0.230148
    [INFO 23-05-21 17:18:02.2506 CDT random_forest.cc:805] Training of tree  147/300 (tree index:146) done accuracy:0.975309 logloss:0.230627
    [INFO 23-05-21 17:18:02.2508 CDT random_forest.cc:805] Training of tree  157/300 (tree index:155) done accuracy:0.975309 logloss:0.230122
    [INFO 23-05-21 17:18:02.2511 CDT random_forest.cc:805] Training of tree  168/300 (tree index:169) done accuracy:0.975309 logloss:0.231888
    [INFO 23-05-21 17:18:02.2513 CDT random_forest.cc:805] Training of tree  178/300 (tree index:181) done accuracy:0.975309 logloss:0.230603
    [INFO 23-05-21 17:18:02.2515 CDT random_forest.cc:805] Training of tree  188/300 (tree index:187) done accuracy:0.975309 logloss:0.229802
    [INFO 23-05-21 17:18:02.2517 CDT random_forest.cc:805] Training of tree  198/300 (tree index:198) done accuracy:0.975309 logloss:0.230169
    [INFO 23-05-21 17:18:02.2519 CDT random_forest.cc:805] Training of tree  208/300 (tree index:209) done accuracy:0.975309 logloss:0.229329
    [INFO 23-05-21 17:18:02.2521 CDT random_forest.cc:805] Training of tree  218/300 (tree index:218) done accuracy:0.975309 logloss:0.228753
    [INFO 23-05-21 17:18:02.2524 CDT random_forest.cc:805] Training of tree  229/300 (tree index:229) done accuracy:0.975309 logloss:0.228238
    [INFO 23-05-21 17:18:02.2527 CDT random_forest.cc:805] Training of tree  239/300 (tree index:239) done accuracy:0.975309 logloss:0.228552
    [INFO 23-05-21 17:18:02.2528 CDT random_forest.cc:805] Training of tree  249/300 (tree index:249) done accuracy:0.975309 logloss:0.228921
    [INFO 23-05-21 17:18:02.2530 CDT random_forest.cc:805] Training of tree  259/300 (tree index:261) done accuracy:0.975309 logloss:0.229078
    [INFO 23-05-21 17:18:02.2532 CDT random_forest.cc:805] Training of tree  269/300 (tree index:271) done accuracy:0.975309 logloss:0.228373
    [INFO 23-05-21 17:18:02.2534 CDT random_forest.cc:805] Training of tree  279/300 (tree index:280) done accuracy:0.975309 logloss:0.227925
    [INFO 23-05-21 17:18:02.2537 CDT random_forest.cc:805] Training of tree  289/300 (tree index:289) done accuracy:0.975309 logloss:0.227682
    [INFO 23-05-21 17:18:02.2539 CDT random_forest.cc:805] Training of tree  299/300 (tree index:299) done accuracy:0.975309 logloss:0.228828
    [INFO 23-05-21 17:18:02.2540 CDT random_forest.cc:805] Training of tree  300/300 (tree index:297) done accuracy:0.975309 logloss:0.228679
    [INFO 23-05-21 17:18:02.2541 CDT random_forest.cc:885] Final OOB metrics: accuracy:0.975309 logloss:0.228679
    [INFO 23-05-21 17:18:02.2543 CDT kernel.cc:926] Export model in log directory: /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpzc6rc0x8 with prefix 1946dd425d384770
    [INFO 23-05-21 17:18:02.2566 CDT kernel.cc:944] Save model in resources
    [INFO 23-05-21 17:18:02.2580 CDT abstract_model.cc:849] Model self evaluation:
    Number of predictions (without weights): 243
    Number of predictions (with weights): 243
    Task: CLASSIFICATION
    Label: __LABEL
    
    Accuracy: 0.975309  CI95[W][0.95185 0.989193]
    LogLoss: : 0.228679
    ErrorRate: : 0.0246913
    
    Default Accuracy: : 0.419753
    Default LogLoss: : 1.05992
    Default ErrorRate: : 0.580247
    
    Confusion Table:
    truth\prediction
       0    1   2   3
    0  0    0   0   0
    1  0  100   1   1
    2  0    1  89   0
    3  0    3   0  48
    Total: 243
    
    One vs other classes:
    [INFO 23-05-21 17:18:02.2619 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpzc6rc0x8/model/ with prefix 1946dd425d384770
    [INFO 23-05-21 17:18:02.2672 CDT decision_forest.cc:660] Model loaded with 300 root(s), 4072 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:18:02.2673 CDT abstract_model.cc:1312] Engine "RandomForestGeneric" built
    [INFO 23-05-21 17:18:02.2673 CDT kernel.cc:1074] Use fast generic engine
    Model trained in 0:00:00.023830
    Compiling model...
    Model compiled.

    <keras.callbacks.History at 0x2af3b5550>


<a id="org762da3d"></a>

# Evaluate the model

    WARNING:tensorflow:6 out of the last 7 calls to <function InferenceCoreModel.make_test_function.<locals>.test_function at 0x29ff61d30> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    WARNING:tensorflow:6 out of the last 7 calls to <function InferenceCoreModel.make_test_function.<locals>.test_function at 0x29ff61d30> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    1/1 [==============================] - 0s 78ms/step - loss: 0.0000e+00 - accuracy: 0.9604
    
    
    loss: 0.0000
    accuracy: 0.9604


<a id="org7191025"></a>

# TensorFlow Serving

    WARNING:absl:Found untraced functions such as call_get_leaves while saving (showing 1 of 1). These functions will not be directly callable after loading.
    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets
    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets


<a id="orgdf60811"></a>

# Model structure and feature importance

    Model: "random_forest_model_4"
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
        1.    "bill_length_mm"  0.431934 ################
        2. "flipper_length_mm"  0.412329 ##############
        3.     "bill_depth_mm"  0.352891 #########
        4.            "island"  0.318666 ######
        5.       "body_mass_g"  0.270240 ##
        6.               "sex"  0.246868 
        7.              "year"  0.245097 
    
    Variable Importance: NUM_AS_ROOT:
        1. "flipper_length_mm" 132.000000 ################
        2.     "bill_depth_mm" 78.000000 #########
        3.    "bill_length_mm" 77.000000 #########
        4.            "island" 12.000000 #
        5.       "body_mass_g"  1.000000 
    
    Variable Importance: NUM_NODES:
        1.    "bill_length_mm" 586.000000 ################
        2.     "bill_depth_mm" 418.000000 ###########
        3. "flipper_length_mm" 331.000000 ########
        4.            "island" 278.000000 #######
        5.       "body_mass_g" 242.000000 ######
        6.               "sex" 26.000000 
        7.              "year"  5.000000 
    
    Variable Importance: SUM_SCORE:
        1.    "bill_length_mm" 25601.603109 ################
        2. "flipper_length_mm" 21013.957566 #############
        3.     "bill_depth_mm" 14394.067534 ########
        4.            "island" 10707.687475 ######
        5.       "body_mass_g" 2164.562192 #
        6.               "sex" 195.347281 
        7.              "year" 14.066157 
    
    
    
    Winner takes all: true
    Out-of-bag evaluation: accuracy:0.975309 logloss:0.228679
    Number of trees: 300
    Total number of nodes: 4072
    
    Number of nodes by tree:
    Count: 300 Average: 13.5733 StdDev: 2.90596
    Min: 7 Max: 23 Ignored: 0
    ----------------------------------------------
    [  7,  8)  3   1.00%   1.00%
    [  8,  9)  0   0.00%   1.00%
    [  9, 10) 27   9.00%  10.00% ###
    [ 10, 11)  0   0.00%  10.00%
    [ 11, 12) 55  18.33%  28.33% ######
    [ 12, 13)  0   0.00%  28.33%
    [ 13, 14) 98  32.67%  61.00% ##########
    [ 14, 15)  0   0.00%  61.00%
    [ 15, 16) 64  21.33%  82.33% #######
    [ 16, 17)  0   0.00%  82.33%
    [ 17, 18) 28   9.33%  91.67% ###
    [ 18, 19)  0   0.00%  91.67%
    [ 19, 20) 18   6.00%  97.67% ##
    [ 20, 21)  0   0.00%  97.67%
    [ 21, 22)  5   1.67%  99.33% #
    [ 22, 23)  0   0.00%  99.33%
    [ 23, 23]  2   0.67% 100.00%
    
    Depth by leafs:
    Count: 2186 Average: 3.15691 StdDev: 0.959658
    Min: 1 Max: 6 Ignored: 0
    ----------------------------------------------
    [ 1, 2)  23   1.05%   1.05%
    [ 2, 3) 577  26.40%  27.45% #######
    [ 3, 4) 812  37.15%  64.59% ##########
    [ 4, 5) 606  27.72%  92.31% #######
    [ 5, 6) 144   6.59%  98.90% ##
    [ 6, 6]  24   1.10% 100.00%
    
    Number of training obs by leaf:
    Count: 2186 Average: 33.3486 StdDev: 32.7396
    Min: 5 Max: 110 Ignored: 0
    ----------------------------------------------
    [   5,  10) 1037  47.44%  47.44% ##########
    [  10,  15)   81   3.71%  51.14% #
    [  15,  20)   57   2.61%  53.75% #
    [  20,  26)   37   1.69%  55.44%
    [  26,  31)   47   2.15%  57.59%
    [  31,  36)   55   2.52%  60.11% #
    [  36,  42)   81   3.71%  63.82% #
    [  42,  47)   93   4.25%  68.07% #
    [  47,  52)   82   3.75%  71.82% #
    [  52,  58)   51   2.33%  74.15%
    [  58,  63)   44   2.01%  76.17%
    [  63,  68)   33   1.51%  77.68%
    [  68,  73)   38   1.74%  79.41%
    [  73,  79)   60   2.74%  82.16% #
    [  79,  84)   94   4.30%  86.46% #
    [  84,  89)  100   4.57%  91.03% #
    [  89,  95)  113   5.17%  96.20% #
    [  95, 100)   54   2.47%  98.67% #
    [ 100, 105)   23   1.05%  99.73%
    [ 105, 110]    6   0.27% 100.00%
    
    Attribute in nodes:
    	586 : bill_length_mm [NUMERICAL]
    	418 : bill_depth_mm [NUMERICAL]
    	331 : flipper_length_mm [NUMERICAL]
    	278 : island [CATEGORICAL]
    	242 : body_mass_g [NUMERICAL]
    	26 : sex [CATEGORICAL]
    	5 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 0:
    	132 : flipper_length_mm [NUMERICAL]
    	78 : bill_depth_mm [NUMERICAL]
    	77 : bill_length_mm [NUMERICAL]
    	12 : island [CATEGORICAL]
    	1 : body_mass_g [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	247 : bill_length_mm [NUMERICAL]
    	210 : flipper_length_mm [NUMERICAL]
    	191 : bill_depth_mm [NUMERICAL]
    	160 : island [CATEGORICAL]
    	69 : body_mass_g [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	438 : bill_length_mm [NUMERICAL]
    	325 : bill_depth_mm [NUMERICAL]
    	278 : flipper_length_mm [NUMERICAL]
    	243 : island [CATEGORICAL]
    	161 : body_mass_g [NUMERICAL]
    	9 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 3:
    	550 : bill_length_mm [NUMERICAL]
    	405 : bill_depth_mm [NUMERICAL]
    	321 : flipper_length_mm [NUMERICAL]
    	272 : island [CATEGORICAL]
    	221 : body_mass_g [NUMERICAL]
    	24 : sex [CATEGORICAL]
    	3 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 5:
    	586 : bill_length_mm [NUMERICAL]
    	418 : bill_depth_mm [NUMERICAL]
    	331 : flipper_length_mm [NUMERICAL]
    	278 : island [CATEGORICAL]
    	242 : body_mass_g [NUMERICAL]
    	26 : sex [CATEGORICAL]
    	5 : year [NUMERICAL]
    
    Condition type in nodes:
    	1582 : HigherCondition
    	304 : ContainsBitmapCondition
    Condition type in nodes with depth <= 0:
    	288 : HigherCondition
    	12 : ContainsBitmapCondition
    Condition type in nodes with depth <= 1:
    	717 : HigherCondition
    	160 : ContainsBitmapCondition
    Condition type in nodes with depth <= 2:
    	1202 : HigherCondition
    	252 : ContainsBitmapCondition
    Condition type in nodes with depth <= 3:
    	1500 : HigherCondition
    	296 : ContainsBitmapCondition
    Condition type in nodes with depth <= 5:
    	1582 : HigherCondition
    	304 : ContainsBitmapCondition
    Node format: NOT_SET
    
    Training OOB:
    	trees: 1, Out-of-bag evaluation: accuracy:0.908163 logloss:3.31013
    	trees: 11, Out-of-bag evaluation: accuracy:0.941176 logloss:0.981108
    	trees: 21, Out-of-bag evaluation: accuracy:0.950617 logloss:0.385867
    	trees: 31, Out-of-bag evaluation: accuracy:0.962963 logloss:0.379752
    	trees: 41, Out-of-bag evaluation: accuracy:0.962963 logloss:0.235278
    	trees: 52, Out-of-bag evaluation: accuracy:0.971193 logloss:0.233288
    	trees: 62, Out-of-bag evaluation: accuracy:0.971193 logloss:0.234777
    	trees: 74, Out-of-bag evaluation: accuracy:0.975309 logloss:0.236124
    	trees: 84, Out-of-bag evaluation: accuracy:0.971193 logloss:0.234085
    	trees: 95, Out-of-bag evaluation: accuracy:0.971193 logloss:0.232744
    	trees: 106, Out-of-bag evaluation: accuracy:0.975309 logloss:0.231187
    	trees: 117, Out-of-bag evaluation: accuracy:0.975309 logloss:0.230766
    	trees: 127, Out-of-bag evaluation: accuracy:0.975309 logloss:0.231985
    	trees: 137, Out-of-bag evaluation: accuracy:0.975309 logloss:0.230148
    	trees: 147, Out-of-bag evaluation: accuracy:0.975309 logloss:0.230627
    	trees: 157, Out-of-bag evaluation: accuracy:0.975309 logloss:0.230122
    	trees: 168, Out-of-bag evaluation: accuracy:0.975309 logloss:0.231888
    	trees: 178, Out-of-bag evaluation: accuracy:0.975309 logloss:0.230603
    	trees: 188, Out-of-bag evaluation: accuracy:0.975309 logloss:0.229802
    	trees: 198, Out-of-bag evaluation: accuracy:0.975309 logloss:0.230169
    	trees: 208, Out-of-bag evaluation: accuracy:0.975309 logloss:0.229329
    	trees: 218, Out-of-bag evaluation: accuracy:0.975309 logloss:0.228753
    	trees: 229, Out-of-bag evaluation: accuracy:0.975309 logloss:0.228238
    	trees: 239, Out-of-bag evaluation: accuracy:0.975309 logloss:0.228552
    	trees: 249, Out-of-bag evaluation: accuracy:0.975309 logloss:0.228921
    	trees: 259, Out-of-bag evaluation: accuracy:0.975309 logloss:0.229078
    	trees: 269, Out-of-bag evaluation: accuracy:0.975309 logloss:0.228373
    	trees: 279, Out-of-bag evaluation: accuracy:0.975309 logloss:0.227925
    	trees: 289, Out-of-bag evaluation: accuracy:0.975309 logloss:0.227682
    	trees: 299, Out-of-bag evaluation: accuracy:0.975309 logloss:0.228828
    	trees: 300, Out-of-bag evaluation: accuracy:0.975309 logloss:0.228679


<a id="orgdfdf6a6"></a>

# Using make<sub>inspector</sub>

    '("bill_depth_mm" (1; #1) 
     "bill_length_mm" (1; #2) 
     "body_mass_g" (1; #3) 
     "flipper_length_mm" (1; #4) 
     "island" (4; #5) 
     "sex" (4; #6) 
     "year" (1; #7))

    '("INV_MEAN_MIN_DEPTH": (("bill_length_mm" (1; #2)  0.43193396458920585) 
      ("flipper_length_mm" (1; #4)  0.4123293823557732) 
      ("bill_depth_mm" (1; #1)  0.35289067365454707) 
      ("island" (4; #5)  0.31866627968920924) 
      ("body_mass_g" (1; #3)  0.27023978516522035) 
      ("sex" (4; #6)  0.2468684136139562) 
      ("year" (1; #7)  0.24509743241871595)) 
     "NUM_AS_ROOT": (("flipper_length_mm" (1; #4)  132.0) 
      ("bill_depth_mm" (1; #1)  78.0) 
      ("bill_length_mm" (1; #2)  77.0) 
      ("island" (4; #5)  12.0) 
      ("body_mass_g" (1; #3)  1.0)) 
     "NUM_NODES": (("bill_length_mm" (1; #2)  586.0) 
      ("bill_depth_mm" (1; #1)  418.0) 
      ("flipper_length_mm" (1; #4)  331.0) 
      ("island" (4; #5)  278.0) 
      ("body_mass_g" (1; #3)  242.0) 
      ("sex" (4; #6)  26.0) 
      ("year" (1; #7)  5.0)) 
     "SUM_SCORE": (("bill_length_mm" (1; #2)  25601.603109234944) 
      ("flipper_length_mm" (1; #4)  21013.957565906458) 
      ("bill_depth_mm" (1; #1)  14394.067533643916) 
      ("island" (4; #5)  10707.687474731356) 
      ("body_mass_g" (1; #3)  2164.5621923580766) 
      ("sex" (4; #6)  195.34728068858385) 
      ("year" (1; #7)  14.066157028079033)))


<a id="org87e5eeb"></a>

# Model self evaluation

    Evaluation(num_examples=243, accuracy=0.9753086419753086, loss=0.2286787007096005, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)


<a id="orgbf7d12b"></a>

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
<td class="org-left">(num<sub>trees</sub>=1 evaluation=Evaluation (num<sub>examples</sub>=98 accuracy=0.9081632653061225 loss=3.3101312676254584 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=11 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9411764705882353 loss=0.9811079443252387 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=21 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9506172839506173 loss=0.38586743727878287 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=31 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9629629629629629 loss=0.3797522790331409 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=41 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9629629629629629 loss=0.2352775759197802 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=52 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.23328769591794093 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=62 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.23477735813446496 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=74 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.23612427034649094 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=84 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.23408457170788644 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=95 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9711934156378601 loss=0.23274359701071012 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=106 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.23118683899687642 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=117 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.2307659005094703 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=127 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.23198460558345044 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=137 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.2301477895428737 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=147 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.23062721626840746 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=157 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.23012209971868453 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=168 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.2318882454960864 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=178 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.23060251625231754 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=188 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.22980225106141694 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=198 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.23016897993691174 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=208 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.22932875239570077 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=218 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.22875296639914744 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=229 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.2282381487144119 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=239 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.2285520337299739 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=249 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.22892061617118095 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=259 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.22907751104545324 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=269 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.22837278972008102 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=279 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.22792513865149683 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=289 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.2276817784058275 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=299 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.22882776671001082 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=300 evaluation=Evaluation (num<sub>examples</sub>=243 accuracy=0.9753086419753086 loss=0.2286787007096005 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
</tr>
</tbody>
</table>

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


<a id="org0cc95b6"></a>

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


<a id="org0ca1421"></a>

# Using a subset of features

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmphbznoqth as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.079003. Found 243 examples.
    Reading validation dataset...
    Num validation examples: tf.Tensor(101, shape=(), dtype=int32)
    Validation dataset read in 0:00:00.056623. Found 101 examples.
    Training model...
    [WARNING 23-05-21 17:18:04.2389 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:04.2389 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:04.2390 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:00.063194
    Compiling model...
    Model compiled.
    1/1 [==============================] - 0s 47ms/step - loss: 0.0000e+00 - accuracy: 0.9208
    {'loss': 0.0, 'accuracy': 0.9207921028137207}
    [INFO 23-05-21 17:18:04.4429 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmphbznoqth/model/ with prefix 2ab7c50af3db4c7e
    [INFO 23-05-21 17:18:04.4470 CDT decision_forest.cc:660] Model loaded with 102 root(s), 3342 node(s), and 2 input feature(s).
    [INFO 23-05-21 17:18:04.4470 CDT kernel.cc:1074] Use fast generic engine

**TF-DF** attaches a **semantics** to each feature. This semantics controls how the feature is used by the model. The following semantics are currently supported.

-   **Numerical**: Generally for quantities or counts with full ordering. For example, the age of a person, or the number of items in a bag. Can be a float or an integer. Missing values are represented with a float(Nan) or with an empty sparse tensor.
-   **Categorical**: Generally for a type/class in finite set of possible values without ordering. For example, the color RED in the set {RED, BLUE, GREEN}. Can be a string or an integer. Missing values are represented as &ldquo;&rdquo; (empty string), value -2 or with an empty sparse tensor.
-   **Categorical-Set**: A set of categorical values. Great to represent tokenized text. Can be a string or an integer in a sparse tensor or a ragged tensor (recommended). The order/index of each item doesnt matter.
    
    If not specified, the semantics is inferred from the representation type and shown in the training logs:
    
    -   int, float (dense or sparse) -> Numerical semantics
    
    -   str, (dense or sparse) -> Categorical semantics
    
    -   int, str (ragged) -> Categorical-Set semantics

In some cases, the inferred semantics is incorrect. For example: An Enum stored as an integer is semantically categorical, but it will be detected as numerical. In this case, you should specify the semantic argument in the input. The education<sub>num</sub> field of the Adult dataset is a classic example.

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpcg0w723c as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.074979. Found 243 examples.
    Reading validation dataset...
    Num validation examples: tf.Tensor(101, shape=(), dtype=int32)
    Validation dataset read in 0:00:00.063200. Found 101 examples.
    Training model...
    Model trained in 0:00:00.045946
    Compiling model...
    [WARNING 23-05-21 17:18:04.7401 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:04.7401 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:04.7401 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    [INFO 23-05-21 17:18:04.9293 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpcg0w723c/model/ with prefix 5b798894a26543f3
    [INFO 23-05-21 17:18:04.9308 CDT decision_forest.cc:660] Model loaded with 33 root(s), 1161 node(s), and 3 input feature(s).
    [INFO 23-05-21 17:18:04.9308 CDT kernel.cc:1074] Use fast generic engine
    Model compiled.

    <keras.callbacks.History at 0x2b10cc490>

Note that `year` is in the list of CATEGORICAL features (unlike the first run)


<a id="org8ee6c44"></a>

# Hyper-parameters

**Hyper-parameters** are paramters of the training algorithm that impact the quality of the final model. They are specified in the model class constructor. The list of hyper-parameters is visible with the *question mark* colab command.

**I will figure out how to obtain that list without the question mark command.**

    Model: "gradient_boosted_trees_model_7"
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
        1.            "island"  0.332674 ################
        2.    "bill_length_mm"  0.311817 ##############
        3.     "bill_depth_mm"  0.284184 ###########
        4. "flipper_length_mm"  0.263190 ##########
        5.       "body_mass_g"  0.217356 ######
        6.              "year"  0.155988 
        7.               "sex"  0.145213 
    
    Variable Importance: NUM_AS_ROOT:
        1.            "island" 515.000000 ################
        2. "flipper_length_mm" 500.000000 ########
        3.    "bill_length_mm" 485.000000 
    
    Variable Importance: NUM_NODES:
        1.    "bill_length_mm" 11434.000000 ################
        2.       "body_mass_g" 9944.000000 #############
        3.     "bill_depth_mm" 8432.000000 ###########
        4. "flipper_length_mm" 6212.000000 ########
        5.            "island" 3662.000000 ####
        6.              "year" 1971.000000 ##
        7.               "sex" 350.000000 
    
    Variable Importance: SUM_SCORE:
        1.    "bill_length_mm" 301.659302 ################
        2. "flipper_length_mm" 214.893044 ###########
        3.            "island" 84.986948 ####
        4.     "bill_depth_mm" 37.895442 #
        5.               "sex"  3.655171 
        6.              "year"  0.552280 
        7.       "body_mass_g"  0.494734 
    
    
    
    Loss: MULTINOMIAL_LOG_LIKELIHOOD
    Validation loss value: 6.42508e-06
    Number of trees per iteration: 3
    Node format: NOT_SET
    Number of trees: 1500
    Total number of nodes: 85510
    
    Number of nodes by tree:
    Count: 1500 Average: 57.0067 StdDev: 5.42807
    Min: 13 Max: 61 Ignored: 0
    ----------------------------------------------
    [ 13, 15)   2   0.13%   0.13%
    [ 15, 17)   0   0.00%   0.13%
    [ 17, 20)   0   0.00%   0.13%
    [ 20, 22)   2   0.13%   0.27%
    [ 22, 25)   0   0.00%   0.27%
    [ 25, 27)   0   0.00%   0.27%
    [ 27, 30)   1   0.07%   0.33%
    [ 30, 32)   0   0.00%   0.33%
    [ 32, 35)   1   0.07%   0.40%
    [ 35, 37)   0   0.00%   0.40%
    [ 37, 39)   1   0.07%   0.47%
    [ 39, 42)  10   0.67%   1.13%
    [ 42, 44)  25   1.67%   2.80%
    [ 44, 47)  28   1.87%   4.67%
    [ 47, 49)  39   2.60%   7.27%
    [ 49, 52) 148   9.87%  17.13% ##
    [ 52, 54) 108   7.20%  24.33% #
    [ 54, 57)  80   5.33%  29.67% #
    [ 57, 59) 201  13.40%  43.07% ##
    [ 59, 61] 854  56.93% 100.00% ##########
    
    Depth by leafs:
    Count: 43505 Average: 5.93817 StdDev: 1.70477
    Min: 2 Max: 8 Ignored: 0
    ----------------------------------------------
    [ 2, 3) 1208   2.78%   2.78% #
    [ 3, 4) 3744   8.61%  11.38% ####
    [ 4, 5) 4558  10.48%  21.86% #####
    [ 5, 6) 6524  15.00%  36.86% #######
    [ 6, 7) 8361  19.22%  56.07% ########
    [ 7, 8) 9206  21.16%  77.23% #########
    [ 8, 8] 9904  22.77% 100.00% ##########
    
    Number of training obs by leaf:
    Count: 43505 Average: 0 StdDev: 0
    Min: 0 Max: 0 Ignored: 0
    ----------------------------------------------
    [ 0, 0] 43505 100.00% 100.00% ##########
    
    Attribute in nodes:
    	11434 : bill_length_mm [NUMERICAL]
    	9944 : body_mass_g [NUMERICAL]
    	8432 : bill_depth_mm [NUMERICAL]
    	6212 : flipper_length_mm [NUMERICAL]
    	3662 : island [CATEGORICAL]
    	1971 : year [NUMERICAL]
    	350 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 0:
    	515 : island [CATEGORICAL]
    	500 : flipper_length_mm [NUMERICAL]
    	485 : bill_length_mm [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	1383 : bill_depth_mm [NUMERICAL]
    	1194 : island [CATEGORICAL]
    	1041 : bill_length_mm [NUMERICAL]
    	572 : flipper_length_mm [NUMERICAL]
    	310 : body_mass_g [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	2696 : bill_depth_mm [NUMERICAL]
    	2352 : bill_length_mm [NUMERICAL]
    	1424 : island [CATEGORICAL]
    	1297 : flipper_length_mm [NUMERICAL]
    	1149 : body_mass_g [NUMERICAL]
    	366 : year [NUMERICAL]
    	8 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 3:
    	4586 : bill_length_mm [NUMERICAL]
    	3297 : bill_depth_mm [NUMERICAL]
    	2673 : body_mass_g [NUMERICAL]
    	2007 : island [CATEGORICAL]
    	1711 : flipper_length_mm [NUMERICAL]
    	848 : year [NUMERICAL]
    	10 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 5:
    	8582 : bill_length_mm [NUMERICAL]
    	6285 : bill_depth_mm [NUMERICAL]
    	6111 : body_mass_g [NUMERICAL]
    	4428 : flipper_length_mm [NUMERICAL]
    	2895 : island [CATEGORICAL]
    	1530 : year [NUMERICAL]
    	143 : sex [CATEGORICAL]
    
    Condition type in nodes:
    	37993 : HigherCondition
    	4012 : ContainsBitmapCondition
    Condition type in nodes with depth <= 0:
    	985 : HigherCondition
    	515 : ContainsBitmapCondition
    Condition type in nodes with depth <= 1:
    	3306 : HigherCondition
    	1194 : ContainsBitmapCondition
    Condition type in nodes with depth <= 2:
    	7860 : HigherCondition
    	1432 : ContainsBitmapCondition
    Condition type in nodes with depth <= 3:
    	13115 : HigherCondition
    	2017 : ContainsBitmapCondition
    Condition type in nodes with depth <= 5:
    	26936 : HigherCondition
    	3038 : ContainsBitmapCondition
    
    Training logs:
    Number of iteration to final model: 500
    	Iter:1 train-loss:0.919230 valid-loss:0.912197  train-accuracy:0.985981 valid-accuracy:1.000000
    	Iter:2 train-loss:0.778176 valid-loss:0.767017  train-accuracy:0.990654 valid-accuracy:1.000000
    	Iter:3 train-loss:0.664605 valid-loss:0.650211  train-accuracy:0.990654 valid-accuracy:1.000000
    	Iter:4 train-loss:0.571750 valid-loss:0.555423  train-accuracy:0.990654 valid-accuracy:1.000000
    	Iter:5 train-loss:0.494960 valid-loss:0.476803  train-accuracy:0.990654 valid-accuracy:1.000000
    	Iter:6 train-loss:0.428629 valid-loss:0.409921  train-accuracy:0.990654 valid-accuracy:1.000000
    	Iter:16 train-loss:0.114274 valid-loss:0.104600  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:26 train-loss:0.031767 valid-loss:0.028723  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:36 train-loss:0.009038 valid-loss:0.008112  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:46 train-loss:0.002645 valid-loss:0.002361  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:56 train-loss:0.000765 valid-loss:0.000784  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:66 train-loss:0.000232 valid-loss:0.000277  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:76 train-loss:0.000091 valid-loss:0.000120  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:86 train-loss:0.000053 valid-loss:0.000075  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:96 train-loss:0.000038 valid-loss:0.000055  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:106 train-loss:0.000030 valid-loss:0.000045  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:116 train-loss:0.000025 valid-loss:0.000039  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:126 train-loss:0.000021 valid-loss:0.000034  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:136 train-loss:0.000019 valid-loss:0.000031  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:146 train-loss:0.000016 valid-loss:0.000028  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:156 train-loss:0.000015 valid-loss:0.000025  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:166 train-loss:0.000013 valid-loss:0.000023  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:176 train-loss:0.000012 valid-loss:0.000022  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:186 train-loss:0.000011 valid-loss:0.000020  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:196 train-loss:0.000011 valid-loss:0.000019  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:206 train-loss:0.000010 valid-loss:0.000018  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:216 train-loss:0.000009 valid-loss:0.000016  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:226 train-loss:0.000009 valid-loss:0.000015  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:236 train-loss:0.000008 valid-loss:0.000015  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:246 train-loss:0.000008 valid-loss:0.000014  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:256 train-loss:0.000008 valid-loss:0.000013  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:266 train-loss:0.000007 valid-loss:0.000013  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:276 train-loss:0.000007 valid-loss:0.000012  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:286 train-loss:0.000007 valid-loss:0.000012  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:296 train-loss:0.000006 valid-loss:0.000011  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:306 train-loss:0.000006 valid-loss:0.000011  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:316 train-loss:0.000006 valid-loss:0.000011  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:326 train-loss:0.000006 valid-loss:0.000010  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:336 train-loss:0.000005 valid-loss:0.000010  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:346 train-loss:0.000005 valid-loss:0.000010  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:356 train-loss:0.000005 valid-loss:0.000009  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:366 train-loss:0.000005 valid-loss:0.000009  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:376 train-loss:0.000005 valid-loss:0.000009  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:386 train-loss:0.000005 valid-loss:0.000009  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:396 train-loss:0.000004 valid-loss:0.000008  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:406 train-loss:0.000004 valid-loss:0.000008  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:416 train-loss:0.000004 valid-loss:0.000008  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:426 train-loss:0.000004 valid-loss:0.000008  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:436 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:446 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:456 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:466 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:476 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:486 train-loss:0.000003 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:496 train-loss:0.000003 valid-loss:0.000006  train-accuracy:1.000000 valid-accuracy:1.000000

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpfcvx5l2g as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.083168. Found 243 examples.
    Training model...
    [WARNING 23-05-21 17:18:05.1362 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:05.1362 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:05.1362 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:02.102269
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:18:07.2201 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpfcvx5l2g/model/ with prefix fb78e0d5cebe4d52
    [INFO 23-05-21 17:18:07.3213 CDT decision_forest.cc:660] Model loaded with 1500 root(s), 85510 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:18:07.3213 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2b2aa7760>

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpo6fl4ddj as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.094879. Found 243 examples.
    Training model...
    [WARNING 23-05-21 17:18:07.6780 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:07.6781 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:07.6781 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:03.583727
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:18:11.2473 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpo6fl4ddj/model/ with prefix fc03b0f60be14024
    [INFO 23-05-21 17:18:11.3575 CDT decision_forest.cc:660] Model loaded with 1500 root(s), 86150 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:18:11.3575 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2b58d5670>

As new training methods are published and implemented, combinations of hyper-parameters can emerge as good or almost-always-better than the default parameters. To avoid changing the default hyper-parameter values these good combinations are indexed and availale as hyper-parameter templates.

For example, the benchmark<sub>rank1</sub> template is the best combination on our internal benchmarks. Those templates are versioned to allow training configuration stability e.g. benchmark<sub>rank1</sub>@v1.

    Resolve hyper-parameter template "benchmark_rank1" to "benchmark_rank1@v1" -> {'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}.
    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpdxzuwz01 as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.085649. Found 243 examples.
    Training model...
    [WARNING 23-05-21 17:18:11.5632 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:11.5633 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:18:11.5633 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:00.972541
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:18:12.5798 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpdxzuwz01/model/ with prefix e0c5b5ede0de4a70
    [INFO 23-05-21 17:18:12.6248 CDT decision_forest.cc:660] Model loaded with 900 root(s), 35524 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:18:12.6248 CDT abstract_model.cc:1312] Engine "GradientBoostedTreesGeneric" built
    [INFO 23-05-21 17:18:12.6248 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2b5fa57f0>

The available templates are available with `predefined_hyperparameters`. Note that different learning algorithms have different templates, even if the name is similar.

    [HyperParameterTemplate(name='better_default', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL'}, description='A configuration that is generally better than the default parameters without being more expensive.'), HyperParameterTemplate(name='benchmark_rank1', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}, description='Top ranking hyper-parameters on our benchmark slightly modified to run in reasonable time.')]

What is returned are the predefined hyper-parameters of the Gradient Boosted Tree model.


<a id="org2ebf064"></a>

# Feature Preprocessing

Pre-processing features is sometimes necessary to consume signals with complex structures, to regularize the model or to apply transfer learning. Pre-processing can be done in one of three ways:

1.  **Preprocessing on the pandas dataframe**: This solution is easy tto implement and generally suitable for experiementation. However, the pre-processing logic will not be exported in the model by model.save()
2.  **Keras Preprocessing**: While more complex than the previous solution, Keras Preprocessing is packaged in the model.
3.  **TensorFlow Feature Columns**: This API is part of the TF Estimator library (!= Keras) and planned for deprecation. This solution is interesting when using existing preprocessing code.

**Note**: Using **TensorFlow Hub** pre-trained embedding is often, a great way to consume text and image with TF-DF.

In the next example, pre-process the body<sub>mass</sub><sub>g</sub> feature into body<sub>mass</sub><sub>kg</sub> = body<sub>mass</sub><sub>g</sub> / 1000. The bill<sub>length</sub><sub>mm</sub> is consumed without preprocessing. Note that such monotonic transformations have generally no impact on decision forest models.

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpzr228e_p as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.083957. Found 243 examples.
    Training model...
    Model trained in 0:00:00.022689
    Compiling model...
    Model compiled.
    Model: "random_forest_model_5"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model_1 (Functional)        {'body_mass_kg': (None,   0         
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
        1. "bill_length_mm"  0.991289 ################
        2.   "body_mass_kg"  0.428996 
    
    Variable Importance: NUM_AS_ROOT:
        1. "bill_length_mm" 298.000000 ################
        2.   "body_mass_kg"  2.000000 
    
    Variable Importance: NUM_NODES:
        1. "bill_length_mm" 1570.000000 ################
        2.   "body_mass_kg" 1332.000000 
    
    Variable Importance: SUM_SCORE:
        1. "bill_length_mm" 46167.461016 ################
        2.   "body_mass_kg" 25632.975844 
    
    
    
    Winner takes all: true
    Out-of-bag evaluation: accuracy:0.925926 logloss:0.602753
    Number of trees: 300
    Total number of nodes: 6104
    
    Number of nodes by tree:
    Count: 300 Average: 20.3467 StdDev: 3.08542
    Min: 13 Max: 29 Ignored: 0
    ----------------------------------------------
    [ 13, 14)  8   2.67%   2.67% #
    [ 14, 15)  0   0.00%   2.67%
    [ 15, 16) 20   6.67%   9.33% ##
    [ 16, 17)  0   0.00%   9.33%
    [ 17, 18) 36  12.00%  21.33% ####
    [ 18, 19)  0   0.00%  21.33%
    [ 19, 20) 63  21.00%  42.33% ########
    [ 20, 21)  0   0.00%  42.33%
    [ 21, 22) 84  28.00%  70.33% ##########
    [ 22, 23)  0   0.00%  70.33%
    [ 23, 24) 59  19.67%  90.00% #######
    [ 24, 25)  0   0.00%  90.00%
    [ 25, 26) 21   7.00%  97.00% ###
    [ 26, 27)  0   0.00%  97.00%
    [ 27, 28)  8   2.67%  99.67% #
    [ 28, 29)  0   0.00%  99.67%
    [ 29, 29]  1   0.33% 100.00%
    
    Depth by leafs:
    Count: 3202 Average: 3.90194 StdDev: 1.30094
    Min: 1 Max: 8 Ignored: 0
    ----------------------------------------------
    [ 1, 2)   26   0.81%   0.81%
    [ 2, 3)  308   9.62%  10.43% ###
    [ 3, 4) 1033  32.26%  42.69% ##########
    [ 4, 5)  992  30.98%  73.67% ##########
    [ 5, 6)  442  13.80%  87.48% ####
    [ 6, 7)  262   8.18%  95.66% ###
    [ 7, 8)  109   3.40%  99.06% #
    [ 8, 8]   30   0.94% 100.00%
    
    Number of training obs by leaf:
    Count: 3202 Average: 22.767 StdDev: 26.7864
    Min: 5 Max: 109 Ignored: 0
    ----------------------------------------------
    [   5,  10) 1935  60.43%  60.43% ##########
    [  10,  15)  284   8.87%  69.30% #
    [  15,  20)   59   1.84%  71.14%
    [  20,  26)   23   0.72%  71.86%
    [  26,  31)   24   0.75%  72.61%
    [  31,  36)   58   1.81%  74.42%
    [  36,  41)   98   3.06%  77.48% #
    [  41,  47)   85   2.65%  80.14%
    [  47,  52)   49   1.53%  81.67%
    [  52,  57)   35   1.09%  82.76%
    [  57,  62)   43   1.34%  84.10%
    [  62,  68)   95   2.97%  87.07%
    [  68,  73)  107   3.34%  90.41% #
    [  73,  78)  101   3.15%  93.57% #
    [  78,  83)   71   2.22%  95.78%
    [  83,  89)   62   1.94%  97.72%
    [  89,  94)   35   1.09%  98.81%
    [  94,  99)   21   0.66%  99.47%
    [  99, 104)   11   0.34%  99.81%
    [ 104, 109]    6   0.19% 100.00%
    
    Attribute in nodes:
    	1570 : bill_length_mm [NUMERICAL]
    	1332 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 0:
    	298 : bill_length_mm [NUMERICAL]
    	2 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	530 : bill_length_mm [NUMERICAL]
    	344 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	888 : bill_length_mm [NUMERICAL]
    	826 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 3:
    	1255 : bill_length_mm [NUMERICAL]
    	1106 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 5:
    	1529 : bill_length_mm [NUMERICAL]
    	1296 : body_mass_kg [NUMERICAL]
    
    Condition type in nodes:
    	2902 : HigherCondition
    Condition type in nodes with depth <= 0:
    	300 : HigherCondition
    Condition type in nodes with depth <= 1:
    	874 : HigherCondition
    Condition type in nodes with depth <= 2:
    	1714 : HigherCondition
    Condition type in nodes with depth <= 3:
    	2361 : HigherCondition
    Condition type in nodes with depth <= 5:
    	2825 : HigherCondition
    Node format: NOT_SET
    
    Training OOB:
    	trees: 1, Out-of-bag evaluation: accuracy:0.928571 logloss:2.57455
    	trees: 11, Out-of-bag evaluation: accuracy:0.9 logloss:1.87794
    	trees: 22, Out-of-bag evaluation: accuracy:0.917695 logloss:1.0157
    	trees: 32, Out-of-bag evaluation: accuracy:0.925926 logloss:1.01684
    	trees: 42, Out-of-bag evaluation: accuracy:0.925926 logloss:0.884884
    	trees: 53, Out-of-bag evaluation: accuracy:0.925926 logloss:0.737891
    	trees: 63, Out-of-bag evaluation: accuracy:0.925926 logloss:0.740678
    	trees: 76, Out-of-bag evaluation: accuracy:0.925926 logloss:0.73959
    	trees: 86, Out-of-bag evaluation: accuracy:0.925926 logloss:0.741681
    	trees: 98, Out-of-bag evaluation: accuracy:0.930041 logloss:0.737855
    	trees: 109, Out-of-bag evaluation: accuracy:0.930041 logloss:0.74172
    	trees: 119, Out-of-bag evaluation: accuracy:0.930041 logloss:0.742952
    	trees: 130, Out-of-bag evaluation: accuracy:0.930041 logloss:0.741761
    	trees: 140, Out-of-bag evaluation: accuracy:0.930041 logloss:0.608502
    	trees: 151, Out-of-bag evaluation: accuracy:0.930041 logloss:0.608368
    	trees: 162, Out-of-bag evaluation: accuracy:0.930041 logloss:0.608285
    	trees: 175, Out-of-bag evaluation: accuracy:0.925926 logloss:0.609561
    	trees: 186, Out-of-bag evaluation: accuracy:0.934156 logloss:0.608255
    	trees: 196, Out-of-bag evaluation: accuracy:0.925926 logloss:0.605742
    	trees: 206, Out-of-bag evaluation: accuracy:0.925926 logloss:0.603163
    	trees: 217, Out-of-bag evaluation: accuracy:0.930041 logloss:0.600887
    	trees: 229, Out-of-bag evaluation: accuracy:0.930041 logloss:0.602244
    	trees: 239, Out-of-bag evaluation: accuracy:0.930041 logloss:0.60093
    	trees: 249, Out-of-bag evaluation: accuracy:0.925926 logloss:0.602192
    	trees: 259, Out-of-bag evaluation: accuracy:0.925926 logloss:0.601079
    	trees: 269, Out-of-bag evaluation: accuracy:0.925926 logloss:0.601953
    	trees: 279, Out-of-bag evaluation: accuracy:0.930041 logloss:0.60238
    	trees: 291, Out-of-bag evaluation: accuracy:0.930041 logloss:0.602238
    	trees: 300, Out-of-bag evaluation: accuracy:0.925926 logloss:0.602753
    /Users/umbertofasci/miniforge3/envs/tensorflow-metal/lib/python3.9/site-packages/keras/engine/functional.py:639: UserWarning: Input dict contained keys ['island', 'bill_depth_mm', 'flipper_length_mm', 'sex', 'year'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)
    [INFO 23-05-21 17:18:13.0832 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpzr228e_p/model/ with prefix 8d2a8d3a2f734ffe
    [INFO 23-05-21 17:18:13.0905 CDT decision_forest.cc:660] Model loaded with 300 root(s), 6104 node(s), and 2 input feature(s).
    [INFO 23-05-21 17:18:13.0905 CDT kernel.cc:1074] Use fast generic engine

The following example re-implements the same logic using TensorFlow Feature Columns.

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpx8u535xe as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.169470. Found 243 examples.
    Training model...
    Model trained in 0:00:00.022710
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:18:13.4364 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpx8u535xe/model/ with prefix 103f1d44a2c64388
    [INFO 23-05-21 17:18:13.4439 CDT decision_forest.cc:660] Model loaded with 300 root(s), 6104 node(s), and 2 input feature(s).
    [INFO 23-05-21 17:18:13.4439 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2b66bf0a0>


<a id="org5e31bf5"></a>

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

    2911 examples in training, 1266 examples for testing.

:RESULTS:

    2/2 [==============================] - 0s 16ms/step - loss: 0.0000e+00 - mse: 3.9467
    
    {'loss': 0.0, 'mse': 3.9467318058013916}
    
    MSE: 3.9467318058013916
    RMSE: 1.9866383178126288

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpa3z6cy9q as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.106255. Found 2911 examples.
    Training model...
    [INFO 23-05-21 17:18:14.8723 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpa3z6cy9q/model/ with prefix 0834b7017a474449
    Model trained in 0:00:00.707866
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:18:15.1801 CDT decision_forest.cc:660] Model loaded with 300 root(s), 260928 node(s), and 8 input feature(s).
    [INFO 23-05-21 17:18:15.1801 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2af5c0d60>

:END:


<a id="org224545f"></a>

# Conclusion

This concludes the basic overview of TensorFlow Decision Forest utility.

