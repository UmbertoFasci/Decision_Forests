
# Table of Contents

1.  [Importing Libraries](#org269555e)
2.  [Training a Random Forest model](#orgb64a85f)
3.  [Evaluate the model](#org025d528)
4.  [TensorFlow Serving](#org32ae1eb)
5.  [Model structure and feature importance](#org79b043d)
6.  [Using make<sub>inspector</sub>](#orgd210c57)
7.  [Model self evaluation](#orgd4bab91)
8.  [Plotting the training logs](#orgf69548b)
9.  [Retrain model with different learning algorithm](#org68a5413)
10. [Using a subset of features](#org70b8b49)
11. [Hyper-parameters](#orgc5742a5)
12. [Feature Preprocessing](#org4501007)
13. [Training a regression model](#orgf7e2fab)
14. [Conclusion](#orgdf66e29)

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


<a id="org269555e"></a>

# Importing Libraries

    import tensorflow_decision_forests as tfdf
    
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import math

    print("Found TensorFlow Decision Forests v" + tfdf.__version__)

    Found TensorFlow Decision Forests v1.3.0


<a id="orgb64a85f"></a>

# Training a Random Forest model

    # Download the dataset
    !wget -q https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv -O /tmp/penguins.csv
    
    # Load the dataset into Pandas DataFrame
    dataset_df = pd.read_csv("/tmp/penguins.csv")
    
    # Display the first 3 examples
    dataset_df.head(3)

      species     island  bill_length_mm  bill_depth_mm  flipper_length_mm   
    0  Adelie  Torgersen            39.1           18.7              181.0  \
    1  Adelie  Torgersen            39.5           17.4              186.0   
    2  Adelie  Torgersen            40.3           18.0              195.0   
    
       body_mass_g     sex  year  
    0       3750.0    male  2007  
    1       3800.0  female  2007  
    2       3250.0  female  2007  

    label = "species"
    
    classes = dataset_df[label].unique().tolist()
    print(f"Label classes: {classes}")
    
    dataset_df[label] = dataset_df[label].map(classes.index)

    Label classes: ['Adelie', 'Gentoo', 'Chinstrap']

    def split_dataset(dataset, test_ratio=0.30):
        test_indices = np.random.rand(len(dataset)) < test_ratio
        return dataset[~test_indices], dataset[test_indices]
    
    train_ds_pd, test_ds_pd = split_dataset(dataset_df)
    print("{} examples in training, {} examples for testing.".format(
        len(train_ds_pd), len(test_ds_pd)))

    238 examples in training, 106 examples for testing.

    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

    # Specify the model
    model_1 = tfdf.keras.RandomForestModel(verbose=2)
    
    # Train the model
    model_1.fit(train_ds)

    Use 8 thread(s) for training
    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpe9wfgcid as temporary training directory
    Reading training dataset...
    Training tensor examples:
    Features: {'island': <tf.Tensor 'data:0' shape=(None,) dtype=string>, 'bill_length_mm': <tf.Tensor 'data_1:0' shape=(None,) dtype=float64>, 'bill_depth_mm': <tf.Tensor 'data_2:0' shape=(None,) dtype=float64>, 'flipper_length_mm': <tf.Tensor 'data_3:0' shape=(None,) dtype=float64>, 'body_mass_g': <tf.Tensor 'data_4:0' shape=(None,) dtype=float64>, 'sex': <tf.Tensor 'data_5:0' shape=(None,) dtype=string>, 'year': <tf.Tensor 'data_6:0' shape=(None,) dtype=int64>}
    Label: Tensor("data_7:0", shape=(None,), dtype=int64)
    Weights: None
    Normalized tensor features:
     {'island': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data:0' shape=(None,) dtype=string>), 'bill_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast:0' shape=(None,) dtype=float32>), 'bill_depth_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_1:0' shape=(None,) dtype=float32>), 'flipper_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_2:0' shape=(None,) dtype=float32>), 'body_mass_g': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_3:0' shape=(None,) dtype=float32>), 'sex': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data_5:0' shape=(None,) dtype=string>), 'year': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_4:0' shape=(None,) dtype=float32>)}
    Training dataset read in 0:00:00.107239. Found 238 examples.
    Training model...
    [INFO 23-05-21 17:46:05.7148 CDT kernel.cc:773] Start Yggdrasil model training
    [INFO 23-05-21 17:46:05.7148 CDT kernel.cc:774] Collect training examples
    [INFO 23-05-21 17:46:05.7148 CDT kernel.cc:787] Dataspec guide:
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
    
    [INFO 23-05-21 17:46:05.7149 CDT kernel.cc:393] Number of batches: 1
    [INFO 23-05-21 17:46:05.7149 CDT kernel.cc:394] Number of examples: 238
    [INFO 23-05-21 17:46:05.7150 CDT kernel.cc:794] Training dataset:
    Number of records: 238
    Number of columns: 8
    
    Number of columns by type:
    	NUMERICAL: 5 (62.5%)
    	CATEGORICAL: 3 (37.5%)
    
    Columns:
    
    NUMERICAL: 5 (62.5%)
    	1: "bill_depth_mm" NUMERICAL num-nas:2 (0.840336%) mean:17.097 min:13.1 max:21.5 sd:2.01644
    	2: "bill_length_mm" NUMERICAL num-nas:2 (0.840336%) mean:43.4949 min:32.1 max:59.6 sd:5.41488
    	3: "body_mass_g" NUMERICAL num-nas:2 (0.840336%) mean:4175.85 min:2700 max:6050 sd:781.34
    	4: "flipper_length_mm" NUMERICAL num-nas:2 (0.840336%) mean:200.852 min:172 max:231 sd:14.1337
    	7: "year" NUMERICAL mean:2008.04 min:2007 max:2009 sd:0.803302
    
    CATEGORICAL: 3 (37.5%)
    	0: "__LABEL" CATEGORICAL integerized vocab-size:4 no-ood-item
    	5: "island" CATEGORICAL has-dict vocab-size:4 zero-ood-items most-frequent:"Biscoe" 116 (48.7395%)
    	6: "sex" CATEGORICAL num-nas:10 (4.20168%) has-dict vocab-size:3 zero-ood-items most-frequent:"female" 119 (52.193%)
    
    Terminology:
    	nas: Number of non-available (i.e. missing) values.
    	ood: Out of dictionary.
    	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
    	tokenized: The attribute value is obtained through tokenization.
    	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
    	vocab-size: Number of unique values.
    
    [INFO 23-05-21 17:46:05.7150 CDT kernel.cc:810] Configure learner
    [INFO 23-05-21 17:46:05.7151 CDT kernel.cc:824] Training config:
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
    
    [INFO 23-05-21 17:46:05.7151 CDT kernel.cc:827] Deployment config:
    cache_path: "/var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpe9wfgcid/working_cache"
    num_threads: 8
    try_resume_training: true
    
    [INFO 23-05-21 17:46:05.7152 CDT kernel.cc:889] Train model
    [INFO 23-05-21 17:46:05.7152 CDT random_forest.cc:416] Training random forest on 238 example(s) and 7 feature(s).
    [INFO 23-05-21 17:46:05.7157 CDT random_forest.cc:805] Training of tree  1/300 (tree index:0) done accuracy:0.917526 logloss:2.97267
    [INFO 23-05-21 17:46:05.7160 CDT random_forest.cc:805] Training of tree  11/300 (tree index:11) done accuracy:0.961207 logloss:0.819998
    [INFO 23-05-21 17:46:05.7162 CDT random_forest.cc:805] Training of tree  21/300 (tree index:24) done accuracy:0.957983 logloss:0.232451
    [INFO 23-05-21 17:46:05.7165 CDT random_forest.cc:805] Training of tree  32/300 (tree index:28) done accuracy:0.970588 logloss:0.0910781
    [INFO 23-05-21 17:46:05.7167 CDT random_forest.cc:805] Training of tree  42/300 (tree index:43) done accuracy:0.97479 logloss:0.087278
    [INFO 23-05-21 17:46:05.7169 CDT random_forest.cc:805] Training of tree  52/300 (tree index:53) done accuracy:0.97479 logloss:0.0842678
    [INFO 23-05-21 17:46:05.7172 CDT random_forest.cc:805] Training of tree  63/300 (tree index:63) done accuracy:0.97479 logloss:0.0861653
    [INFO 23-05-21 17:46:05.7175 CDT random_forest.cc:805] Training of tree  73/300 (tree index:71) done accuracy:0.978992 logloss:0.0857139
    [INFO 23-05-21 17:46:05.7177 CDT random_forest.cc:805] Training of tree  83/300 (tree index:82) done accuracy:0.97479 logloss:0.0869588
    [INFO 23-05-21 17:46:05.7179 CDT random_forest.cc:805] Training of tree  93/300 (tree index:93) done accuracy:0.97479 logloss:0.0901721
    [INFO 23-05-21 17:46:05.7182 CDT random_forest.cc:805] Training of tree  103/300 (tree index:95) done accuracy:0.97479 logloss:0.0891007
    [INFO 23-05-21 17:46:05.7184 CDT random_forest.cc:805] Training of tree  115/300 (tree index:117) done accuracy:0.97479 logloss:0.0850493
    [INFO 23-05-21 17:46:05.7186 CDT random_forest.cc:805] Training of tree  125/300 (tree index:125) done accuracy:0.97479 logloss:0.0847583
    [INFO 23-05-21 17:46:05.7189 CDT random_forest.cc:805] Training of tree  135/300 (tree index:133) done accuracy:0.97479 logloss:0.0865018
    [INFO 23-05-21 17:46:05.7191 CDT random_forest.cc:805] Training of tree  145/300 (tree index:141) done accuracy:0.97479 logloss:0.0883725
    [INFO 23-05-21 17:46:05.7195 CDT random_forest.cc:805] Training of tree  155/300 (tree index:158) done accuracy:0.97479 logloss:0.089598
    [INFO 23-05-21 17:46:05.7197 CDT random_forest.cc:805] Training of tree  168/300 (tree index:166) done accuracy:0.97479 logloss:0.0901934
    [INFO 23-05-21 17:46:05.7199 CDT random_forest.cc:805] Training of tree  178/300 (tree index:180) done accuracy:0.97479 logloss:0.090912
    [INFO 23-05-21 17:46:05.7202 CDT random_forest.cc:805] Training of tree  188/300 (tree index:190) done accuracy:0.97479 logloss:0.0914183
    [INFO 23-05-21 17:46:05.7204 CDT random_forest.cc:805] Training of tree  198/300 (tree index:197) done accuracy:0.97479 logloss:0.0916178
    [INFO 23-05-21 17:46:05.7206 CDT random_forest.cc:805] Training of tree  210/300 (tree index:211) done accuracy:0.97479 logloss:0.0924943
    [INFO 23-05-21 17:46:05.7209 CDT random_forest.cc:805] Training of tree  220/300 (tree index:221) done accuracy:0.97479 logloss:0.0927163
    [INFO 23-05-21 17:46:05.7212 CDT random_forest.cc:805] Training of tree  230/300 (tree index:230) done accuracy:0.970588 logloss:0.0928855
    [INFO 23-05-21 17:46:05.7215 CDT random_forest.cc:805] Training of tree  244/300 (tree index:242) done accuracy:0.970588 logloss:0.0916531
    [INFO 23-05-21 17:46:05.7218 CDT random_forest.cc:805] Training of tree  254/300 (tree index:250) done accuracy:0.97479 logloss:0.0921954
    [INFO 23-05-21 17:46:05.7222 CDT random_forest.cc:805] Training of tree  264/300 (tree index:262) done accuracy:0.970588 logloss:0.092925
    [INFO 23-05-21 17:46:05.7223 CDT random_forest.cc:805] Training of tree  274/300 (tree index:273) done accuracy:0.970588 logloss:0.0928888
    [INFO 23-05-21 17:46:05.7226 CDT random_forest.cc:805] Training of tree  284/300 (tree index:286) done accuracy:0.970588 logloss:0.0926265
    [INFO 23-05-21 17:46:05.7229 CDT random_forest.cc:805] Training of tree  295/300 (tree index:292) done accuracy:0.970588 logloss:0.0933569
    [INFO 23-05-21 17:46:05.7230 CDT random_forest.cc:805] Training of tree  300/300 (tree index:297) done accuracy:0.970588 logloss:0.0937341
    [INFO 23-05-21 17:46:05.7231 CDT random_forest.cc:885] Final OOB metrics: accuracy:0.970588 logloss:0.0937341
    [INFO 23-05-21 17:46:05.7233 CDT kernel.cc:926] Export model in log directory: /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpe9wfgcid with prefix 09345d2f577c4a02
    [INFO 23-05-21 17:46:05.7260 CDT kernel.cc:944] Save model in resources
    [INFO 23-05-21 17:46:05.7274 CDT abstract_model.cc:849] Model self evaluation:
    Number of predictions (without weights): 238
    Number of predictions (with weights): 238
    Task: CLASSIFICATION
    Label: __LABEL
    
    Accuracy: 0.970588  CI95[W][0.945468 0.986116]
    LogLoss: : 0.0937341
    ErrorRate: : 0.0294118
    
    Default Accuracy: : 0.462185
    Default LogLoss: : 1.02755
    Default ErrorRate: : 0.537815
    
    Confusion Table:
    truth\prediction
       0    1   2   3
    0  0    0   0   0
    1  0  107   0   3
    2  0    1  86   0
    3  0    3   0  38
    Total: 238
    
    One vs other classes:
    [INFO 23-05-21 17:46:05.7320 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpe9wfgcid/model/ with prefix 09345d2f577c4a02
    [INFO 23-05-21 17:46:05.7372 CDT decision_forest.cc:660] Model loaded with 300 root(s), 4118 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:46:05.7372 CDT abstract_model.cc:1312] Engine "RandomForestGeneric" built
    [INFO 23-05-21 17:46:05.7372 CDT kernel.cc:1074] Use fast generic engine
    Model trained in 0:00:00.025920
    Compiling model...
    Model compiled.

    <keras.callbacks.History at 0x2c96ee2e0>


<a id="org025d528"></a>

# Evaluate the model

    model_1.compile(metrics=["accuracy"])
    evaluation = model_1.evaluate(test_ds, return_dict=True)
    print()
    
    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

    1/1 [==============================] - 0s 79ms/step - loss: 0.0000e+00 - accuracy: 0.9717
    
    
    loss: 0.0000
    accuracy: 0.9717


<a id="org32ae1eb"></a>

# TensorFlow Serving

    model_1.save("/tmp/my_saved_model")

    WARNING:absl:Found untraced functions such as call_get_leaves while saving (showing 1 of 1). These functions will not be directly callable after loading.
    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets
    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets


<a id="org79b043d"></a>

# Model structure and feature importance

    model_1.summary()

    Model: "random_forest_model_16"
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
        1.    "bill_length_mm"  0.468322 ################
        2. "flipper_length_mm"  0.439328 #############
        3.            "island"  0.310042 ####
        4.     "bill_depth_mm"  0.309718 ####
        5.       "body_mass_g"  0.269412 #
        6.               "sex"  0.246070 
        7.              "year"  0.245399 
    
    Variable Importance: NUM_AS_ROOT:
        1. "flipper_length_mm" 147.000000 ################
        2.    "bill_length_mm" 101.000000 ##########
        3.     "bill_depth_mm" 36.000000 ##
        4.            "island" 16.000000 
    
    Variable Importance: NUM_NODES:
        1.    "bill_length_mm" 685.000000 ################
        2.     "bill_depth_mm" 380.000000 ########
        3. "flipper_length_mm" 333.000000 #######
        4.            "island" 245.000000 #####
        5.       "body_mass_g" 224.000000 ####
        6.               "sex" 26.000000 
        7.              "year" 16.000000 
    
    Variable Importance: SUM_SCORE:
        1.    "bill_length_mm" 27069.145823 ################
        2. "flipper_length_mm" 23516.176866 #############
        3.            "island" 9972.388440 #####
        4.     "bill_depth_mm" 7953.189392 ####
        5.       "body_mass_g" 1544.543065 
        6.               "sex" 146.537975 
        7.              "year" 45.239813 
    
    
    
    Winner takes all: true
    Out-of-bag evaluation: accuracy:0.970588 logloss:0.0937341
    Number of trees: 300
    Total number of nodes: 4118
    
    Number of nodes by tree:
    Count: 300 Average: 13.7267 StdDev: 2.76983
    Min: 7 Max: 25 Ignored: 0
    ----------------------------------------------
    [  7,  8)  2   0.67%   0.67%
    [  8,  9)  0   0.00%   0.67%
    [  9, 10) 19   6.33%   7.00% ##
    [ 10, 11)  0   0.00%   7.00%
    [ 11, 12) 65  21.67%  28.67% ########
    [ 12, 13)  0   0.00%  28.67%
    [ 13, 14) 76  25.33%  54.00% #########
    [ 14, 15)  0   0.00%  54.00%
    [ 15, 16) 86  28.67%  82.67% ##########
    [ 16, 17)  0   0.00%  82.67%
    [ 17, 18) 32  10.67%  93.33% ####
    [ 18, 19)  0   0.00%  93.33%
    [ 19, 20) 14   4.67%  98.00% ##
    [ 20, 21)  0   0.00%  98.00%
    [ 21, 22)  5   1.67%  99.67% #
    [ 22, 23)  0   0.00%  99.67%
    [ 23, 24)  0   0.00%  99.67%
    [ 24, 25)  0   0.00%  99.67%
    [ 25, 25]  1   0.33% 100.00%
    
    Depth by leafs:
    Count: 2209 Average: 3.1589 StdDev: 0.939845
    Min: 1 Max: 7 Ignored: 0
    ----------------------------------------------
    [ 1, 2)  10   0.45%   0.45%
    [ 2, 3) 599  27.12%  27.57% #######
    [ 3, 4) 808  36.58%  64.15% ##########
    [ 4, 5) 633  28.66%  92.80% ########
    [ 5, 6) 142   6.43%  99.23% ##
    [ 6, 7)  15   0.68%  99.91%
    [ 7, 7]   2   0.09% 100.00%
    
    Number of training obs by leaf:
    Count: 2209 Average: 32.3223 StdDev: 33.3828
    Min: 5 Max: 122 Ignored: 0
    ----------------------------------------------
    [   5,  10) 1070  48.44%  48.44% ##########
    [  10,  16)   86   3.89%  52.33% #
    [  16,  22)   70   3.17%  55.50% #
    [  22,  28)   76   3.44%  58.94% #
    [  28,  34)  118   5.34%  64.28% #
    [  34,  40)  119   5.39%  69.67% #
    [  40,  46)   63   2.85%  72.52% #
    [  46,  52)   19   0.86%  73.38%
    [  52,  58)   10   0.45%  73.83%
    [  58,  64)   31   1.40%  75.24%
    [  64,  69)   36   1.63%  76.87%
    [  69,  75)   66   2.99%  79.86% #
    [  75,  81)   88   3.98%  83.84% #
    [  81,  87)  102   4.62%  88.46% #
    [  87,  93)  111   5.02%  93.48% #
    [  93,  99)   65   2.94%  96.42% #
    [  99, 105)   50   2.26%  98.69%
    [ 105, 111)   18   0.81%  99.50%
    [ 111, 117)    8   0.36%  99.86%
    [ 117, 122]    3   0.14% 100.00%
    
    Attribute in nodes:
    	685 : bill_length_mm [NUMERICAL]
    	380 : bill_depth_mm [NUMERICAL]
    	333 : flipper_length_mm [NUMERICAL]
    	245 : island [CATEGORICAL]
    	224 : body_mass_g [NUMERICAL]
    	26 : sex [CATEGORICAL]
    	16 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 0:
    	147 : flipper_length_mm [NUMERICAL]
    	101 : bill_length_mm [NUMERICAL]
    	36 : bill_depth_mm [NUMERICAL]
    	16 : island [CATEGORICAL]
    
    Attribute in nodes with depth <= 1:
    	269 : bill_length_mm [NUMERICAL]
    	237 : flipper_length_mm [NUMERICAL]
    	186 : bill_depth_mm [NUMERICAL]
    	150 : island [CATEGORICAL]
    	48 : body_mass_g [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	513 : bill_length_mm [NUMERICAL]
    	299 : flipper_length_mm [NUMERICAL]
    	288 : bill_depth_mm [NUMERICAL]
    	217 : island [CATEGORICAL]
    	147 : body_mass_g [NUMERICAL]
    	4 : sex [CATEGORICAL]
    	3 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 3:
    	654 : bill_length_mm [NUMERICAL]
    	362 : bill_depth_mm [NUMERICAL]
    	328 : flipper_length_mm [NUMERICAL]
    	242 : island [CATEGORICAL]
    	206 : body_mass_g [NUMERICAL]
    	19 : sex [CATEGORICAL]
    	14 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 5:
    	685 : bill_length_mm [NUMERICAL]
    	379 : bill_depth_mm [NUMERICAL]
    	333 : flipper_length_mm [NUMERICAL]
    	245 : island [CATEGORICAL]
    	224 : body_mass_g [NUMERICAL]
    	26 : sex [CATEGORICAL]
    	16 : year [NUMERICAL]
    
    Condition type in nodes:
    	1638 : HigherCondition
    	271 : ContainsBitmapCondition
    Condition type in nodes with depth <= 0:
    	284 : HigherCondition
    	16 : ContainsBitmapCondition
    Condition type in nodes with depth <= 1:
    	740 : HigherCondition
    	150 : ContainsBitmapCondition
    Condition type in nodes with depth <= 2:
    	1250 : HigherCondition
    	221 : ContainsBitmapCondition
    Condition type in nodes with depth <= 3:
    	1564 : HigherCondition
    	261 : ContainsBitmapCondition
    Condition type in nodes with depth <= 5:
    	1637 : HigherCondition
    	271 : ContainsBitmapCondition
    Node format: NOT_SET
    
    Training OOB:
    	trees: 1, Out-of-bag evaluation: accuracy:0.917526 logloss:2.97267
    	trees: 11, Out-of-bag evaluation: accuracy:0.961207 logloss:0.819998
    	trees: 21, Out-of-bag evaluation: accuracy:0.957983 logloss:0.232451
    	trees: 32, Out-of-bag evaluation: accuracy:0.970588 logloss:0.0910781
    	trees: 42, Out-of-bag evaluation: accuracy:0.97479 logloss:0.087278
    	trees: 52, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0842678
    	trees: 63, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0861653
    	trees: 73, Out-of-bag evaluation: accuracy:0.978992 logloss:0.0857139
    	trees: 83, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0869588
    	trees: 93, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0901721
    	trees: 103, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0891007
    	trees: 115, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0850493
    	trees: 125, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0847583
    	trees: 135, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0865018
    	trees: 145, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0883725
    	trees: 155, Out-of-bag evaluation: accuracy:0.97479 logloss:0.089598
    	trees: 168, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0901934
    	trees: 178, Out-of-bag evaluation: accuracy:0.97479 logloss:0.090912
    	trees: 188, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0914183
    	trees: 198, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0916178
    	trees: 210, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0924943
    	trees: 220, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0927163
    	trees: 230, Out-of-bag evaluation: accuracy:0.970588 logloss:0.0928855
    	trees: 244, Out-of-bag evaluation: accuracy:0.970588 logloss:0.0916531
    	trees: 254, Out-of-bag evaluation: accuracy:0.97479 logloss:0.0921954
    	trees: 264, Out-of-bag evaluation: accuracy:0.970588 logloss:0.092925
    	trees: 274, Out-of-bag evaluation: accuracy:0.970588 logloss:0.0928888
    	trees: 284, Out-of-bag evaluation: accuracy:0.970588 logloss:0.0926265
    	trees: 295, Out-of-bag evaluation: accuracy:0.970588 logloss:0.0933569
    	trees: 300, Out-of-bag evaluation: accuracy:0.970588 logloss:0.0937341


<a id="orgd210c57"></a>

# Using make<sub>inspector</sub>

    model_1.make_inspector().features()

    '("bill_depth_mm" (1; #1) 
     "bill_length_mm" (1; #2) 
     "body_mass_g" (1; #3) 
     "flipper_length_mm" (1; #4) 
     "island" (4; #5) 
     "sex" (4; #6) 
     "year" (1; #7))

    model_1.make_inspector().variable_importances()

    '("NUM_NODES": (("bill_length_mm" (1; #2)  685.0) 
      ("bill_depth_mm" (1; #1)  380.0) 
      ("flipper_length_mm" (1; #4)  333.0) 
      ("island" (4; #5)  245.0) 
      ("body_mass_g" (1; #3)  224.0) 
      ("sex" (4; #6)  26.0) 
      ("year" (1; #7)  16.0)) 
     "SUM_SCORE": (("bill_length_mm" (1; #2)  27069.14582312759) 
      ("flipper_length_mm" (1; #4)  23516.17686584592) 
      ("island" (4; #5)  9972.388440005481) 
      ("bill_depth_mm" (1; #1)  7953.189392188564) 
      ("body_mass_g" (1; #3)  1544.5430652815849) 
      ("sex" (4; #6)  146.53797520697117) 
      ("year" (1; #7)  45.23981338739395)) 
     "NUM_AS_ROOT": (("flipper_length_mm" (1; #4)  147.0) 
      ("bill_length_mm" (1; #2)  101.0) 
      ("bill_depth_mm" (1; #1)  36.0) 
      ("island" (4; #5)  16.0)) 
     "INV_MEAN_MIN_DEPTH": (("bill_length_mm" (1; #2)  0.4683219561395876) 
      ("flipper_length_mm" (1; #4)  0.4393284550758127) 
      ("island" (4; #5)  0.31004197086925284) 
      ("bill_depth_mm" (1; #1)  0.30971777507778236) 
      ("body_mass_g" (1; #3)  0.2694122781916975) 
      ("sex" (4; #6)  0.24606978136394508) 
      ("year" (1; #7)  0.24539932893402994)))


<a id="orgd4bab91"></a>

# Model self evaluation

    model_1.make_inspector().evaluation()

    Evaluation(num_examples=238, accuracy=0.9705882352941176, loss=0.09373412305340484, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)


<a id="orgf69548b"></a>

# Plotting the training logs

    model_1.make_inspector().training_logs()

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
</colgroup>
<tbody>
<tr>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=1 evaluation=Evaluation (num<sub>examples</sub>=97 accuracy=0.9175257731958762 loss=2.972672295324581 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=11 evaluation=Evaluation (num<sub>examples</sub>=232 accuracy=0.9612068965517241 loss=0.8199979668675825 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=21 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.957983193277311 loss=0.23245143949860284 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=32 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9705882352941176 loss=0.09107812112119018 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=42 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.08727803964073919 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=52 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.08426781418193288 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=63 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.08616531408634506 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=73 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9789915966386554 loss=0.0857138522282368 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=83 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.08695884272769219 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=93 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.09017215275495243 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=103 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.08910066165624797 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=115 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.0850493421684168 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=125 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.08475826460742901 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=135 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.08650181885464352 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=145 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.08837252333290688 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=155 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.08959802707751133 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=168 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.09019344003025849 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=178 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.0909120119372461 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=188 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.09141832443361148 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=198 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.09161780661112871 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=210 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.09249428587284785 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=220 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.09271626073491424 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=230 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9705882352941176 loss=0.09288551964789253 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=244 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9705882352941176 loss=0.09165309573828924 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=254 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9747899159663865 loss=0.0921954356071328 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=264 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9705882352941176 loss=0.0929249894038281 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=274 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9705882352941176 loss=0.09288884070897553 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=284 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9705882352941176 loss=0.09262651266498852 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=295 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9705882352941176 loss=0.09335688060392164 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
<td class="org-left">TrainLog</td>
<td class="org-left">(num<sub>trees</sub>=300 evaluation=Evaluation (num<sub>examples</sub>=238 accuracy=0.9705882352941176 loss=0.09373412305340484 rmse=None ndcg=None aucs=None auuc=None qini=None))</td>
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

![img](./.ob-jupyter/4d4a0989b568a975de1690e6a573c0b285317be3.png)


<a id="org68a5413"></a>

# Retrain model with different learning algorithm

    tfdf.keras.get_all_models()

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


<a id="org70b8b49"></a>

# Using a subset of features

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

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp88y50vit as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.080752. Found 238 examples.
    Reading validation dataset...
    Num validation examples: tf.Tensor(106, shape=(), dtype=int32)
    Validation dataset read in 0:00:00.059259. Found 106 examples.
    Training model...
    [WARNING 23-05-21 17:46:07.7763 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:07.7763 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:07.7763 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:00.057390
    Compiling model...
    Model compiled.
    1/1 [==============================] - 0s 47ms/step - loss: 0.0000e+00 - accuracy: 0.9528
    {'loss': 0.0, 'accuracy': 0.9528301954269409}
    [INFO 23-05-21 17:46:07.9791 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp88y50vit/model/ with prefix d09ddd50cfc94e8a
    [INFO 23-05-21 17:46:07.9825 CDT decision_forest.cc:660] Model loaded with 87 root(s), 2691 node(s), and 2 input feature(s).
    [INFO 23-05-21 17:46:07.9825 CDT kernel.cc:1074] Use fast generic engine

**TF-DF** attaches a **semantics** to each feature. This semantics controls how the feature is used by the model. The following semantics are currently supported.

-   **Numerical**: Generally for quantities or counts with full ordering. For example, the age of a person, or the number of items in a bag. Can be a float or an integer. Missing values are represented with a float(Nan) or with an empty sparse tensor.
-   **Categorical**: Generally for a type/class in finite set of possible values without ordering. For example, the color RED in the set {RED, BLUE, GREEN}. Can be a string or an integer. Missing values are represented as &ldquo;&rdquo; (empty string), value -2 or with an empty sparse tensor.
-   **Categorical-Set**: A set of categorical values. Great to represent tokenized text. Can be a string or an integer in a sparse tensor or a ragged tensor (recommended). The order/index of each item doesnt matter.
    
    If not specified, the semantics is inferred from the representation type and shown in the training logs:
    
    -   int, float (dense or sparse) -> Numerical semantics
    
    -   str, (dense or sparse) -> Categorical semantics
    
    -   int, str (ragged) -> Categorical-Set semantics

In some cases, the inferred semantics is incorrect. For example: An Enum stored as an integer is semantically categorical, but it will be detected as numerical. In this case, you should specify the semantic argument in the input. The education<sub>num</sub> field of the Adult dataset is a classic example.

    feature_1 = tfdf.keras.FeatureUsage(name="year", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
    feature_2 = tfdf.keras.FeatureUsage(name="bill_length_mm")
    feature_3 = tfdf.keras.FeatureUsage(name="sex")
    all_features = [feature_1, feature_2, feature_3]
    
    model_3 = tfdf.keras.GradientBoostedTreesModel(features=all_features, exclude_non_specified_features=True)
    model_3.compile(metrics=["accuracy"])
    
    model_3.fit(train_ds, validation_data=test_ds)

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpegaef5no as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.075301. Found 238 examples.
    Reading validation dataset...
    Num validation examples: tf.Tensor(106, shape=(), dtype=int32)
    Validation dataset read in 0:00:00.064907. Found 106 examples.
    Training model...
    Model trained in 0:00:00.045418
    Compiling model...
    [WARNING 23-05-21 17:46:08.2754 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:08.2755 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:08.2755 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    [INFO 23-05-21 17:46:08.4663 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpegaef5no/model/ with prefix c2e66e97e6314421
    [INFO 23-05-21 17:46:08.4677 CDT decision_forest.cc:660] Model loaded with 33 root(s), 1069 node(s), and 3 input feature(s).
    [INFO 23-05-21 17:46:08.4678 CDT kernel.cc:1074] Use fast generic engine
    Model compiled.

    <keras.callbacks.History at 0x2cc7aae50>

Note that `year` is in the list of CATEGORICAL features (unlike the first run)


<a id="orgc5742a5"></a>

# Hyper-parameters

**Hyper-parameters** are paramters of the training algorithm that impact the quality of the final model. They are specified in the model class constructor. The list of hyper-parameters is visible with the *question mark* colab command.

**I will figure out how to obtain that list without the question mark command.**

    # A classical but slightly more complex model.
    model_6 = tfdf.keras.GradientBoostedTreesModel(
        num_trees=500, growing_strategy="BEST_FIRST_GLOBAL", max_depth=8)
    
    model_6.fit(train_ds)

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpwnmhau3b as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.083562. Found 238 examples.
    Training model...
    [WARNING 23-05-21 17:46:08.6633 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:08.6633 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:08.6633 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:02.133292
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:46:10.7792 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpwnmhau3b/model/ with prefix ec5f7421561b49bf
    [INFO 23-05-21 17:46:10.8794 CDT decision_forest.cc:660] Model loaded with 1500 root(s), 84402 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:46:10.8794 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2cdcd09a0>

    model_6.summary()

    Model: "gradient_boosted_trees_model_22"
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
        1.    "bill_length_mm"  0.371781 ################
        2.     "bill_depth_mm"  0.355530 ##############
        3. "flipper_length_mm"  0.260212 #######
        4.            "island"  0.250166 #######
        5.       "body_mass_g"  0.184921 ##
        6.              "year"  0.178427 #
        7.               "sex"  0.152791 
    
    Variable Importance: NUM_AS_ROOT:
        1.            "island" 538.000000 ################
        2.    "bill_length_mm" 500.000000 ########
        3. "flipper_length_mm" 462.000000 
    
    Variable Importance: NUM_NODES:
        1.    "bill_length_mm" 11206.000000 ################
        2.     "bill_depth_mm" 10907.000000 ###############
        3. "flipper_length_mm" 6955.000000 #########
        4.       "body_mass_g" 6629.000000 #########
        5.              "year" 3424.000000 ####
        6.            "island" 1873.000000 ##
        7.               "sex" 457.000000 
    
    Variable Importance: SUM_SCORE:
        1.    "bill_length_mm" 293.398818 ################
        2. "flipper_length_mm" 214.741815 ###########
        3.            "island" 78.523861 ####
        4.     "bill_depth_mm" 21.879692 #
        5.       "body_mass_g"  1.044086 
        6.              "year"  0.185587 
        7.               "sex"  0.014236 
    
    
    
    Loss: MULTINOMIAL_LOG_LIKELIHOOD
    Validation loss value: 5.39856e-06
    Number of trees per iteration: 3
    Node format: NOT_SET
    Number of trees: 1500
    Total number of nodes: 84402
    
    Number of nodes by tree:
    Count: 1500 Average: 56.268 StdDev: 5.68883
    Min: 13 Max: 61 Ignored: 0
    ----------------------------------------------
    [ 13, 15)   1   0.07%   0.07%
    [ 15, 17)   2   0.13%   0.20%
    [ 17, 20)   0   0.00%   0.20%
    [ 20, 22)   0   0.00%   0.20%
    [ 22, 25)   0   0.00%   0.20%
    [ 25, 27)   0   0.00%   0.20%
    [ 27, 30)   1   0.07%   0.27%
    [ 30, 32)   0   0.00%   0.27%
    [ 32, 35)   1   0.07%   0.33%
    [ 35, 37)   0   0.00%   0.33%
    [ 37, 39)   1   0.07%   0.40%
    [ 39, 42)   6   0.40%   0.80%
    [ 42, 44)  24   1.60%   2.40%
    [ 44, 47)  38   2.53%   4.93%
    [ 47, 49)  56   3.73%   8.67% #
    [ 49, 52) 230  15.33%  24.00% ###
    [ 52, 54) 113   7.53%  31.53% #
    [ 54, 57) 150  10.00%  41.53% ##
    [ 57, 59) 105   7.00%  48.53% #
    [ 59, 61] 772  51.47% 100.00% ##########
    
    Depth by leafs:
    Count: 42951 Average: 5.69186 StdDev: 1.58152
    Min: 2 Max: 8 Ignored: 0
    ----------------------------------------------
    [ 2, 3)   803   1.87%   1.87% #
    [ 3, 4)  2684   6.25%   8.12% ###
    [ 4, 5)  7828  18.23%  26.34% ########
    [ 5, 6)  7545  17.57%  43.91% #######
    [ 6, 7) 10069  23.44%  67.35% ##########
    [ 7, 8)  6814  15.86%  83.22% #######
    [ 8, 8]  7208  16.78% 100.00% #######
    
    Number of training obs by leaf:
    Count: 42951 Average: 0 StdDev: 0
    Min: 0 Max: 0 Ignored: 0
    ----------------------------------------------
    [ 0, 0] 42951 100.00% 100.00% ##########
    
    Attribute in nodes:
    	11206 : bill_length_mm [NUMERICAL]
    	10907 : bill_depth_mm [NUMERICAL]
    	6955 : flipper_length_mm [NUMERICAL]
    	6629 : body_mass_g [NUMERICAL]
    	3424 : year [NUMERICAL]
    	1873 : island [CATEGORICAL]
    	457 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 0:
    	538 : island [CATEGORICAL]
    	500 : bill_length_mm [NUMERICAL]
    	462 : flipper_length_mm [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	1526 : bill_depth_mm [NUMERICAL]
    	1472 : bill_length_mm [NUMERICAL]
    	691 : island [CATEGORICAL]
    	582 : flipper_length_mm [NUMERICAL]
    	126 : body_mass_g [NUMERICAL]
    	103 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	3581 : bill_depth_mm [NUMERICAL]
    	2198 : bill_length_mm [NUMERICAL]
    	1226 : island [CATEGORICAL]
    	1166 : flipper_length_mm [NUMERICAL]
    	804 : body_mass_g [NUMERICAL]
    	648 : year [NUMERICAL]
    	74 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 3:
    	5360 : bill_depth_mm [NUMERICAL]
    	4881 : bill_length_mm [NUMERICAL]
    	2316 : flipper_length_mm [NUMERICAL]
    	1923 : body_mass_g [NUMERICAL]
    	1551 : island [CATEGORICAL]
    	1031 : year [NUMERICAL]
    	345 : sex [CATEGORICAL]
    
    Attribute in nodes with depth <= 5:
    	8798 : bill_length_mm [NUMERICAL]
    	8714 : bill_depth_mm [NUMERICAL]
    	5138 : flipper_length_mm [NUMERICAL]
    	5098 : body_mass_g [NUMERICAL]
    	2765 : year [NUMERICAL]
    	1690 : island [CATEGORICAL]
    	435 : sex [CATEGORICAL]
    
    Condition type in nodes:
    	39121 : HigherCondition
    	2330 : ContainsBitmapCondition
    Condition type in nodes with depth <= 0:
    	962 : HigherCondition
    	538 : ContainsBitmapCondition
    Condition type in nodes with depth <= 1:
    	3809 : HigherCondition
    	691 : ContainsBitmapCondition
    Condition type in nodes with depth <= 2:
    	8397 : HigherCondition
    	1300 : ContainsBitmapCondition
    Condition type in nodes with depth <= 3:
    	15511 : HigherCondition
    	1896 : ContainsBitmapCondition
    Condition type in nodes with depth <= 5:
    	30513 : HigherCondition
    	2125 : ContainsBitmapCondition
    
    Training logs:
    Number of iteration to final model: 500
    	Iter:1 train-loss:0.917350 valid-loss:0.910603  train-accuracy:0.985714 valid-accuracy:1.000000
    	Iter:2 train-loss:0.776164 valid-loss:0.765260  train-accuracy:0.985714 valid-accuracy:1.000000
    	Iter:3 train-loss:0.663153 valid-loss:0.649719  train-accuracy:0.985714 valid-accuracy:1.000000
    	Iter:4 train-loss:0.570034 valid-loss:0.555627  train-accuracy:0.985714 valid-accuracy:1.000000
    	Iter:5 train-loss:0.493178 valid-loss:0.477276  train-accuracy:0.985714 valid-accuracy:1.000000
    	Iter:6 train-loss:0.428267 valid-loss:0.411905  train-accuracy:0.990476 valid-accuracy:1.000000
    	Iter:16 train-loss:0.115784 valid-loss:0.109536  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:26 train-loss:0.033945 valid-loss:0.035785  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:36 train-loss:0.009931 valid-loss:0.011811  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:46 train-loss:0.002791 valid-loss:0.003380  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:56 train-loss:0.000798 valid-loss:0.001090  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:66 train-loss:0.000242 valid-loss:0.000333  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:76 train-loss:0.000097 valid-loss:0.000140  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:86 train-loss:0.000057 valid-loss:0.000086  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:96 train-loss:0.000040 valid-loss:0.000062  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:106 train-loss:0.000031 valid-loss:0.000047  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:116 train-loss:0.000025 valid-loss:0.000039  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:126 train-loss:0.000021 valid-loss:0.000033  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:136 train-loss:0.000018 valid-loss:0.000029  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:146 train-loss:0.000016 valid-loss:0.000026  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:156 train-loss:0.000014 valid-loss:0.000023  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:166 train-loss:0.000013 valid-loss:0.000021  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:176 train-loss:0.000012 valid-loss:0.000019  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:186 train-loss:0.000011 valid-loss:0.000018  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:196 train-loss:0.000010 valid-loss:0.000017  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:206 train-loss:0.000009 valid-loss:0.000016  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:216 train-loss:0.000009 valid-loss:0.000015  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:226 train-loss:0.000008 valid-loss:0.000014  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:236 train-loss:0.000008 valid-loss:0.000013  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:246 train-loss:0.000007 valid-loss:0.000012  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:256 train-loss:0.000007 valid-loss:0.000012  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:266 train-loss:0.000007 valid-loss:0.000011  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:276 train-loss:0.000006 valid-loss:0.000010  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:286 train-loss:0.000006 valid-loss:0.000010  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:296 train-loss:0.000006 valid-loss:0.000010  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:306 train-loss:0.000005 valid-loss:0.000009  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:316 train-loss:0.000005 valid-loss:0.000009  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:326 train-loss:0.000005 valid-loss:0.000009  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:336 train-loss:0.000005 valid-loss:0.000008  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:346 train-loss:0.000005 valid-loss:0.000008  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:356 train-loss:0.000005 valid-loss:0.000008  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:366 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:376 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:386 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:396 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:406 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:416 train-loss:0.000004 valid-loss:0.000007  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:426 train-loss:0.000004 valid-loss:0.000006  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:436 train-loss:0.000004 valid-loss:0.000006  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:446 train-loss:0.000004 valid-loss:0.000006  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:456 train-loss:0.000003 valid-loss:0.000006  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:466 train-loss:0.000003 valid-loss:0.000006  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:476 train-loss:0.000003 valid-loss:0.000006  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:486 train-loss:0.000003 valid-loss:0.000006  train-accuracy:1.000000 valid-accuracy:1.000000
    	Iter:496 train-loss:0.000003 valid-loss:0.000005  train-accuracy:1.000000 valid-accuracy:1.000000

    # A more complex, but possibly, more accurate model.
    model_7 = tfdf.keras.GradientBoostedTreesModel(
        num_trees=500,
        growing_strategy="BEST_FIRST_GLOBAL",
        max_depth=8,
        split_axis="SPARSE_OBLIQUE",
        categorical_algorithm="RANDOM",
        )
    
    model_7.fit(train_ds)

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp521op7ij as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.096454. Found 238 examples.
    Training model...
    [WARNING 23-05-21 17:46:11.2374 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:11.2375 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:11.2375 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:03.757683
    Compiling model...
    [INFO 23-05-21 17:46:14.9671 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp521op7ij/model/ with prefix 989c093b34dc46b0
    [INFO 23-05-21 17:46:15.0785 CDT decision_forest.cc:660] Model loaded with 1500 root(s), 86570 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:46:15.0785 CDT kernel.cc:1074] Use fast generic engine
    Model compiled.

    <keras.callbacks.History at 0x2d10e0b20>

As new training methods are published and implemented, combinations of hyper-parameters can emerge as good or almost-always-better than the default parameters. To avoid changing the default hyper-parameter values these good combinations are indexed and availale as hyper-parameter templates.

For example, the benchmark<sub>rank1</sub> template is the best combination on our internal benchmarks. Those templates are versioned to allow training configuration stability e.g. benchmark<sub>rank1</sub>@v1.

    # A good template of hyper-parameters.
    model_8 = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1")
    model_8.fit(train_ds)

    Resolve hyper-parameter template "benchmark_rank1" to "benchmark_rank1@v1" -> {'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}.
    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp4c80d5uf as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.084818. Found 238 examples.
    Training model...
    [WARNING 23-05-21 17:46:15.2974 CDT gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:15.2974 CDT gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-21 17:46:15.2974 CDT gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    Model trained in 0:00:01.057409
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:46:16.3963 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp4c80d5uf/model/ with prefix 31a6eec0383045f0
    [INFO 23-05-21 17:46:16.4421 CDT decision_forest.cc:660] Model loaded with 900 root(s), 36330 node(s), and 7 input feature(s).
    [INFO 23-05-21 17:46:16.4421 CDT abstract_model.cc:1312] Engine "GradientBoostedTreesGeneric" built
    [INFO 23-05-21 17:46:16.4421 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2d16adbb0>

The available templates are available with `predefined_hyperparameters`. Note that different learning algorithms have different templates, even if the name is similar.

    print(tfdf.keras.GradientBoostedTreesModel.predefined_hyperparameters())

    [HyperParameterTemplate(name='better_default', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL'}, description='A configuration that is generally better than the default parameters without being more expensive.'), HyperParameterTemplate(name='benchmark_rank1', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}, description='Top ranking hyper-parameters on our benchmark slightly modified to run in reasonable time.')]

What is returned are the predefined hyper-parameters of the Gradient Boosted Tree model.


<a id="org4501007"></a>

# Feature Preprocessing

Pre-processing features is sometimes necessary to consume signals with complex structures, to regularize the model or to apply transfer learning. Pre-processing can be done in one of three ways:

1.  **Preprocessing on the pandas dataframe**: This solution is easy tto implement and generally suitable for experiementation. However, the pre-processing logic will not be exported in the model by model.save()
2.  **Keras Preprocessing**: While more complex than the previous solution, Keras Preprocessing is packaged in the model.
3.  **TensorFlow Feature Columns**: This API is part of the TF Estimator library (!= Keras) and planned for deprecation. This solution is interesting when using existing preprocessing code.

**Note**: Using **TensorFlow Hub** pre-trained embedding is often, a great way to consume text and image with TF-DF.

In the next example, pre-process the body<sub>mass</sub><sub>g</sub> feature into body<sub>mass</sub><sub>kg</sub> = body<sub>mass</sub><sub>g</sub> / 1000. The bill<sub>length</sub><sub>mm</sub> is consumed without preprocessing. Note that such monotonic transformations have generally no impact on decision forest models.

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

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpim1_40_4 as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.085818. Found 238 examples.
    Training model...
    Model trained in 0:00:00.022079
    Compiling model...
    Model compiled.
    Model: "random_forest_model_17"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model_4 (Functional)        {'body_mass_kg': (None,   0         
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
        1. "bill_length_mm"  0.996678 ################
        2.   "body_mass_kg"  0.422931 
    
    Variable Importance: NUM_AS_ROOT:
        1. "bill_length_mm" 299.000000 ################
        2.   "body_mass_kg"  1.000000 
    
    Variable Importance: NUM_NODES:
        1. "bill_length_mm" 1535.000000 ################
        2.   "body_mass_kg" 1188.000000 
    
    Variable Importance: SUM_SCORE:
        1. "bill_length_mm" 46431.641759 ################
        2.   "body_mass_kg" 21274.587792 
    
    
    
    Winner takes all: true
    Out-of-bag evaluation: accuracy:0.932773 logloss:0.629315
    Number of trees: 300
    Total number of nodes: 5746
    
    Number of nodes by tree:
    Count: 300 Average: 19.1533 StdDev: 2.81244
    Min: 11 Max: 27 Ignored: 0
    ----------------------------------------------
    [ 11, 12)  1   0.33%   0.33%
    [ 12, 13)  0   0.00%   0.33%
    [ 13, 14)  7   2.33%   2.67% #
    [ 14, 15)  0   0.00%   2.67%
    [ 15, 16) 30  10.00%  12.67% ###
    [ 16, 17)  0   0.00%  12.67%
    [ 17, 18) 63  21.00%  33.67% #######
    [ 18, 19)  0   0.00%  33.67%
    [ 19, 20) 91  30.33%  64.00% ##########
    [ 20, 21)  0   0.00%  64.00%
    [ 21, 22) 60  20.00%  84.00% #######
    [ 22, 23)  0   0.00%  84.00%
    [ 23, 24) 36  12.00%  96.00% ####
    [ 24, 25)  0   0.00%  96.00%
    [ 25, 26)  9   3.00%  99.00% #
    [ 26, 27)  0   0.00%  99.00%
    [ 27, 27]  3   1.00% 100.00%
    
    Depth by leafs:
    Count: 3023 Average: 3.79722 StdDev: 1.21936
    Min: 1 Max: 7 Ignored: 0
    ----------------------------------------------
    [ 1, 2)  10   0.33%   0.33%
    [ 2, 3) 462  15.28%  15.61% #####
    [ 3, 4) 813  26.89%  42.51% #########
    [ 4, 5) 907  30.00%  72.51% ##########
    [ 5, 6) 542  17.93%  90.44% ######
    [ 6, 7) 255   8.44%  98.88% ###
    [ 7, 7]  34   1.12% 100.00%
    
    Number of training obs by leaf:
    Count: 3023 Average: 23.6189 StdDev: 29.3231
    Min: 5 Max: 117 Ignored: 0
    ----------------------------------------------
    [   5,  10) 1917  63.41%  63.41% ##########
    [  10,  16)  179   5.92%  69.34% #
    [  16,  21)   23   0.76%  70.10%
    [  21,  27)   60   1.98%  72.08%
    [  27,  33)  122   4.04%  76.12% #
    [  33,  38)   89   2.94%  79.06%
    [  38,  44)   33   1.09%  80.15%
    [  44,  50)    9   0.30%  80.45%
    [  50,  55)   14   0.46%  80.91%
    [  55,  61)   32   1.06%  81.97%
    [  61,  67)   73   2.41%  84.39%
    [  67,  72)   80   2.65%  87.03%
    [  72,  78)   79   2.61%  89.65%
    [  78,  84)   66   2.18%  91.83%
    [  84,  89)   63   2.08%  93.91%
    [  89,  95)   93   3.08%  96.99%
    [  95, 101)   53   1.75%  98.74%
    [ 101, 106)   26   0.86%  99.60%
    [ 106, 112)    7   0.23%  99.83%
    [ 112, 117]    5   0.17% 100.00%
    
    Attribute in nodes:
    	1535 : bill_length_mm [NUMERICAL]
    	1188 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 0:
    	299 : bill_length_mm [NUMERICAL]
    	1 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	548 : bill_length_mm [NUMERICAL]
    	342 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	952 : bill_length_mm [NUMERICAL]
    	656 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 3:
    	1245 : bill_length_mm [NUMERICAL]
    	986 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 5:
    	1528 : bill_length_mm [NUMERICAL]
    	1178 : body_mass_kg [NUMERICAL]
    
    Condition type in nodes:
    	2723 : HigherCondition
    Condition type in nodes with depth <= 0:
    	300 : HigherCondition
    Condition type in nodes with depth <= 1:
    	890 : HigherCondition
    Condition type in nodes with depth <= 2:
    	1608 : HigherCondition
    Condition type in nodes with depth <= 3:
    	2231 : HigherCondition
    Condition type in nodes with depth <= 5:
    	2706 : HigherCondition
    Node format: NOT_SET
    
    Training OOB:
    	trees: 1, Out-of-bag evaluation: accuracy:0.907216 logloss:3.34426
    	trees: 11, Out-of-bag evaluation: accuracy:0.909091 logloss:1.9347
    	trees: 22, Out-of-bag evaluation: accuracy:0.92437 logloss:1.59223
    	trees: 32, Out-of-bag evaluation: accuracy:0.920168 logloss:1.45826
    	trees: 45, Out-of-bag evaluation: accuracy:0.920168 logloss:1.17812
    	trees: 57, Out-of-bag evaluation: accuracy:0.928571 logloss:1.03782
    	trees: 68, Out-of-bag evaluation: accuracy:0.932773 logloss:0.894137
    	trees: 78, Out-of-bag evaluation: accuracy:0.928571 logloss:0.892783
    	trees: 91, Out-of-bag evaluation: accuracy:0.928571 logloss:0.759772
    	trees: 101, Out-of-bag evaluation: accuracy:0.928571 logloss:0.762015
    	trees: 112, Out-of-bag evaluation: accuracy:0.928571 logloss:0.757229
    	trees: 122, Out-of-bag evaluation: accuracy:0.928571 logloss:0.758477
    	trees: 132, Out-of-bag evaluation: accuracy:0.928571 logloss:0.760109
    	trees: 145, Out-of-bag evaluation: accuracy:0.928571 logloss:0.759074
    	trees: 155, Out-of-bag evaluation: accuracy:0.928571 logloss:0.754636
    	trees: 165, Out-of-bag evaluation: accuracy:0.928571 logloss:0.756061
    	trees: 176, Out-of-bag evaluation: accuracy:0.928571 logloss:0.756819
    	trees: 187, Out-of-bag evaluation: accuracy:0.928571 logloss:0.756277
    	trees: 200, Out-of-bag evaluation: accuracy:0.928571 logloss:0.625045
    	trees: 210, Out-of-bag evaluation: accuracy:0.928571 logloss:0.624248
    	trees: 221, Out-of-bag evaluation: accuracy:0.928571 logloss:0.625646
    	trees: 232, Out-of-bag evaluation: accuracy:0.932773 logloss:0.626204
    	trees: 243, Out-of-bag evaluation: accuracy:0.932773 logloss:0.626587
    	trees: 254, Out-of-bag evaluation: accuracy:0.932773 logloss:0.627495
    	trees: 264, Out-of-bag evaluation: accuracy:0.932773 logloss:0.627329
    	trees: 275, Out-of-bag evaluation: accuracy:0.932773 logloss:0.628722
    	trees: 285, Out-of-bag evaluation: accuracy:0.932773 logloss:0.628634
    	trees: 295, Out-of-bag evaluation: accuracy:0.932773 logloss:0.628926
    	trees: 300, Out-of-bag evaluation: accuracy:0.932773 logloss:0.629315
    /Users/umbertofasci/miniforge3/envs/tensorflow-metal/lib/python3.9/site-packages/keras/engine/functional.py:639: UserWarning: Input dict contained keys ['island', 'bill_depth_mm', 'flipper_length_mm', 'sex', 'year'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)
    [INFO 23-05-21 17:46:16.8361 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpim1_40_4/model/ with prefix cf8c50d031f849b6
    [INFO 23-05-21 17:46:16.8431 CDT decision_forest.cc:660] Model loaded with 300 root(s), 5746 node(s), and 2 input feature(s).
    [INFO 23-05-21 17:46:16.8431 CDT kernel.cc:1074] Use fast generic engine

The following example re-implements the same logic using TensorFlow Feature Columns.

    def g_to_kg(x):
        return x / 1000
    
    feature_columns = [
        tf.feature_column.numeric_column("body_mass_g", normalizer_fn=g_to_kg),
        tf.feature_column.numeric_column("bill_length_mm"),
    ]
    
    preprocessing = tf.keras.layers.DenseFeatures(feature_columns)
    
    model_5 = tfdf.keras.RandomForestModel(preprocessing=preprocessing)
    model_5.fit(train_ds)

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpyq0muwgc as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.086347. Found 238 examples.
    
    Training model...Model trained in 0:00:00.021749
    
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:46:17.1041 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmpyq0muwgc/model/ with prefix 00ac7594b60f4164
    [INFO 23-05-21 17:46:17.1113 CDT decision_forest.cc:660] Model loaded with 300 root(s), 5746 node(s), and 2 input feature(s).
    [INFO 23-05-21 17:46:17.1114 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2d32c50d0>


<a id="orgf7e2fab"></a>

# Training a regression model

The previous example trains a classification model(TF-DF does not differentiate between binary classification and multi-class classification). In the next example, train a regression model on the Abalone dataset. The objective of this dataset is to predict the number of rings on a shell of a abalone.

**Note**: The csv file is assembled by appending UCI&rsquo;s header and data files. No preprocessing was applied.

    !wget -q https://storage.googleapis.com/download.tensorflow.org/data/abalone_raw.csv -O /tmp/abalone.csv
    
    dataset_df = pd.read_csv("/tmp/abalone.csv")
    print(dataset_df.head(3))

      Type  LongestShell  Diameter  Height  WholeWeight  ShuckedWeight   
    0    M         0.455     0.365   0.095       0.5140         0.2245  \
    1    M         0.350     0.265   0.090       0.2255         0.0995   
    2    F         0.530     0.420   0.135       0.6770         0.2565   
    
       VisceraWeight  ShellWeight  Rings  
    0         0.1010         0.15     15  
    1         0.0485         0.07      7  
    2         0.1415         0.21      9  

    # Split the dataset into a training and testing dataset.
    train_ds_pd, test_ds_pd = split_dataset(dataset_df)
    print("{} examples in training, {} examples for testing.".format(
        len(train_ds_pd), len(test_ds_pd)))
    
    # Name of the label column.
    label = "Rings"
    
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

    2904 examples in training, 1273 examples for testing.

    # Configure the model
    model_7 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
    
    # Train the model
    model_7.fit(train_ds)

    Use /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp1lldexvf as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.104565. Found 2904 examples.
    Training model...
    [INFO 23-05-21 17:46:18.8175 CDT kernel.cc:1242] Loading model from path /var/folders/cs/mqzpymhx1qx4m12w19sz9jhc0000gn/T/tmp1lldexvf/model/ with prefix c7950ef49048458b
    Model trained in 0:00:00.707304
    Compiling model...
    Model compiled.
    [INFO 23-05-21 17:46:19.1232 CDT decision_forest.cc:660] Model loaded with 300 root(s), 258480 node(s), and 8 input feature(s).
    [INFO 23-05-21 17:46:19.1232 CDT kernel.cc:1074] Use fast generic engine

    <keras.callbacks.History at 0x2c6cab9d0>

    # Evaluate the model on the test dataset
    model_7.compile(metrics=["mse"])
    evaluation = model_7.evaluate(test_ds, return_dict=True)
    
    print(evaluation)
    print()
    print(f"MSE: {evaluation['mse']}")
    print(f"RMSE: {math.sqrt(evaluation['mse'])}")

    2/2 [==============================] - 0s 16ms/step - loss: 0.0000e+00 - mse: 4.2830
    
    {'loss': 0.0, 'mse': 4.283010005950928}
    
    MSE: 4.283010005950928
    RMSE: 2.069543429346417


<a id="orgdf66e29"></a>

# Conclusion

This concludes the basic overview of TensorFlow Decision Forest utility.

