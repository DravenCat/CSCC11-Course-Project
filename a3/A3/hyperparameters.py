"""
CSCC11 - Introduction to Machine Learning, Winter 2022, Assignment 3
B. Chan, E. Franco, D. Fleet

This file specifies the hyperparameters for the two real life datasets.
Note that different hyperparameters will affect the runtime of the 
algorithm.
"""

# ====================================================
# TODO: Use Validation Set to Tune hyperparameters for the Amazon dataset
# Use Optimal Parameters to get good accuracy on Test Set
AMAZON_HYPERPARAMETERS = {
    # NOTE: 10000 takes quite a while to run. Suppose if we ever want to modify the code again,
    #       we could parallelize the code such that each tree is constructed in parallel (why not right?)
    "num_trees": 700,
    "features_percent": 0.2,
    "data_percent": 0.2,
    "max_depth": 25,
    "min_leaf_data": 20,
    "min_entropy": 1e-3,
    "num_split_retries": 10
}
# ====================================================

# ====================================================
# TODO: Use Validation Set to Tune hyperparameters for the Titanic dataset
# Use Optimal Parameters to get good accuracy on Test Set
TITANIC_HYPERPARAMETERS = {
    "num_trees": 20,
    "features_percent": 0.85,
    "data_percent": 0.85,
    "max_depth": 15,
    "min_leaf_data": 10,
    "min_entropy": 1e-1,
    "num_split_retries": 10
}
# ====================================================
