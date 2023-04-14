# KNN-Shopping-recommender
This is e-commernce shopping recommendation using KNN recommender model
This is a Python script that includes code for building a recommender system using the K-Nearest Neighbors algorithm. It loads a dataset of electronic products and applies preprocessing steps such as dropping duplicates and filtering the dataset to include only items with an item_id less than or equal to 100. It then pivots the dataset to create a sparse matrix and fits a KNN model to this matrix.

The script also includes code for evaluating the performance of the model using cross-validation, calculating performance metrics such as mean squared error and accuracy. Finally, it sets up a Flask app with a home page and a recommender page that takes in an input item name and number of recommendations, and returns a list of recommended items based on the KNN model.

Overall, this script provides an example of how to build a simple recommender system using the KNN algorithm and Flask framework in Python.
