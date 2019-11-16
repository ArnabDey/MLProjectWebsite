# INTRODUCTION
This project was designed to create a supervised machine learning model that would predict house prices. The problem we tried to solve is to help consumers gain basic knowledge on what features of a house will affect prices. As this is something that most people will experience at least once in their life, it is important to provide a resilient method of help.
<p align="center">
  <img width="460" height="300" src="Images/house.svg">
</p>

# DATASET AND APPROACH
Our dataset was called Victoria Real Estate and it is from Kaggle. This dataset has around 100,000 records with 15 columns such as:
- suburb
- region
- number of bedrooms
- number of bathrooms
- number of parking spaces
- price

This data is recent as the set was created around 1 year ago and each entry contains a list date from October or November 2018. The possbile features of this dataset appealed to us as it contains the information that we believe would directly affect the price of a house. We also decided to use features that could be generalized to all regions in an attempt to get to a solution for this general problem.
## Data Cleaning
We removed entries without price values as there was only around 6,000 which was 5% of our dataset.

## Detecting Outliers
After cleaning the data, converting all the categorical data into to numeric data using label encoding and removing all the invalid data, we wanted to ensure that there were no outliers in our dataset. We initially did PCA on all of the numeric features excluding our labels, the price of houses, to one dimension. The new dimension, which is the compressed version of all the features, was plotted along the price of the house.
<p align="center">
  <img width="460" height="300" src="Images/PCAofAllFeatures.png">
</p>

We next conducted PCA ignoring all the categorical data in order to ensure that the categorical data was not displaying invalid outliers.
<p align="center">
  <img width="460" height="300" src="Images/PCAOnlyNumericFeatures.png">
</p>
Since both of the plots had the same trends, we removed the four points, which are circled in red, that were far away from the large cluster on the plots.
<p align="center">
  <img width="460" height="300" src="Images/PCAofAllFeaturesRemovingOutliers.png">
</p>

## Feature Selection
<p align="center">
  <img width="100%" height="300" src="Images/sns.png">
</p>


# EXPERIMENTS
How did you evaluate your approach?
What are the results?

## Overview
We tested 5 models on the dataset, with the goal of finding the best one: Ridge Regression, Decision Tree, Random Forest, Neural Network, and K-Means Clustering (unsupervised). For each model, we used 80% of the data as training data, and 20% of the data as testing data.

After runnning each model, we calculated the RMSE and the Adjusted R^2 value. The Root Mean Square error tells us how 'off' the predicted prices are from the ground truth prices. The Adjusted R^2 value tells us how good the model's prediction is compared to a model predicting the mean value of all predictions, which serves as a benchmark for the model's accuracy (Source: https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/).

To ensure that the model's accuracy is not impacted by the train-test split, we used 10-fold cross validation on a shuffled version of the data to run our models. Hence, there are 10 RMSEs and 10 Adjusted R^2 values, of which we found the average of each.

## Ridge Regression

### Description

Ridge Regression aims to fit a function to the dataset such that the following error function is minimized:

![ridgeeq](https://latex.codecogs.com/gif.latex?E%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28f%28x_i%2C%5Ctheta%29-y_i%29%5E2%20&plus;%20%5Cfrac%7B%5Clambda%7D%7BN%7D%7C%7C%5Ctheta%7C%7C%5E2)


In this model, there is a parameter called the regularization strength, which determines how much penalty to add to the loss function. The purpose of this parameter is to prevent overfitting. We used a set of 5 possible regularization strength values, of which we needed to choose 1: [0, 0.1, 1, 5, 10, 100, 1000]. We chose this set because it was the same one used in HW3. To find the best one, we used 10-fold cross validation on the training set. Since we were using the Scikit Learn Ridge Cross Validation library, we don't know what regularization strength the model actually ended up picking.

### Overview

## Random Forest
### Process
We wanted to determine the correct hyperparameters in order to increase the Root Mean Square Error (RMSE), so we adjusted the Minimum Samples needed in order to create a leaf. In order to ensure that we could get the highest RMSE, we varied the Minimum Samples Leaf Size from 1 to 100. We determined the optimal leaf size by looking for the point on the plot where the Training RMSE continued to decrease and the Testing RMSE started to increase when we look at plot in decreasing order of leaf size. By determining the optimal leaf size, we also reduced the chance for overfitting. Overfitting occurs when the RMSE is high for the Training data but low for the Testing data in comparison to other hyperparameter value.
<p align="center">
  <img width="460" height="300" src="Images/RMSE vs Leaf Size.png">
</p>
Next, we determined the optimal max depth in order to prune our Random Forest to further decrease the chance for overfitting. Therefore, we varied the Max Depth between 1 - 100, which means that when we are at our Max Depth limit a leaf will be created instead of recursively trying to split the data further. We determined the optimal depth by finding the minimum RMSE and the corresponding Max Depth. Since the RMSE was asymptotic for both the Training and Testing datasets after the depth was greater than 20, we knew that overfitting was not occurring because of the depth.
<p align="center">
  <img width="460" height="300" src="Images/RMSE vs Max Depth.png">
</p>
Finally, we ran K-Fold cross validation with 10 folds, and we computed the RMSE, RMSE Percentage, R Squared, and time needed for execution.
<p align="center">
  <img width="460" height="300" src="Images/RMSE vs K Fold.png">
</p>
<p align="center">
  <img width="460" height="300" src="Images/R Squared vs K Fold.png">
</p>
Overall, the Random Forest was effective because the RMSE is quite low, 40757.9, the R Squared value, 0.669, is close to 1. The Random Forest also was very efficient as it took 10.5 seconds for K-Fold Validation with 10 folds.



# BEST MODEL
What is the best model?
How do you compare your method to other methods?

