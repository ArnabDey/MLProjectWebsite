# INTRODUCTION
This project is designed to create a supervised machine learning model that will predict house prices. The problem we are trying to solve is to help consumers gain basic knowledge on what features of a house will affect prices.
![Hello](https://drive.google.com/file/d/1AAQUn2chpHqa_6yMqFIMtfKpha0uu0vG/view?usp=sharing)

# DATASET AND APPROACH
How did you get your dataset?
What are its characteristics (e.g. number of features, # of records, temporal or not, etc.)
Why do you think your approach can effectively solve your problem?
What is new in your approach?

## Detecting Outliers
After cleaning the data, converting all the categorical data into to numeric data using label encoding and removing all the invalid data, we wanted to ensure that there were no outliers in our dataset. We initially did PCA on all of the numeric features excluding our labels, the price of houses, to one dimension. The new dimension, which is the compressed version of all the features, was plotted along the price of the house.
<p align="center">
  <img width="460" height="300" src="Images/PCA of All Features">
</p>

We next conducted PCA ignoring all the categorical data in order to ensure that the categorical data was not displaying invalid outliers.
<p align="center">
  <img width="460" height="300" src="Images/PCA Only Numeric Features">
</p>
Since both of the plots had the same trends, we removed the four points, which are circled in red, that were far away from the large cluster on the plots.
<p align="center">
  <img width="460" height="300" src="Images/PCA of All Features Removing Outliers">
</p>



# EXPERIMENTS
How did you evaluate your approach?
What are the results?

# BEST MODEL
What is the best model?
How do you compare your method to other methods?

