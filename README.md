# Pipelines_Regression

In this repository, a feature preprocessing is carry out through application of Pipelines. 
Four pipelines are defined according with type of data we are processing:

i) Numerical pipeline: replace missing values by median, and scale features;
ii) Date pipeline: transform date string into new columns indicating temporal characteristics like day of week, day of year
iii) Categorical pipeline: replace missing values by most frequent values and applies one hot enconding
iv) Text pipeline: it has two possible approaches TFIDF or Bag-of-words Vectorization.

These four pipelines are joined to composed the input data for LightGBM Regressor which is trained for predicting house prices in a real dataset.
