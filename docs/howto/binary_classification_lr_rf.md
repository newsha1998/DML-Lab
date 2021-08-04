# Binary Classification Algorithms 

In this document, details of the code of two algorithms (Logistic Regression and Random Forest) for binary classification is explained. Both of these algorithms are implemented using the spark ml library.

## Logistic Regression

At first, a spark session is built
```{python}
spark = SparkSession \
		.builder \
		.appName('Logistic Regression App') \
		.getOrCreate()
 ```
Then the data should be read and loaded (the format in this example is 'libsvm')
```{python}
dataset = spark.read.format('libsvm').option("numFeatures","2000").load(train_dataset_path, header=False)
 ```
 (The number of features is explicitly passed to the function to speed up the process.)
After reading the dataset, a spark ml Logistic Regression instance should be initialized and applied on the dataset. 

```{python}
lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(dataset)
lrModel.save(path=model_path)
 ```
Finally, some info about the training process can be retrieved from `trainingSummary`. Details such as `objectiveHistory` and `accuracy` are stored in this module.

In order to test the trained model, the saved model should be loaded and applied on the test dataset using `model.transform`.

## Random Forest

The procedure for dataset loading and training in Random Forest is similar to Logistic Regression except the initialization of RF module
```{python}
rf = RandomForestClassifier(numTrees=128)
 ```
 where hyperparameters such as number of trees can be set. The rest of the process for train and test is the same as logistic regression.