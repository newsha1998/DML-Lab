# Spark Logisitc Regression Benchmark 

**Note**: There is also a video for this tutorial [here](https://drive.google.com/file/d/1E70eKm5idY9IxjMHFT5FNVVyl9CNsr3l/view?usp=sharing) (in Persian)

### 1. Model
Spark Logistic Regression with L2 regularization, using LBFGS method.
### 2. Dataset 
[epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) dataset which contains 400K samples for training and 100K for test with 2000 features.
### 3. Hyperparameters
| Hyperparameter | Value |
| ------------- | ------------- |
| MaxIter  | 11  |
| elasticNetParam  | 0  |
| regParam  | 0.0000025  |

### 4. Notes
* Binomial logisic regression in Spark expects the input dataset to have the labels 0 and 1. (epsilon dataset has -1 and 1 so it is needed that -1's are converted to 0's)


### 5. Results
* loss during train: 
0.6931467889473747,
0.5860625251656979,
0.5672249536419827,
0.5420088120488071,
0.41967800272035366,
0.3709140683327953,
0.32893290259389457,
0.3173728800214577,
0.3126617443847263,
0.30438364727541734,
0.3000431368924191,
0.28113450702557374

* Accuracy on test: 0.88604
* Train + Test time with different configs

| num of executors | executor cores | executor memory | time(train+test) |
| ----------------- | ---------------- | ---------------- | ------------- |
| 1 | 3 | 9g | 12m 42s |
| 1 | 3 | 3g | 21m 51s |
| 3 | 1 | 3g | 15m 49s |

## How to run
Using the spark-submit command (like this [example](https://github.com/newsha1998/DML-Lab/blob/master/Documentation/SparkSubmit.sh)), we can submit a spark job on the k8s cluster. It is possible to monitor the spark job using influx and grafana (more details in the video).
