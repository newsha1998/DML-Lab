# IN THE NAME OF ALLAH

import os
import argparse
from os.path import dirname
import shutil
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col, when, trim, isnan
from pyspark.sql import SparkSession

HDFS_HOME = "hdfs://localhost:8020"
HDFS_DATASETS = HDFS_HOME + "/data/"


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--train')
    parser.add_argument('-M', '--model')
    parser.add_argument('-P', '--predict')
    parser.add_argument('-O', '--output')
    args = parser.parse_args()
    return args


# todo complete this
def clean_data(df):
    return df.select([to_null(c).alias(c) for c in df.columns]).na.drop()


def to_null(col_name):
    return when(~(col(col_name).isNull() | isnan(col(col_name)) | (trim(col(col_name)) == "")), col(col_name))


def cast_data(df):
    return df.select(col('id').cast('int'),
                     *(col(c).cast("float").alias(c) for c in list(set(df.columns) - {'id'})))


def extract_features(df):
    required_features = list(set(df.columns) - {'id', 'label'})
    required_features.sort()
    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    return assembler.transform(df)


def scale_features(df):
    scalar = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=False)
    model = scalar.fit(df)
    sc_df = model.transform(df)
    sc_df = sc_df.drop('features')
    sc_df = sc_df.select(*(col(c) for c in list(set(sc_df.columns) - {'scaled_features'})),
                         col('scaled_features').alias('features'))
    return sc_df


def mature_data(df):
    df = cast_data(df)
    df = clean_data(df)
    df = extract_features(df)
    df = scale_features(df)
    return df


def train(train_dataset, model_name):
    if model_name is None:
        model_name = 'model'
    model_path = os.path.join(dirname(os.getcwd()), 'models', model_name)
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)

    spark = SparkSession \
        .builder \
        .master('local') \
        .appName('Logistic App') \
        .getOrCreate()

    # todo Delete the next line
    spark.sparkContext.setLogLevel('OFF')


    #todo change this
    train_path = HDFS_DATASETS + train_dataset
    raw_data = spark.read.csv(train_path, header=True)

    dataset = mature_data(raw_data)

    lr = LogisticRegression(maxIter=10)
    lrModel = lr.fit(dataset)

    lrModel.save(path=model_path)


def predict(test_dataset, model_name, output_path):
    if model_name is None:
        model_name = 'model'
    if output_path is None:
        output_path = os.path.join(dirname(os.getcwd()), 'predict.csv')

    model_path = os.path.join(dirname(os.getcwd()), 'models', model_name)

    spark = SparkSession \
        .builder \
        .master('local') \
        .appName('Logistic App') \
        .getOrCreate()

    # todo Delete the next line
    spark.sparkContext.setLogLevel('OFF')

    model = LogisticRegressionModel.load(path=model_path)

    #todo change this
    test_path = HDFS_DATASETS + test_dataset
    raw_data = spark.read.csv(test_path, header=True)

    dataset = mature_data(raw_data)

    prediction_df = model.transform(dataset).select(col('id'), col('prediction').cast('int'))
    prediction_df = prediction_df.toPandas()
    prediction_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    args = get_arguments()

    train_dataset = args.train
    test_dataset = args.predict
    model_name = args.model
    output_path = args.output

    if train_dataset is not None:
        train(train_dataset, model_name)

    if test_dataset is not None:
        predict(test_dataset, model_name, output_path)
