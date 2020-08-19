# IN THE NAME OF ALLAH

import os
import argparse
from os.path import dirname
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col
from pyspark.sql import SparkSession


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--train')
    parser.add_argument('-M', '--model')
    parser.add_argument('-P', '--predict')
    parser.add_argument('-O', '--output')
    args = parser.parse_args()
    return args


def train(train_path, model_name):
    if model_name is None:
        model_name = 'model'
    model_path = os.path.join(dirname(os.getcwd()), 'models', model_name)
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)

    spark = SparkSession \
        .builder \
        .master('local') \
        .appName('Titanic Data') \
        .getOrCreate()

    # todo Delete the next line
    spark.sparkContext.setLogLevel('OFF')

    for i in range(5):
        print()
    df = spark.read.csv(train_path, header=True)

    dataset = df.select(col('label').cast('float'),
                        col('Embarked_Code').cast('float'),
                        col('Sex_Code').cast('float'),
                        col('Pclass').cast('float'),
                        col('Title_Code').cast('float'),
                        col('FamilySize').cast('float'),
                        col('AgeBin_Code').cast('float'),
                        col('FareBin_Code').cast('float')
                        )
    dataset.show()

    print(df.dtypes)

    required_features = ['Embarked_Code',
                         'Sex_Code',
                         'Pclass',
                         'Title_Code',
                         'FamilySize',
                         'AgeBin_Code',
                         'FareBin_Code',
                         ]

    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    transformed_data = assembler.transform(dataset)

    print(transformed_data.show(5))

    lr = LogisticRegression(maxIter=10)
    lrModel = lr.fit(transformed_data)

    lrModel.save(path=model_path)


def predict(test_path, model_name, output_path):
    if model_name is None:
        model_name = 'model'
    if output_path is None:
        output_path = os.path.join(dirname(os.getcwd()), 'predict.csv')

    model_path = os.path.join(dirname(os.getcwd()), 'models', model_name)

    spark = SparkSession \
        .builder \
        .master('local') \
        .appName('Titanic Data') \
        .getOrCreate()

    # todo Delete the next line
    spark.sparkContext.setLogLevel('OFF')

    for i in range(5):
        print()

    model = LogisticRegressionModel.load(path=model_path)

    df = spark.read.csv(test_path, header=True)

    dataset = df.select(col('Embarked_Code').cast('float'),
                        col('Sex_Code').cast('float'),
                        col('Pclass').cast('float'),
                        col('Title_Code').cast('float'),
                        col('FamilySize').cast('float'),
                        col('AgeBin_Code').cast('float'),
                        col('FareBin_Code').cast('float')
                        )
    dataset.show()

    print(df.dtypes)

    required_features = ['Embarked_Code',
                         'Sex_Code',
                         'Pclass',
                         'Title_Code',
                         'FamilySize',
                         'AgeBin_Code',
                         'FareBin_Code',
                         ]

    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    transformed_data = assembler.transform(dataset)

    pred = model.transform(transformed_data).select('prediction')
    pred = pred.toPandas()
    pred.to_csv(output_path)


if __name__ == '__main__':
    args = get_arguments()

    train_path = args.train
    test_path = args.predict
    model_name = args.model
    output_path = args.output

    if train_path is not None:
        train(train_path, model_name)

    if test_path is not None:
        predict(test_path, model_name, output_path)
