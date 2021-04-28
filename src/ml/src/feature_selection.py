# IN THE NAME OF ALLAH
import argparse

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

from model_types.logistic import mature_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('--maxIter', type=float)
    parser.add_argument('--regParam', type=float)
    parser.add_argument('--elasticNetParam', type=float)

    args = parser.parse_args()
    path = args.filepath
    maxIter = args.maxIter
    if maxIter is None:
        maxIter = 10
    regParam = args.regParam
    if regParam is None:
        regParam = 0.1
    elasticNetParam = args.elasticNetParam
    if elasticNetParam is None:
        elasticNetParam = 1

    spark = SparkSession \
        .builder \
        .master('local') \
        .appName('Logistic App') \
        .getOrCreate()

    # todo Delete the next line
    spark.sparkContext.setLogLevel('OFF')

    raw_data = spark.read.csv(path, header=True)

    dataset = mature_data(raw_data)

    lr = LogisticRegression(maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
    lrModel = lr.fit(dataset)

    cols = list(set(dataset.schema.names) - {'id', 'label', 'features'})
    cols.sort()
    cofs = lrModel.coefficients

    for ind, cof in enumerate(cofs):
        if cof != 0:
            print(cols[ind])
