import click
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
import xgboost as xgb
import mlflow
import mlflow.sklearn


@click.command()
@click.option("--train-data")
@click.option("--test-data")
@click.option("--max_depth", default=5)
@click.option("--n_trees", default=50)
@click.option("--learning_rate", default=0.005)
def main(train_data, test_data, max_depth, n_trees, learning_rate):
    trainDF = pd.read_csv(train_data)
    testDF = pd.read_csv(test_data)
    label_col = "price"
    yTrain = trainDF[[label_col]]
    XTrain = trainDF.drop([label_col], axis=1)
    yTest = testDF[[label_col]]
    XTest = testDF.drop([label_col], axis=1)

    xgbRegressor = xgb.XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_trees,
        learning_rate=learning_rate,
        random_state=42,
        seed=42,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=1,
        gamma=1)
    pipeline = Pipeline(steps=[("regressor", xgbRegressor)])

    pipeline.fit(XTrain, yTrain)
    yPred = pipeline.predict(XTrain)
    yPredTest = pipeline.predict(XTest)
    rmse = np.sqrt(mean_squared_error(yTrain, yPred))
    rmse_val = np.sqrt(mean_squared_error(yTest, yPredTest))
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("val_rmse", rmse_val)
    mlflow.sklearn.log_model(pipeline, "model", conda_env="conda.yml")

if __name__ == "__main__":
    main()
