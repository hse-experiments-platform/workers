import json
import os
import random
import typing

import pandas as pd
import mlflow
import sklearn.base
from mlflow.models import infer_signature
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib import minio


class TrainParams(BaseModel):
    use_cv: bool
    cv_chunks: int
    train_test_split_ratio: float


class TrainRequest(BaseModel):
    user_id: str
    dataset_id: str
    launch_id: str
    target_col: str
    train_model_name: str
    hyperparameters: typing.Dict[str, typing.Any]
    metrics: typing.List[str]
    train_params: TrainParams


def train(base_model: sklearn.base.BaseEstimator, req: TrainRequest):
    filename = minio.get_s3_object_file(minio.MinioClient().get(), req.user_id, f'{req.dataset_id}.csv')
    df = pd.read_csv(filename)

    return json.dumps(do_train(df, base_model, req.hyperparameters, req.metrics, req.train_params, req.target_col, req.user_id,
             req.dataset_id, req.launch_id))


def do_train(df: pd.DataFrame, base_model, hyperparams: typing.Dict[str, typing.Any], metrics: typing.List[str],
             train_params: TrainParams, target: str, user_id: str, dataset_id: str, launch_id: str):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(f"{user_id}_dataset-{dataset_id}_training")
    with mlflow.start_run(run_name=f"dataset_training_{launch_id}") as run:
        random.seed = 0
        seed = random.randint(0, 1000)
        random.seed = seed
        mlflow.log_param("random_seed", seed)
        model = base_model(hyperparams=hyperparams, metrics=metrics)
        mlflow.log_params(model.get_merged_params(deep=True))

        X_old = df.drop([target], axis=1)
        X = StandardScaler().fit_transform(X_old)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_params.train_test_split_ratio)

        model.fit(X_train, y_train)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_old.sample(n=10),
            registered_model_name=f'{model.__class__.__name__}_{run.info.run_id}',
        )

        test_scores = model.scores(X_test, y_test)
        mlflow.log_metrics(test_scores, step=0)

        if train_params.use_cv and len(metrics) > 0:
            all_scores = model.cv_scores(X, y, cv=train_params.cv_chunks)
            print(all_scores)
            for i in range(len(all_scores[f'test_{metrics[0]}'])):
                mlflow.log_metrics({m: all_scores[f'test_{m}'][i] for m in metrics}, step=i + 1)

        return run.info.run_id
