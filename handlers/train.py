import typing

import pandas as pd
import mlflow
from mlflow.models import infer_signature
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TrainParams(BaseModel):
    use_cv: bool
    cv_chunks: int
    train_test_split_ratio: float


class TrainRequest(BaseModel):
    user_id: str
    dataset_id: str
    target_col: str
    train_model_name: str
    hyperparameters: typing.Dict[str, typing.Any]
    metrics: typing.List[str]
    train_params: TrainParams


def train(df: pd.DataFrame, base_model, hyperparams: typing.Dict[str, typing.Any], metrics: typing.List[str],
          train_params: TrainParams, target: str):
    mlflow.set_tracking_uri('http://127.0.0.1:5500')
    with mlflow.start_run() as run:
        model = base_model(hyperparams=hyperparams, metrics=metrics)
        mlflow.log_params(model.get_merged_params(deep=True))


        X_old = df.drop([target], axis=1)
        X = StandardScaler().fit_transform(X_old)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_params.train_test_split_ratio,
                                                            random_state=44)

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
