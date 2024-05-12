import typing
import io

from handlers import train, predict, preprocess as prep, upload, set_column_types as set_types
from lib import minio
from models_storage.logreg.model import LogRegression
from models_storage.linreg.model import LinearRegression

from minio import Minio
import ray
import pandas as pd
import mlflow

from fastapi import FastAPI
from starlette import status
from starlette.responses import Response


@ray.remote
def ray_wrapper_deprecated(func: typing.Callable, user_id: str, dataset_id: str, *args):
    return with_dataframe_in_first_arg(func, *args, user_id=user_id, dataset_id=dataset_id)


@ray.remote
def ray_wrapper(func: typing.Callable, *args):
    return func(*args)


def with_dataframe_in_first_arg(
        func: typing.Callable, *args,
        user_id: str, dataset_id: str, dataset_filename="dataset.csv"
):
    df = pd.read_csv(minio.get_s3_object(minio.MinioClient().get(), user_id, f'{dataset_id}.csv'))

    return func(df, *args)


ray.init(ignore_reinit_error=True)
mlflow.set_tracking_uri('http://127.0.0.1:5500')
app = FastAPI()


@app.post('/train')
async def train_handler(*, body: train.TrainRequest):
    if not body.train_model_name.isalpha():
        return Response(content=f'{body.train_model_name}.isalpha() is false', status_code=status.HTTP_400_BAD_REQUEST)
    try:
        model = eval(body.train_model_name)
    except Exception as e:
        return Response(content=f'eval {body.train_model_name} fails, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    # вернуть body.dataset_id
    res = ray_wrapper_deprecated.remote(train.train, body.user_id, '36', model,
                                        body.hyperparameters, body.metrics,
                                        body.train_params, body.target_col)

    try:
        resp = ray.get(res)
    except Exception as e:
        return Response(content=f'error during executing train, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


@app.post('/predict')
async def predict_handler(*, body: predict.PredictRequest):
    res = ray_wrapper_deprecated.remote(predict.predict, body.user_id, body.dataset_id, body.training_run_id,
                                        body.user_id)

    try:
        resp = ray.get(res)
    except Exception as e:
        return Response(content=f'error during executing predict, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


@app.post('/preprocess')
async def preprocess_handler(*, body: prep.PreprocessRequest):
    res = ray_wrapper_deprecated.remote(prep.preprocess, body.user_id, body.dataset_id, body.schema, body.user_id,
                                        body.dataset_id)

    try:
        resp = ray.get(res)
    except Exception as e:
        return Response(content=f'error during executing preprocess, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


@app.post('/upload')
async def upload_handler(*, body: upload.UploadRequest):
    res = ray_wrapper.remote(upload.upload, body)

    try:
        resp = ray.get(res)
    except Exception as e:
        return Response(content=f'error during executing upload, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


@app.post('/set_column_types')
async def set_column_types(*, body: set_types.PreprocessRequest):
    res = ray_wrapper.remote(set_types.set_column_types, body)

    try:
        resp = ray.get(res)
    except Exception as e:
        return Response(content=f'error during executing upload, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


if __name__ == '__main__':
    pass
    # print(with_dataframe_in_first_arg(train.train, LogRegression, {'max_iter': 150},
    #                                   ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr"],
    #                                   train.TrainParams(use_cv=True, cv_chunks=10, train_test_split_ratio=0.3),
    #                                   'quality', user_id='user-2', dataset_id='36'))
