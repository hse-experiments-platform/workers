import logging
import os
import typing

import fastapi
import mlflow

from dataset_operations_library.transformation import schema
from models_storage.logreg.model import LogRegression
from models_storage.linreg.model import LinearRegression

from handlers import train, predict, transform as prep, upload, set_column_types as set_types, transform
from lib import minio

import ray
import pandas as pd
from dotenv import load_dotenv

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


load_dotenv(os.environ.get("DOTENV_FILE"))
# mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
ray.init(ignore_reinit_error=True, runtime_env={"env_vars": dict(os.environ)})
app = fastapi.FastAPI()


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
    res = ray_wrapper.remote(train.train, model, body)

    try:
        resp = ray.get(res)
    except Exception as e:
        print(e)
        return Response(content=f'error during executing train, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


@app.post('/predict')
async def predict_handler(*, body: predict.PredictRequest):
    res = ray_wrapper.remote(predict.predict, body)

    try:
        resp = ray.get(res)
    except Exception as e:
        print(e)
        return Response(content=f'error during executing predict, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


@app.post('/transform')
async def transform_handler(*, body: transform.TransformRequest):
    res = ray_wrapper.remote(transform.transform, body)

    try:
        resp = ray.get(res)
    except Exception as e:
        print(e)
        return Response(content=f'error during executing upload, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


@app.post('/upload')
async def upload_handler(*, body: upload.UploadRequest):
    res = ray_wrapper.remote(upload.upload, body)

    try:
        resp = ray.get(res)
    except Exception as e:
        print(e)
        return Response(content=f'error during executing upload, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


@app.post('/set_column_types')
async def set_column_types(*, body: set_types.PreprocessRequest):
    res = ray_wrapper.remote(set_types.set_column_types, body)

    try:
        resp = ray.get(res)
    except Exception as e:
        print(e)
        return Response(content=f'error during executing upload, error={e}',
                        status_code=status.HTTP_400_BAD_REQUEST)

    return Response(content=resp, status_code=status.HTTP_200_OK)


if __name__ == '__main__':
    pass
    # print(with_dataframe_in_first_arg(train.train, LogRegression, {'max_iter': 150},
    #                                   ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr"],
    #                                   train.TrainParams(use_cv=True, cv_chunks=10, train_test_split_ratio=0.3),
    #                                   'quality', user_id='user-2', dataset_id='36'))
