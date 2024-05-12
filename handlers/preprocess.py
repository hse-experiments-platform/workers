import io
import typing

import pandas as pd
from minio import Minio
from pydantic import BaseModel

allowed_types = ['string', 'int', 'float', 'dropped', 'categorial']


class PreprocessRequest(BaseModel):
    launch_id: str
    user_id: str
    dataset_id: str
    schema: typing.Dict[str, typing.Dict[str, str]]


def preprocess(df: pd.DataFrame, schema: typing.List[typing.Tuple[str, str]], result_bucket: str, result_object: str):
    for (name, type) in schema:
        try:
            if type == 'int':
                df[name] = df[name].astype(int)
            elif type == 'float':
                df[name] = df[name].astype(float)
            elif type == 'string' or type == 'categorial':
                df[name] = df[name].astype(str)
            elif type == 'dropped':
                df = df.drop(name, axis=1)
            else:
                raise Exception(f"type not in {allowed_types}, got {type}")
        except ValueError as err:
            raise Exception(f"cannot convert column {name} to type {type}")

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, sep=',', index=False)
    csv_buffer.seek(0)

    Minio("localhost:9000", access_key="ROOTNAME", secret_key="CHANGEME123", secure=False).put_object(
        result_bucket,
        result_object,
        data=csv_buffer,
        length=csv_buffer.getbuffer().nbytes,
        # Указываем размер данных, чтобы MinIO мог правильно их обрабатывать
        content_type='text/csv'
    )
