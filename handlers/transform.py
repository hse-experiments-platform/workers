import io
import typing

import pandas as pd
from minio import Minio
from pydantic import BaseModel
from dataset_operations_library.transformation import schema
import dataset_operations_library.transformation.transform as tf
from lib import minio, upload

allowed_types = ['string', 'int', 'float', 'dropped', 'categorial']


class TransformRequest(BaseModel):
    launch_id: str
    user_id: str
    dataset_id: str
    new_dataset_id: str
    schema: typing.Dict[str, schema.ColumnTransformSettings]


def transform(req: TransformRequest):
    filename = minio.get_s3_object_file(minio.MinioClient().get(), req.user_id, f'{req.dataset_id}.csv')
    df = pd.read_csv(filename)

    df_new = do_transform(df, req.schema)

    new_filename = f'/tmp/{req.new_dataset_id}.csv'
    df_new.to_csv(new_filename, header=True, index=False)

    upload.upload(df_new, req.user_id, req.new_dataset_id)


def do_transform(df: pd.DataFrame, schema: typing.Dict[str, schema.ColumnTransformSettings]) -> pd.DataFrame:
    return tf.transform(df, schema)
