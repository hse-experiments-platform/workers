import io
import logging
import typing

import pandas as pd
import pydantic

from dataset_operations_library.preprocessing import preprocess, schema
from lib import minio, upload

allowed_types = ['string', 'int', 'float', 'dropped', 'categorial']


class EmptiesStrategy(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    technique: str
    constant_value: typing.Optional[any] = None,
    aggregate_function: typing.Optional[str] = None


class DatasetPreprocessingSettings(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    column_type: str
    empties_settings: EmptiesStrategy


class PreprocessRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    launch_id: str
    user_id: str
    dataset_id: str
    new_dataset_id: str
    settings: typing.Dict[str, DatasetPreprocessingSettings]


def set_column_types(req: PreprocessRequest):
    filename = minio.get_s3_object_file(minio.MinioClient().get(), req.user_id, f'{req.dataset_id}.csv')
    df = pd.read_csv(filename)

    m = {}
    for column, settings in req.settings.items():
        af = None
        if settings.empties_settings.aggregate_function in schema.AggregateFunction.__members__:
            af = schema.AggregateFunction(settings.empties_settings.aggregate_function)

        m[column] = preprocess.DatasetPreprocessingSettings(
            column_type=settings.column_type,
            empties_settings=schema.EmptiesStrategy(
                technique=schema.ProcessingMode(settings.empties_settings.technique),
                constant_value=settings.empties_settings.constant_value,
                aggregate_function=af)
        )

        if m[column].empties_settings.technique == schema.ProcessingMode.FillWithConstant and \
                m[column].column_type not in ['dropped', 'string']:
            m[column].empties_settings.constant_value = float(m[column].empties_settings.constant_value)


    df_new = preprocess.preprocess(df, m)

    new_filename = f'/tmp/{req.new_dataset_id}.csv'
    df_new.to_csv(new_filename, header=True, index=False)

    upload.upload(df_new, req.user_id, req.new_dataset_id)
