import json
import typing

import pandas as pd

from lib import storages


def get_dataframe_columns(df: pd.DataFrame) -> typing.List[str]:
    return df.columns.tolist()


def upload(df: pd.DataFrame, user_id: str, dataset_id: str):
    df.to_csv(f"/tmp/{dataset_id}.csv", index=False, header=True)

    cols = get_dataframe_columns(df)

    storages.upload_to_storages(f"/tmp/{dataset_id}.csv", user_id, dataset_id, cols)

    print("Table created successfully")

    return json.dumps(cols)
