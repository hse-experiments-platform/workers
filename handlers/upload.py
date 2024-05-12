import io
import json
import os
import typing

import pandas as pd
from minio import Minio
from pydantic import BaseModel
from tqdm import tqdm
import requests
import psycopg2

from lib import storages, upload as lib_upload


class UploadRequest(BaseModel):
    launch_id: str
    user_id: str
    dataset_id: str
    url: str


def load_file_to_local_file(url: str, filename: str):
    # download file by url to filename
    response = requests.get(url, stream=True)

    with open(filename, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)


def upload_to_s3(req: UploadRequest):
    client = Minio("localhost:9000",
                   access_key="ROOTNAME",
                   secret_key="CHANGEME123",
                   secure=False
                   )
    client.fput_object(req.user_id, f"{req.dataset_id}.csv", f"/tmp/{req.dataset_id}.csv")


def upload(req: UploadRequest):
    # download dataset by link to file (not by pandas)
    load_file_to_local_file(req.url, f"/tmp/{req.dataset_id}.csv")

    df = pd.read_csv(f"/tmp/{req.dataset_id}.csv")
    # write with deduplicated columns

    return lib_upload.upload(df, req.user_id, req.dataset_id)
