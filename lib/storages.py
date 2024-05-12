import os
import typing

from minio import Minio

from lib import postgres
from lib.minio import MinioClient


def upload_to_s3(client: Minio, user_id: str, dataset_id: str, filepath: str):
    client.fput_object(user_id, f"{dataset_id}.csv", filepath)


def upload_to_postgres(filename: str, dataset_id: str, cols: typing.List[str]):
    cols_str = ', '.join([f'\\"{col}\\" text' for col in cols])
    cmd = (f'psql {postgres.get_conn_string()} ' +
           f'''-c "DROP TABLE IF EXISTS dataset_{dataset_id}" ''' +
           f'''-c "CREATE TABLE dataset_{dataset_id} ({cols_str});" '''
           f'''-c "\copy dataset_{dataset_id} FROM '{filename}' DELIMITER ',' CSV HEADER;"''')
    os.system(cmd)


def upload_to_storages(filename: str, user_id: str, dataset_id: str, cols: typing.List[str]):
    client = MinioClient().get()

    upload_to_s3(client, user_id, dataset_id, filename)

    upload_to_postgres(filename, dataset_id, cols)
