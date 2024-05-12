import io

from minio import Minio
import os


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MinioClient(metaclass=MetaSingleton):
    client = None

    def get(self):
        if self.client is None:
            self.client = get_minio_client()
        return self.client

def get_minio_client() -> Minio:
    endpoint, access_key, secret_key = os.environ.get("MINIO_ENDPOINT"), os.environ.get(
        "MINIO_ACCESS_KEY"), os.environ.get("MINIO_SECRET_KEY")
    if endpoint and access_key and secret_key:
        return Minio(endpoint,
                     access_key=access_key,
                     secret_key=secret_key,
                     secure=False
                     )
    else:
        return Minio("localhost:9000",
                     access_key="ROOTNAME",
                     secret_key="CHANGEME123",
                     secure=False
                     )


def get_s3_object_file(client: Minio, bucket: str, object: str, filename: str = None) -> str:
    if filename is None:
        filename = '/tmp/' + object + '.csv'

    client.fget_object(bucket, object, filename)

    return filename


def get_s3_object(client: Minio, bucket: str, object: str):
    try:
        resp = client.get_object(bucket, object)
        data_bytes = io.BytesIO(resp.read())
        return data_bytes
    finally:
        resp.close()
        resp.release_conn()
