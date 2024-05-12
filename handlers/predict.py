import io

from minio import Minio
import pandas as pd
import numpy as np
import mlflow
from pydantic import BaseModel


class PredictRequest(BaseModel):
    user_id: str
    dataset_id: str
    training_run_id: str


def predict(df: pd.DataFrame, run_id: str, result_bucket):
    with mlflow.start_run() as run:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        res = model.predict(df)
        print(res)

        csv_buffer = io.BytesIO()
        np.savetxt(csv_buffer, res, delimiter=',')
        csv_buffer.seek(0)

        Minio("localhost:9000", access_key="ROOTNAME", secret_key="CHANGEME123", secure=False).put_object(
            result_bucket,
            f"prediction_{run.info.run_id}.csv",
            data=csv_buffer,
            length=csv_buffer.getbuffer().nbytes,
            # Указываем размер данных, чтобы MinIO мог правильно их обрабатывать
            content_type='text/csv'
        )

        return run.info.run_id
