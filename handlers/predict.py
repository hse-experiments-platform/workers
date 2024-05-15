import io
import json

import pandas as pd
import numpy as np
import mlflow
from pydantic import BaseModel

from lib import minio


class PredictRequest(BaseModel):
    user_id: str
    dataset_id: str
    training_run_id: str
    launch_id: str


def predict(req: PredictRequest):
    filename = minio.get_s3_object_file(minio.MinioClient().get(), req.user_id, f'{req.dataset_id}.csv')
    df = pd.read_csv(filename)

    return json.dumps(do_predict(df, req.training_run_id, req.user_id, req.dataset_id, req.launch_id))


def do_predict(df: pd.DataFrame, run_id: str, user_id: str, dataset_id: str, launch_id: str):
    mlflow.set_experiment(f"{user_id}_dataset-{dataset_id}_prediction")
    with mlflow.start_run(run_name=f"dataset_prediction_{launch_id}") as run:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        res = model.predict(df)

        csv_buffer = io.BytesIO()
        np.savetxt(csv_buffer, res, delimiter=',')
        csv_buffer.seek(0)

        minio.MinioClient().get().put_object(
            user_id,
            f"prediction_{run.info.run_id}.csv",
            data=csv_buffer,
            length=csv_buffer.getbuffer().nbytes,
            content_type='text/csv'
        )

        return {'object_name': f"prediction_{run.info.run_id}.csv", 'run_id': run.info.run_id}
