import os


def get_conn_string() -> str:
    return os.environ.get('DB_CONNECT_STRING')
