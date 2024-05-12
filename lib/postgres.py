import os


def get_conn_string() -> str:
    if os.environ.get('DATABASE_URL'):
        return os.environ.get('DATABASE_URL')
    else:
        return 'postgres://hseuser:P%40ssw0rd@localhost:6433/datasetsdb'
