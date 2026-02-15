# Celery worker entrypoint
from tasks import celery_app

if __name__ == '__main__':
    # Start Celery worker (solo pool is simplest for local/dev)
    celery_app.worker_main(['worker', '--loglevel=info', '--pool=solo'])