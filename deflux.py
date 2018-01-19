# Application written with Flask
# http://flask.pocoo.org/docs/0.12/

from flask import Flask, request, jsonify
from redis import Redis, RedisError
from celery import Celery
import os
import sys

# Connect to Redis
redis = Redis(host='redis', db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://redis:6379'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Celery task for running allocation
# http://docs.celeryproject.org/en/latest/
@celery.task()
def allocate():
    return []

@app.route('/api/allocations/<task_id>/', methods=['GET'])
def get_allocation(task_id):
    task = allocate.AsyncResult(task_id)
    response = {'state': task.state}
    return jsonify(response)

@app.route('/api/allocations/', methods=['POST'])
def create_allocation():
    # Initiate a new allocation job
    result = allocate.delay()

    # Return the job ID (this will be through the job queue processor)
    response = {'task_id': result.id}
    return jsonify(response)

if __name__ == '__main__':
    app.run()