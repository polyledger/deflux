# Deflux

> Microservice for Conditional Value at Risk API

### Getting Started
Start the application with `docker-compose up --build`
To stop the application, run `docker-compose stop`

### Try it out
#### Request an allocation
This returns a `task_id` that you use to check the status of the allocation processing.
```
curl -d '{"coins": ["BTC", "ETH", "LTC"]}' -H "Content-Type: application/json" -X POST http://localhost:5000/api/allocations/
```

#### Check the status of your allocation request
Pass in `task_id` from the previous step to check the status.
```
curl -i -H "Accept: application/json" -H "Content-Type: application/json" -X GET http://localhost:5000/api/allocations/<task_id>/
```

### Testing
Get `<CONTAINER_ID` for the container by running `docker ps`
Run `docker exec -it <CONTAINER_ID> python /app/tests.py`
