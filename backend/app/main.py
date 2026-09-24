from fastapi import FastAPI

app = FastAPI(title="ParkVision AI")


@app.get("/")
def root():
    return {
        "message": "ParkVision AI backend is running!"
    }