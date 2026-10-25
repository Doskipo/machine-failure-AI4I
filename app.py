"""FastAPI service for the AI4I machine-failure classifier."""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

# Loaded once at import time, not per request: unpickling a model costs milliseconds we don't want on every call.
model = joblib.load("model.joblib")

app = FastAPI(title="AI4I failure predictor")


class Reading(BaseModel):
    """One sensor reading from a machine. FastAPI validates the request body against this class."""

    # Physically plausible bounds, wider than the training ranges: they reject garbage and unit
    # mistakes (e.g. Celsius instead of Kelvin), not unusual-but-real readings.
    machine_type: Literal["L", "M", "H"]
    air_temperature: float = Field(ge=273, le=320)      # K; training range 295.3-304.5
    process_temperature: float = Field(ge=283, le=330)  # K; ~10 K above air; training 305.7-313.8
    rotational_speed: float = Field(ge=500, le=4000)    # rpm; a stopped machine is out of scope; training 1168-2886
    torque: float = Field(ge=0, le=100)                 # Nm; training 3.8-76.6
    tool_wear: float = Field(ge=0, le=500)              # min; training 0-253


@app.get("/health")
def health():
    # The model loads at import, so if the app is answering at all, the model is loaded.
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: Reading):
    # One-row DataFrame: the pipeline selects columns *by name*, so they must match what train.py used.
    df = pd.DataFrame({"Type": [reading.machine_type],
                        "Air temperature": [reading.air_temperature] , 
                        "Process temperature" : [reading.process_temperature] ,
                        "Rotational speed" : [reading.rotational_speed],
                        "Torque" : [reading.torque],
                        "Tool wear" : [reading.tool_wear]})

    # predict_proba returns [[P(no failure), P(failure)]]: row 0, column 1 is the failure probability.
    # float() turns numpy.float64 into a plain Python float so it serializes to JSON cleanly.
    failure_probability = float(model.predict_proba(df)[0][1])

    # 0.5 is the same threshold model.predict() uses, so this matches the recall measured in train.py.
    return {"failure_probability": failure_probability, "failure": failure_probability >= 0.5}
