# Data Handling
import pickle
import numpy as np
from pydantic import BaseModel

# Server
import uvicorn
from fastapi import FastAPI

# Modeling
import lightgbm

app = FastAPI()

# Initialize files
clf = pickle.load(open('model.pickle', 'rb'))
enc = pickle.load(open('encoder.pickle', 'rb'))
features = pickle.load(open('features.pickle', 'rb'))


class Data(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: float
    average_montly_hours: float
    time_spend_company: float
    Work_accident: float
    promotion_last_5years: float
    sales: str
    salary: str
        
        
@app.post("/predict")
def predict(data: Data):
    
    # Extract data in correct order
    data_dict = data.dict()
    to_predict = [data_dict[feature] for feature in features]
     
    # Apply one-hot encoding
    encoded_features = list(enc.transform(np.array(to_predict[-2:]).reshape(1, -1))[0])
    to_predict = np.array(to_predict[:-2] + encoded_features)
    
    # Create and return prediction
    prediction = clf.predict(to_predict.reshape(1, -1))
    
    return {"prediction": int(prediction[0])}
