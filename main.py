from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load


# load model
clf = load('lr.joblib')

# Define the request body model
class WineFeatures(BaseModel):
    alcohol: float
    volatile_acidity: float

# Make a prediction using a trained model on the user-entered data
def get_prediction(alcohol, volatile_acidity):
    
    # Define your features Matrix
    X = [[alcohol, volatile_acidity]]

    # Get the prediction and the predict probability
    y_pred = clf.predict(X)[0]  
    y_proba = clf.predict_proba(X)[0].tolist()  

    return {'prediction': int(y_pred), 'probability': y_proba}


# initiate API
app = FastAPI()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello stranger! This API allow you to evaluate the quality of red wine. Go to the /docs for more details."}


# define the predict endpoint
@app.post("/predict")
async def predict(wine: WineFeatures):
    pred = get_prediction(wine.alcohol, wine.volatile_acidity)
    return pred