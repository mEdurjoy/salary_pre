from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

app = FastAPI()

# Mount static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
with open("salary_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            Age: int = Form(...),
            Gender: int = Form(...),
            Education_Level: int = Form(...),
            Job_Title: int = Form(...),
            Years_of_Experience: int = Form(...)):

    X = np.array([[Age, Gender, Education_Level, Job_Title, Years_of_Experience]])
    prediction = model.predict(X)[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": round(prediction, 2)})
