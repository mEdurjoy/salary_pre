from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import numpy as np
import uvicorn

app = FastAPI()

# Model load
with open("salary_model.pkl", "rb") as f:
    model = pickle.load(f)

html_form = """
<h2>Salary Prediction</h2>
<form action="/predict" method="post">
Age: <input type="number" name="age"><br>
Gender (0=Female,1=Male): <input type="number" name="gender"><br>
Education Level (0-3): <input type="number" name="education"><br>
Job Title (0-5): <input type="number" name="job"><br>
Years of Experience: <input type="number" name="exp"><br>
<input type="submit" value="Predict">
</form>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return html_form

@app.post("/predict", response_class=HTMLResponse)
def predict(age: int = Form(...), gender: int = Form(...), education: int = Form(...),
            job: int = Form(...), exp: int = Form(...)):
    features = np.array([[age, gender, education, job, exp]])
    prediction = model.predict(features)[0]
    return f"<h2>Predicted Salary: {round(prediction, 2)}</h2>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
