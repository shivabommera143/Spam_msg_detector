from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import uvicorn

# Paths
DATASET_PATH = Path("D:\shiva-py\spam-msd-detector\spam.csv")
MODEL_PATH = Path("D:\shiva-py\spam-msd-detector\spam_pipeline.joblib")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def train_and_save_model():
    print("Training model...")
    df = pd.read_csv(DATASET_PATH, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    print("Model trained and saved as spam_pipeline.joblib")

def load_model():
    if MODEL_PATH.exists():
        print("Model already exists. Skipping training.")
    else:
        train_and_save_model()
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, message: str = Form(...)):
    prediction = model.predict([message])[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
