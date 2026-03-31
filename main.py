from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re

# Initialize the FastAPI app
app = FastAPI(title="Email Spam Detection API")

# Load the trained model on startup
try:
    model = joblib.load('spam_classifier.pkl')
except FileNotFoundError:
    raise RuntimeError("Model file not found. Run train_model.py first.")

# Define the hardcoded keyword filter (use lowercase)
SPAM_KEYWORDS = {
    "wire transfer",
    "enlarge your",
    "nigerian prince",
    "click here to claim",
    "100% free",
    "guaranteed return"
}

# Define the expected request body schema
class EmailPayload(BaseModel):
    text: str

def check_keywords(text: str) -> bool:
    """Returns True if any spam keyword is found in the text."""
    text_lower = text.lower()
    for keyword in SPAM_KEYWORDS:
        # Using regex to ensure we match whole words/phrases, not substrings
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            return True
    return False

@app.post("/detect")
def detect_email(payload: EmailPayload):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    # Step 1: Keyword Filter (Fast execution)
    if check_keywords(payload.text):
        return {
            "prediction": "spam",
            "confidence": 1.0,
            "method": "keyword_filter"
        }

    # Step 2: Machine Learning Prediction (If keywords pass)
    try:
        # predict() returns an array, we take the first element
        raw_prediction = model.predict([payload.text])[0]
        
        # CONVERT numpy.int64 to standard Python int
        pred_int = int(raw_prediction)
        
        # Map the dataset's integer labels back to text (0 = ham, 1 = spam)
        label_map = {0: "ham", 1: "spam"}
        final_prediction = label_map.get(pred_int, "unknown")

        # Get probabilities and convert the highest one to a standard float
        probabilities = model.predict_proba([payload.text])[0]
        max_prob = float(max(probabilities))

        return {
            "prediction": final_prediction,
            "confidence": round(max_prob, 4),
            "method": "ml_model"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))