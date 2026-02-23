from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware


# =====================================================
# FASTAPI SETUP
# =====================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD MODEL
# =====================================================
MODEL_PATH = "model/final_model"

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

# =====================================================
# REQUEST FORMATS
# =====================================================
class Question(BaseModel):
    text: str


# Multiple question input
class Assessment(BaseModel):
    questions: list[str]


# =====================================================
# LABEL MAP
# =====================================================
label_map = {
    0: "Remember",
    1: "Understand",
    2: "Apply",
    3: "Analyse",
    4: "Evaluate",
    5: "Create"
}

# =====================================================
# BLOOM ACTION VERBS (Educational reasoning)
# =====================================================
bloom_verbs = {
    "Remember": ["define", "list", "name", "identify", "recall"],
    "Understand": ["explain", "describe", "summarize", "interpret"],
    "Apply": ["apply", "solve", "use", "demonstrate", "implement"],
    "Analyse": ["analyze", "compare", "differentiate", "examine"],
    "Evaluate": ["evaluate", "justify", "criticize", "assess"],
    "Create": ["design", "create", "develop", "construct", "formulate"]
}


# =====================================================
# EXPLANATION GENERATOR
# =====================================================
def generate_explanation(question_text, predicted_level):
    text = question_text.lower()
    explanations = []

    # Verb detection
    for level, verbs in bloom_verbs.items():
        for verb in verbs:
            if verb in text:
                explanations.append(
                    f"Detected cognitive verb '{verb}' associated with {level} level."
                )

    # Educational reasoning
    if predicted_level == "Remember":
        explanations.append("Question focuses on recalling factual information.")

    elif predicted_level == "Understand":
        explanations.append("Question requires conceptual explanation or understanding.")

    elif predicted_level == "Apply":
        explanations.append("Question expects use of knowledge in a practical scenario.")

    elif predicted_level == "Analyse":
        explanations.append("Question requires breaking concepts into parts or reasoning.")

    elif predicted_level == "Evaluate":
        explanations.append("Question asks for judgement or critical assessment.")

    elif predicted_level == "Create":
        explanations.append("Question demands generation of new ideas or design.")

    if not explanations:
        explanations.append("Prediction based on semantic understanding of the question.")

    return explanations


# =====================================================
# SINGLE QUESTION PREDICTION CORE
# =====================================================
def predict_single_question(question_text):

    inputs = tokenizer(
        question_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, 3)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "level": label_map[idx.item()],
            "confidence": round(prob.item(), 3)
        })

    return results


# =====================================================
# MULTI QUESTION ANALYSIS
# =====================================================
def analyze_questions(question_list):

    counts = {
        "Remember": 0,
        "Understand": 0,
        "Apply": 0,
        "Analyse": 0,
        "Evaluate": 0,
        "Create": 0
    }

    for q in question_list:
        predictions = predict_single_question(q)
        level = predictions[0]["level"]
        counts[level] += 1

    total = len(question_list)

    percentages = {
        level: round((count / total) * 100, 2)
        for level, count in counts.items()
    }

    return percentages


# =====================================================
# EDUCATIONAL INSIGHT ENGINE
# =====================================================
def generate_assessment_insight(distribution):

    low_order = distribution["Remember"] + distribution["Understand"]
    high_order = (
        distribution["Analyse"]
        + distribution["Evaluate"]
        + distribution["Create"]
    )

    if low_order > 60:
        return "Assessment is dominated by lower-order cognitive questions."

    elif high_order > 40:
        return "Assessment promotes higher-order thinking skills."

    else:
        return "Assessment shows moderate cognitive balance."


# =====================================================
# ROOT CHECK
# =====================================================
@app.get("/")
def home():
    return {"message": "DeepBloom Cognitive Analysis API Running"}


# =====================================================
# SINGLE QUESTION ENDPOINT
# =====================================================
@app.post("/predict")
def predict(data: Question):

    results = predict_single_question(data.text)

    final_level = results[0]["level"]

    explanation = generate_explanation(data.text, final_level)

    return {
        "question": data.text,
        "top_predictions": results,
        "final_prediction": final_level,
        "explanation": explanation
    }


# =====================================================
# COGNITIVE COMPLEXITY SCORE (Research Style)
# =====================================================
def calculate_complexity_score(distribution):

    # research-weighted Bloom hierarchy
    weights = {
        "Remember": 1.0,
        "Understand": 2.0,
        "Apply": 3.0,
        "Analyse": 4.5,
        "Evaluate": 5.5,
        "Create": 6.5
    }

    weighted_sum = 0

    for level, percent in distribution.items():
        weighted_sum += percent * weights[level]

    # maximum possible (if 100% Create)
    max_score = 100 * weights["Create"]

    normalized_score = (weighted_sum / max_score) * 10
    score = round(normalized_score, 2)

    # interpretation layer
    if score < 3:
        complexity_level = "Low cognitive complexity"
    elif score < 6:
        complexity_level = "Moderate cognitive complexity"
    else:
        complexity_level = "High cognitive complexity"

    return score, complexity_level


# =====================================================
# ASSESSMENT ANALYSIS ENDPOINT
# =====================================================
@app.post("/analyze-assessment")
def analyze_assessment(data: Assessment):

    distribution = analyze_questions(data.questions)

    insight = generate_assessment_insight(distribution)

    complexity_score, complexity_level = calculate_complexity_score(distribution)

    return {
        "total_questions": len(data.questions),
        "cognitive_distribution_percent": distribution,
        "complexity_score_out_of_10": complexity_score,
        "complexity_level": complexity_level,
        "insight": insight

    }
    import os

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )


