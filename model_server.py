from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from keybert import KeyBERT
import time

# Load Models (ONLY THESE - no new models)
sentiment_model = pipeline("sentiment-analysis",
                          model="nlptown/bert-base-multilingual-uncased-sentiment")

hazard_model = pipeline("zero-shot-classification",
                       model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

app = FastAPI()

class TextIn(BaseModel):
    text: str
    top_k: int = 5

@app.post("/analyze")
def analyze(payload: TextIn):
    text = payload.text.strip()
    
    if not text or len(text) < 5:
        return {"error": "Text too short for meaningful analysis"}
    
    try:
        # 1. RELEVANCE CLASSIFICATION (with small delay)
        relevance_labels = ["disaster emergency hazard incident", "irrelevant personal chat advertisement"]
        relevance_result = hazard_model(text, relevance_labels, multi_label=False)
        time.sleep(0.2)  # Small delay between classifications
        
        # 2. URGENCY DETECTION (with small delay)
        urgency_labels = ["urgent immediate rescue emergency", "high priority important", "routine information", "informational general"]
        urgency_result = hazard_model(text, urgency_labels, multi_label=False)
        time.sleep(0.2)  # Small delay between classifications
        
        # 3. HAZARD CLASSIFICATION
        hazard_labels = ["flood", "cyclone", "earthquake", "tsunami",
                        "oil spill", "landslide", "drought", "fire"]
        hazard_result = hazard_model(text, hazard_labels, multi_label=False)

        # 4. SENTIMENT ANALYSIS
        sentiment = sentiment_model(text[:512])
        
        # 5. KEYWORD EXTRACTION
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            top_n=payload.top_k
        )
        
        # 6. ENTITY RECOGNITION
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        return {
            "relevance_classification": {
                "label": relevance_result["labels"][0],
                "score": round(relevance_result["scores"][0], 4),
                "is_relevant": "irrelevant" not in relevance_result["labels"][0].lower()
            },
            "urgency_detection": {
                "label": urgency_result["labels"][0],
                "score": round(urgency_result["scores"][0], 4),
                "is_urgent": any(word in urgency_result["labels"][0].lower() 
                               for word in ["urgent", "emergency", "critical"])
            },
            "hazard_classification": {
                "label": hazard_result["labels"][0],
                "score": round(hazard_result["scores"][0], 4)
            },
            "sentiment": [{
                "label": sentiment[0]["label"],
                "score": round(sentiment[0]["score"], 4)
            }],
            "keywords": [(kw[0], round(kw[1], 4)) for kw in keywords],
            "entities": entities
        }
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

# Test endpoint
@app.get("/test")
def test_models():
    test_text = "Earthquake in Ahmedabad, buildings collapsed, need immediate rescue"
    
    # Test all classifications sequentially
    relevance_result = hazard_model(test_text, ["disaster emergency", "irrelevant"])
    time.sleep(0.2)
    urgency_result = hazard_model(test_text, ["urgent", "high priority", "routine", "informational"])
    time.sleep(0.2)
    hazard_result = hazard_model(test_text, ["earthquake", "flood", "cyclone"])
    
    return {
        "relevance_test": relevance_result,
        "urgency_test": urgency_result,
        "hazard_test": hazard_result
    }