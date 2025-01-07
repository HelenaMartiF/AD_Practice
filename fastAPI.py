from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd 
import requests
from transformers import pipeline

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Running correctly. Welcome."}

@app.get("/sentiment")
def sentiment_classifier(query: str):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(query)
    return {"Sentiment": result[0]['label'], "Confidence": result[0]['score']}

@app.get("/summary")
def summarize_text(query: str):
    summarizer = pipeline("summarization")
    summary = summarizer(query)
    return {"Summary": summary[0]['summary_text']}



ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
class TextRequest(BaseModel):
    text: str

def convert_to_serializable(result):
    for entity in result:
        entity['score'] = float(entity['score'])
    return result

@app.post("/ner")
def ner(request: TextRequest):
    """
    Realiza el reconocimiento de entidades nombradas (NER) sobre un texto.
    """
    result = ner_pipeline(request.text)
    result = convert_to_serializable(result) 
    return {"entities": result}


RIOT_API_KEY = "RGAPI-fb264af6-4732-4a13-a547-06a5f2c5ea6e"
RIOT_API_URL = "https://ddragon.leagueoflegends.com/cdn/12.20.1/data/en_US/champion.json"  

@app.get("/LoL/champion/{champion_name}")
def get_champion_habilities(champion_name: str):
    """
    Obtener información sobre un campeón específico, incluidas sus habilidades.
    """
    url = f"https://ddragon.leagueoflegends.com/cdn/12.20.1/data/en_US/champion/{champion_name}.json"

    response = requests.get(url)
    
    if response.status_code == 200:
        champion_data = response.json()
        abilities = champion_data.get("data", {}).get(champion_name, {}).get("spells", [])
        
        skills = []
        for ability in abilities:
            skills.append({
                "name": ability.get("name"),
                "description": ability.get("description")
            })

        return {"champion": champion_name, "skills": skills}
    else:
        return {"error": f"Champion {champion_name} not found"}
    
@app.get("/LoL/champions")
def get_champions():
    """
    Obtener lista de campeones disponibles.
    """
    response = requests.get(RIOT_API_URL)
    
    if response.status_code == 200:
        champions_data = response.json()
        champions = champions_data.get("data", {})
        champion_names = list(champions.keys())
        return {"champions": champion_names}
    else:
        return {"error": "Could not retrieve champion data"}
    
























