"""
Total APIs to build 

1. chat assistent 
2. llm generated explanation 
3. Prediction using a list data 
4. Generates clinical explanations using LLMs with specific variant citations and biological mechanisms
5. Provides dosing recommendations aligned with CPIC guidelines

"""




"""
API for Assistant in webpage using OpenRouter.
"""

import os
import json
import logging
from typing import Annotated, List, Dict
from requests import request

import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------- Configuration ----------
# Load API key from environment variable (safer than hardcoding)
API_KEY = "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"
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

# OpenRouter endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
model = "openrouter/free"

# Default model (can be overridden per request if needed)
DEFAULT_MODEL = "openai/gpt-4o-mini" # "arcee-ai/trinity-mini:free"  # or "openai/gpt-4o-mini"

# System prompt / persona
SYSTEM_PROMPT = "You are a PharmaGuard AI Chatbot, a helpful assistant for pharmaceutical information. do not use chars for styling. only and only provide plain text. try to give the result under 2 to 3 lines if it is not possible than use more"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://0.0.0.0",  # Replace with your actual domain
        "X-Title": "PharmaGuard AI Chatbot",
    }

# ---------- Pydantic models ----------
class Data(BaseModel):
    """Request body containing the user's prompt."""
    prompt: Annotated[str, Field(description="The prompt given by the user.")]
    # Optional: allow client to override model
    #model: Annotated[str | None, Field(description="Optional model override.")] = None

#---------Prediction Data Model-------------# 
class PredictData(BaseModel):
    input_data: Annotated[
        Dict[str, int],
        Field(description="Enter your input data in key:value format")
    ]
class AllDetails(BaseModel):
    data: Annotated[str, Field(..., description="All information what is goint to be ")]
    
# ---------- FastAPI app ----------
app = FastAPI(title="PharmaGuard AI Chatbot API")

# Enable CORS for web page access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "PharmaGuard AI Chatbot API is running."}

@app.post("/chat")
async def chat(request: Data):
    """
    Generate a response from the AI chatbot.
    Expects a JSON body with a "prompt" field.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://0.0.0.0",  # Replace with your actual domain
        "X-Title": "PharmaGuard AI Chatbot",
    }

    
    model = "openrouter/free" # Working

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt},
        ],
    }

    logger.info(f"Sending request to OpenRouter with model: {model}")

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30  # avoid hanging
        )
        response.raise_for_status()  # raise exception for 4xx/5xx
        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]
        return {
            "response": assistant_message}  

    except requests.exceptions.Timeout:
        logger.error("Request to OpenRouter timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="AI service timed out. Please try again."
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to OpenRouter failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {str(e)}"
        )
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Unexpected response format from OpenRouter: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid response from AI service."
        )
    


## /chat explanation endpoint completed 



## /llm_generated_explanation
@app.post("/llm_generated_explanation")
def llm_generated_explanation(data:AllDetails):
        llm_generation_prompt = "You are a n"

        payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt},
        ],
    }
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30  # avoid hanging
        )
        response.raise_for_status()  # raise exception for 4xx/5xx
        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]
        return {
            "response": assistant_message}  




