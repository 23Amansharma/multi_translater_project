from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import openai
import os
from typing import Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Multilanguage Translator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LANGUAGE_MAP = {
    "auto": "Auto Detect",
    "en": "English",
    "hi": "Hindi - हिंदी",
    "mr": "Marathi - मराठी", 
    "ta": "Tamil - தமிழ்",
    "fr": "French - Français",
    "de": "German - Deutsch",
    "ja": "Japanese - 日本語"
}

LANGUAGE_PATTERNS = {
    'hi': r'[\u0900-\u097F]',
    'mr': r'[\u0900-\u097F]',
    'ta': r'[\u0B80-\u0BFF]',
    'ja': r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]',
    'fr': r'[àâäéèêëîïôöùûüÿç]',
    'de': r'[äöüßÄÖÜ]',
}

class TranslationRequest(BaseModel):
    text: str = Field(..., max_length=5000, description="Text to translate")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")

class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    original_text: str

openai.api_key = os.getenv("OPENAI_API_KEY")

def detect_language_by_script(text: str) -> str:
    text_sample = text[:200]
    
    for lang, pattern in LANGUAGE_PATTERNS.items():
        if re.search(pattern, text_sample):
            return lang
    
    if re.search(r'[a-zA-Z]', text_sample):
        return 'en'
    
    return 'en'

async def detect_language_with_ai(text: str) -> str:
    try:
        script_detected = detect_language_by_script(text)
        
        if script_detected in ['hi', 'mr'] or script_detected == 'en':
            detection_prompt = f"""Identify the language of this text. Respond with only the language code.

Text: "{text[:500]}"

Language codes:
- en: English
- hi: Hindi (हिंदी)
- mr: Marathi (मराठी)
- ta: Tamil (தமிழ்)
- fr: French
- de: German  
- ja: Japanese

Respond with only the 2-letter code:"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a language detection expert. Identify the language and respond with only the 2-letter language code."
                    },
                    {
                        "role": "user",
                        "content": detection_prompt
                    }
                ],
                max_tokens=5,
                temperature=0.1
            )
            
            detected_lang = response.choices[0].message.content.strip().lower()
            
            if detected_lang in LANGUAGE_MAP:
                return detected_lang
        
        return script_detected
        
    except Exception as e:
        logger.error(f"AI Language detection error: {str(e)}")
        return detect_language_by_script(text)

def get_enhanced_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    source_name = LANGUAGE_MAP.get(source_lang, source_lang)
    target_name = LANGUAGE_MAP.get(target_lang, target_lang)
    
    context_examples = {
        'hi': "Examples: नमस्कार = Hello, धन्यवाद = Thank you",
        'mr': "Examples: नमस्कार = Hello, धन्यवाद = Thank you", 
        'ta': "Examples: வணக்கம் = Hello, நன்றி = Thank you",
        'fr': "Examples: Bonjour = Hello, Merci = Thank you",
        'de': "Examples: Hallo = Hello, Danke = Thank you",
        'ja': "Examples: こんにちは = Hello, ありがとう = Thank you"
    }
    
    target_context = context_examples.get(target_lang, "")
    
    if source_lang == "auto":
        prompt = f"""You are a professional translator. Translate the following text to {target_name}.

Text to translate: "{text}"

Translation Guidelines:
- Provide accurate, natural translation
- Maintain original meaning and tone
- Keep cultural context appropriate
- Preserve formatting and punctuation
- For names and proper nouns, keep original form
- If text is already in target language, return as-is
{target_context}

Provide only the translated text, no explanations:"""
    else:
        prompt = f"""You are a professional translator. Translate from {source_name} to {target_name}.

Source text ({source_name}): "{text}"

Translation Guidelines:
- Provide accurate, natural translation to {target_name}
- Maintain original meaning and tone  
- Keep cultural context appropriate
- Preserve formatting and punctuation
- For names and proper nouns, keep original form
{target_context}

Provide only the translated text, no explanations:"""
    
    return prompt

async def translate_with_enhanced_ai(text: str, source_lang: str, target_lang: str) -> tuple:
    try:
        detected_source = source_lang
        if source_lang == "auto":
            detected_source = await detect_language_with_ai(text)
            logger.info(f"Detected language: {detected_source}")
            
            if detected_source == target_lang:
                return text.strip(), detected_source
        
        if len(text.strip()) < 2:
            return text, detected_source
            
        prompt = get_enhanced_translation_prompt(text, detected_source, target_lang)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert multilingual translator specializing in Hindi, Marathi, Tamil, English, French, German, and Japanese. Provide accurate, natural translations that preserve meaning and cultural context."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=min(2000, len(text) * 3),
            temperature=0.2,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        translated_text = response.choices[0].message.content.strip()
        translated_text = clean_translation_output(translated_text, text)
        
        logger.info(f"Translation successful: {detected_source} -> {target_lang}")
        return translated_text, detected_source
        
    except openai.error.RateLimitError:
        logger.error("OpenAI Rate limit exceeded")
        raise HTTPException(status_code=429, detail="Translation service temporarily unavailable. Please try again in a moment.")
    except openai.error.InvalidRequestError as e:
        logger.error(f"OpenAI Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid translation request")
    except openai.error.AuthenticationError:
        logger.error("OpenAI Authentication failed")
        raise HTTPException(status_code=500, detail="Translation service configuration error")
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def clean_translation_output(translation: str, original: str) -> str:
    cleaned = translation.strip()
    
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1]
    
    prefixes_to_remove = [
        "Translation: ",
        "Translated text: ", 
        "The translation is: ",
        "Here is the translation: "
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    
    if not cleaned or cleaned.lower() == original.lower():
        return original
    
    return cleaned

@app.get("/")
async def root():
    return {
        "message": "AI Multilanguage Translator API", 
        "status": "active",
        "supported_languages": list(LANGUAGE_MAP.keys()),
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    api_status = "connected" if openai.api_key else "not_configured"
    return {
        "status": "healthy", 
        "version": "2.0.0",
        "openai_status": api_status,
        "supported_languages": len(LANGUAGE_MAP)
    }

@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        clean_text = request.text.strip()
        if len(clean_text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        if request.source_language == request.target_language and request.source_language != "auto":
            raise HTTPException(status_code=400, detail="Source and target languages cannot be the same")
        
        if request.source_language not in LANGUAGE_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid source language: {request.source_language}")
        
        if request.target_language not in LANGUAGE_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid target language: {request.target_language}")
        
        if not openai.api_key:
            raise HTTPException(status_code=500, detail="Translation service not configured - missing API key")
        
        logger.info(f"Translation request: '{clean_text[:50]}...' ({request.source_language} -> {request.target_language})")
        
        translated_text, detected_source = await translate_with_enhanced_ai(
            clean_text, 
            request.source_language, 
            request.target_language
        )
        
        if not translated_text:
            translated_text = clean_text
        
        response = TranslationResponse(
            translated_text=translated_text,
            source_language=detected_source,
            target_language=request.target_language,
            original_text=clean_text
        )
        
        logger.info(f"Translation completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected translation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during translation")

@app.get("/api/languages")
async def get_supported_languages():
    return {
        "languages": [
            {"code": code, "name": name, "native": name} 
            for code, name in LANGUAGE_MAP.items()
        ],
        "total": len(LANGUAGE_MAP),
        "features": ["auto_detect", "real_time_translation", "script_recognition"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")