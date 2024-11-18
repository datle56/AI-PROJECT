from fastapi import APIRouter
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
# import lambdaTTS
from . import lambdaSpeechToScore
# import lambdaGetSample
import eng_to_ipa as ipa
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

router = APIRouter(prefix="", tags=["Pronouce"])

@router.post("/GetIPA")
async def get_accuracy_from_recorded_audio(request: TextRequest):
    text = request.text
    ipa_text = ipa.convert(text)
    ipa_text = ipa_text.replace("ˈ", "")
    return JSONResponse(content={"text": text, "ipa": ipa_text})


# @router.post("/GetIPA")
# async def get_accuracy_from_recorded_audio(request: TextRequest):
#     text = request.text
#     ipa_text = ipa.convert(text)
#     ipa_text = ipa_text.replace("ˈ", "")
#     return JSONResponse(content={"text": text, "ipa": ipa_text})