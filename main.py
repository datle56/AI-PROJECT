from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from features.user.router import router as user_router
# from features.pronounce.router import router as pronounce_router
from features.talk.routers import chat as chat_router



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(user_router)
# app.include_router(pronounce_router)
app.include_router(chat_router)

