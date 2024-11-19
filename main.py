from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from features.user.router import router as user_router
from features.talk.routers.chat import router as chat_router
<<<<<<< HEAD
from features.grammar.router import router as grammer_router

=======
>>>>>>> 9e4522f0121440a16d4101b7a8c6ae042565f66a

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(user_router, prefix="/user", tags=["user"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
<<<<<<< HEAD
app.include_router(chat_router, prefix="/", tags=["grammer_router"])

=======
>>>>>>> 9e4522f0121440a16d4101b7a8c6ae042565f66a

