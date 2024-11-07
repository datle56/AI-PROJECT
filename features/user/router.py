from fastapi import APIRouter, HTTPException, Form, Depends
from features.user.service import login_user, register_user
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from session.dependencies import get_db
from session.database import User, Admin
from session.auth import oauth2_scheme, Token, TokenData, create_access_token, get_current_user
from datetime import datetime, timedelta
import sys



router = APIRouter(prefix="", tags=["Auth"])



@router.post("/login")
async def register(
    username: str = Form(...), 
    password: str = Form(...), 
    db: Session = Depends(get_db)
):
    return login_user(username=username, password=password,db=db)


@router.post("/register")
async def register(
    username: str = Form(...), 
    email: str = Form(...), 
    password: str = Form(...), 
    role: str = Form(...), 
    db: Session = Depends(get_db)
):

    return register_user(username=username, email=email, password=password, role=role, db=db)