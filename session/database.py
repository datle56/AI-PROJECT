from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from pydantic import BaseModel
from typing import Optional

# Define the base class for declarative models
Base = declarative_base()

# Define the User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)

# Define the Admin model
class Admin(Base):
    __tablename__ = "admin"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
