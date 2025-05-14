from sqlalchemy import Column, String, Integer, DateTime, Boolean, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from uuid import uuid4
from datetime import datetime
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db 

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_admin = Column(Boolean, default=False)  
    user_role = Column(String, nullable=True, default='user')  

    __table_args__ = (
        CheckConstraint("user_role IN ('sales', 'user') OR user_role IS NULL"),  
    )

    def __init__(self, email, password_hash, is_admin=False, user_role='user'):
        self.id = str(uuid4())  
        self.email = email
        self.password_hash = password_hash
        self.is_admin = is_admin
        self.user_role = user_role
