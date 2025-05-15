from sqlalchemy import Column, String, Integer, DateTime, Boolean, CheckConstraint, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from uuid import uuid4
from datetime import datetime
import sys
import os
from pathlib import Path
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
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
    first_name = Column(String, nullable=True) 
    last_name = Column(String, nullable=True)  

    __table_args__ = (
        CheckConstraint("user_role IN ('sales', 'user') OR user_role IS NULL"),  
    )

    def __init__(self, email, password_hash, is_admin=False, user_role='user', first_name=None, last_name=None):
        self.id = str(uuid4())  
        self.email = email
        self.password_hash = password_hash
        self.is_admin = is_admin
        self.user_role = user_role
        self.first_name = first_name
        self.last_name = last_name


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String, nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    revoked = Column(Boolean, default=False)
