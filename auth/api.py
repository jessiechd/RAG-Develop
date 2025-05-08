import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from database import get_db  
from auth.utils import create_access_token, verify_token, get_password_hash, verify_password
from auth.schemas import UserCreate, UserLogin
from auth.models import User
from auth.dependencies import get_current_user 


router = APIRouter()


@router.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password) 
    new_user = User(email=user.email, password_hash=hashed_password)
    new_user.role = "user"

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User created successfully"}

@router.post("/login")
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user is None or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    access_token = create_access_token(data={
        "sub": db_user.email,
        "role": db_user.role  
    }) 
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/admin-only")
def only_admin(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    return {"message": f"Welcome, admin {current_user['email']}!"}
