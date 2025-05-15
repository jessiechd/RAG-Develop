from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from dotenv import load_dotenv
import os
from pathlib import Path
import secrets

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 3


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub") 
        is_admin = payload.get("is_admin")  
        user_role = payload.get("user_role")  
        return {"email": email, "is_admin": is_admin, "user_role": user_role}
    except JWTError:
        return None

def create_refresh_token():
    return secrets.token_urlsafe(32)

def get_refresh_token_expiry():
    return datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
