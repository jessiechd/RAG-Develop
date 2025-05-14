from pydantic import BaseModel
from typing import Optional


class UserCreate(BaseModel):
    email: str
    password: str
    is_admin: Optional[bool] = False
    user_role: Optional[str] = 'user'   


class UserLogin(BaseModel):
    email: str
    password: str
