from pydantic import BaseModel, EmailStr, Field
from app.models import UserRole


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length = 8, max_length = 32)
    role: UserRole = UserRole.user


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length = 8, max_length = 32)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CurrentUserResponse(BaseModel):
    id: str
    email: EmailStr
    role: UserRole
    is_active: bool