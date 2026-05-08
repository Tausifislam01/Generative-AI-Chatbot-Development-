from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Chatbot API", validation_alias="APP_NAME")
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    debug: bool = Field(default=False, validation_alias="APP_DEBUG")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/auth.sqlite3",
        validation_alias="DATABASE_URL",
    )
    auto_create_tables: bool = Field(default=True, validation_alias="AUTO_CREATE_TABLES")

    jwt_secret_key: str = Field(default="local-dev-change-me", validation_alias="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    seed_admin_email: str = Field(default="admin@gmail.com", validation_alias="SEED_ADMIN_EMAIL")
    seed_admin_password: str = Field(default="1234", validation_alias="SEED_ADMIN_PASSWORD")
    seed_user_email: str = Field(default="test@gmail.com", validation_alias="SEED_USER_EMAIL")
    seed_user_password: str = Field(default="1234", validation_alias="SEED_USER_PASSWORD")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
