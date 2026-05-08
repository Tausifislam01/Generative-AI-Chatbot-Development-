from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base


engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    from app.core.security import hash_password
    from app.models.roles import UserRole
    from app.models.user import User

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        admin_email = settings.seed_admin_email.lower()
        existing_admin = await session.scalar(select(User).where(User.email == admin_email))
        if existing_admin is None:
            session.add(
                User(
                    email=admin_email,
                    hashed_password=hash_password(settings.seed_admin_password),
                    role=UserRole.ADMIN,
                )
            )

        user_email = settings.seed_user_email.lower()
        existing_user = await session.scalar(select(User).where(User.email == user_email))
        if existing_user is None:
            session.add(
                User(
                    email=user_email,
                    hashed_password=hash_password(settings.seed_user_password),
                    role=UserRole.USER,
                )
            )

        if existing_admin is None or existing_user is None:
            await session.commit()
