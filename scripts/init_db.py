"""
Initialize Database Schema
"""
import asyncio
from sqlalchemy import create_engine
from backend.config import settings
from backend.services.metadata_db import Base
from backend.services.user_profile import Base as UserProfileBase

def init_database():
    """Initialize database tables"""
    engine = create_engine(settings.DATABASE_URL, echo=True)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    UserProfileBase.metadata.create_all(bind=engine)
    
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_database()
