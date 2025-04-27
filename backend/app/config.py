import os

class Config:
    # Use an environment variable for the DATABASE_URL if possible for security, or use this default:
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "postgresql://myuser:admin@localhost/towns_db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = True
    SECRET_KEY = "CHANGE_ME_TO_A_SECURE_RANDOM_KEY"
