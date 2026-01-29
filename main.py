from fastapi import FastAPI
from contextlib import asynccontextmanager
from db.database import Base, engine

# Registers models
from models import journal

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind = engine)
    yield
    
app = FastAPI(
    title = "Innernest API",
    lifespan = lifespan
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}