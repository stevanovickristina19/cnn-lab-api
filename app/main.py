from fastapi import FastAPI
from app.routers import model_router

app = FastAPI(
    title="Build your own CNN",
    description="API for building and evaluating your own CNN model",
    version="1.0.0"
)

app.include_router(model_router.router)

@app.get("/")
def root():
    return {"message": "Start building  your own CNN"}

# 👇 Add this startup event
@app.on_event("startup")
async def startup_event():
    print("\n Server is running!")
    print(" Build your CNN here: http://127.0.0.1:8000/docs")
    print(" Root endpoint: http://127.0.0.1:8000/\n")