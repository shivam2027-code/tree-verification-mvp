from fastapi import FastAPI
from backend.routers import register, verify
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Tree Identity Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(register.router, prefix="/ai")
app.include_router(verify.router, prefix="/ai")

@app.get("/")
def root():
    return {"status": "ok"}
