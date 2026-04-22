from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.agent_orchestrator import AgentOrchestrator
from app.utils.logging_utils import setup_logging

setup_logging()

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}

templates = Jinja2Templates(directory="app/templates")

orchestrator = AgentOrchestrator()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    user_message = payload.get("message", "")
    session_id = payload.get("session_id", "default")

    response = orchestrator.handle_message(session_id, user_message)
    return JSONResponse(response)
