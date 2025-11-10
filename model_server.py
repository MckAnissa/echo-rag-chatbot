# model_server.py
import threading
import time
import os
import sys
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Make sure the path lets us import your EchoChatbot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from echo_rag import EchoChatbot
except Exception as e:
    raise ImportError(f"Couldn't import EchoChatbot: {e}")

app = FastAPI(title="Echo Model Server")

# Global server state
_server_state = {
    "chatbot": None,
    "ready": False,
    "last_error": None,
    "model_name": None,
    "loading_started_at": None,
}

class GenerateRequest(BaseModel):
    prompt: str
    use_retrieval: bool = True

@app.get("/status")
def status() -> Dict:
    """Return simple machine-readable status for the Streamlit UI to poll."""
    return {
        "ready": _server_state["ready"],
        "model_name": _server_state["model_name"],
        "last_error": str(_server_state["last_error"]) if _server_state["last_error"] else None,
        "loading_started_at": _server_state["loading_started_at"],
    }

@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate response using the chatbot. Requires server to be ready."""
    if not _server_state["ready"] or _server_state["chatbot"] is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        resp = _server_state["chatbot"].chat(req.prompt, use_retrieval=req.use_retrieval)
        return {"response": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load")
def load_model(model_name: str = "microsoft/phi-2", device: str = "cpu"):
    """
    Trigger loading of model on the server. This will return immediately while loading runs
    in background thread; poll /status to learn when done.
    """
    if _server_state["ready"]:
        return {"status": "already_ready", "model_name": _server_state["model_name"]}

    if _server_state["loading_started_at"] is not None:
        return {"status": "loading_already_in_progress", "model_name": _server_state["model_name"]}

    def _load():
        try:
            _server_state["loading_started_at"] = time.time()
            _server_state["model_name"] = model_name
            # instantiate EchoChatbot with load_model=True to force real loading
            cb = EchoChatbot(model_name=model_name, load_model=True)
            _server_state["chatbot"] = cb
            _server_state["ready"] = True
        except Exception as e:
            _server_state["last_error"] = e
            _server_state["ready"] = False

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    return {"status": "loading_started", "model_name": model_name}

@app.post("/shutdown")
def shutdown():
    # Not safe in production; just a helper for dev server
    def _stop():
        time.sleep(0.2)
        os._exit(0)
    threading.Thread(target=_stop).start()
    return {"status": "shutting_down"}
