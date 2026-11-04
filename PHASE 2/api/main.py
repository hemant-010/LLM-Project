"""CLI entrypoint to run the FastAPI server."""
import os
import sys
from pathlib import Path
import uvicorn

# Ensure PHASE 2 directory is in sys.path
PHASE2_ROOT = Path(__file__).resolve().parent.parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

def run():
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() in {"1", "true", "yes"}
    
    print(f"Starting Short Interest Prediction REST API on http://{host}:{port}")
    uvicorn.run("api.app:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    run()

