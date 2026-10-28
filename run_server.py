"""Application launcher script for Procurement Invoice & Contract Auditor."""

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Procurement Invoice & Contract Auditor on http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("Interactive Auditor Cockpit & Chatbot: http://127.0.0.1:8000/")
    print("=" * 70)
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
