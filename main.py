import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gradcam_inference import predict_and_explain

app = FastAPI(title="NeuroScan-AI API", description="Brain Tumor Classification & Grad-CAM Explainability API")

# CORS middleware to allow requests from http://localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Root endpoint providing service status and API docs link."""
    return {
        "message": "NeuroScan-AI API is running",
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict": "POST /api/predict or /predict (multipart/form-data with key 'file')"
        }
    }

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.post("/api/predict")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (key: 'file'), saves it temporarily,
    runs predict_and_explain, returns the classification and Grad-CAM base64 result,
    and ensures the temporary file is deleted afterwards.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Determine file extension safely
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    if not suffix:
        suffix = ".jpg"

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file_path = temp_file.name

    try:
        # Save uploaded file contents
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        temp_file.close()

        # Run inference and Grad-CAM explanation
        result = predict_and_explain(temp_file_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process image.")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        # Guarantee deletion of temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as cleanup_err:
                print(f"Warning: Failed to delete temporary file {temp_file_path}: {cleanup_err}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
