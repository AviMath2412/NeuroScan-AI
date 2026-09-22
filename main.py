import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gradcam_inference import predict_and_explain

app = FastAPI(title="NeuroScan-AI API", description="Brain Tumor Classification & Grad-CAM Explainability API")

# CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "NeuroScan-AI API is running", "docs_url": "/docs", "health_check": "/health"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/predict")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file_path = temp_file.name
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        temp_file.close()
        
        result = predict_and_explain(temp_file_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process image.")
            
        if isinstance(result, dict):
            predicted_class = result.get("predicted_class", "")
            confidence = result.get("confidence", 0.0)
            raw_b64 = result.get("heatmap_base64", "")
            all_probs = result.get("all_probabilities", {})

            b64_str = raw_b64 if raw_b64.startswith("data:") else f"data:image/jpeg;base64,{raw_b64}"

            return {
                "predicted_class": predicted_class,
                "prediction": predicted_class.capitalize(),
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "all_probabilities": all_probs,
                "heatmap_base64": b64_str
            }
        else:
            predicted_class, confidence, heatmap_img = result
            import base64
            from io import BytesIO
            buffered = BytesIO()
            heatmap_img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {
                "predicted_class": predicted_class,
                "prediction": predicted_class.capitalize(),
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "heatmap_base64": f"data:image/jpeg;base64,{img_base64}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)