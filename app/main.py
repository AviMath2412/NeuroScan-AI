from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np

from app.utils.preprocess import preprocess_image

app = FastAPI(title="Brain Tumor Detection API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("best_model.h5")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_arr = preprocess_image(img)
        predictions = model.predict(input_arr)
        pred_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        
        return {
            "prediction": class_names[pred_class],
            "class_index": int(pred_class),
            "confidence": confidence,
            "all_probabilities": {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        }
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}
