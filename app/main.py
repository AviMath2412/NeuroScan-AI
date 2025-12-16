import io
import os

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.utils.preprocess import preprocess_image


def load_tf_model(model_path: str):
	try:
		from tensorflow.keras.models import load_model as tf_load_model  # type: ignore
	except Exception as err:
		print(f'❌ TensorFlow import failed: {err}')
		return None
	try:
		return tf_load_model(model_path)
	except Exception as err:
		print(f'❌ Failed to load {model_path}: {err}')
		return None

app = FastAPI(title='Brain Tumor Detection API')

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
static_dir = os.path.join(base_dir, 'static')
static_exists = os.path.isdir(static_dir)

if static_exists:
	app.mount('/static', StaticFiles(directory=static_dir), name='static')
else:
	print(f'⚠️ Static directory missing at {static_dir}. Skipping mount.')

app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_methods=['*'],
	allow_headers=['*'],
)

model_files = ['best_model.h5', 'best_model_improved.keras', 'bestmodel.h5']
model = None

for model_file in model_files:
	if os.path.exists(model_file):
		model = load_tf_model(model_file)
		if model is not None:
			print(f'✅ Loaded model: {model_file}')
			break

if model is None:
	print('⚠️ No model loaded. API will run in demo mode.')


@app.get('/')
def serve_ui():
	if static_exists:
		index_path = os.path.join(static_dir, 'index.html')
		if os.path.exists(index_path):
			return FileResponse(index_path)
	return {
		'message': 'Brain Tumor Detection API',
		'docs': '/docs',
		'health': '/health',
	}


@app.get('/health')
def health_check():
	return {'status': 'API is running'}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
	try:
		if model is None:
			return {
				'error': (
					'Model not loaded. Please download the model file and '
					'place it in the root directory.'
				),
				'instructions': (
					"Download model from releases page and place as "
					"'best_model.h5'"
				),
			}

		image_bytes = await file.read()
		img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

		input_arr = preprocess_image(img)
		predictions = model.predict(input_arr)
		pred_class = np.argmax(predictions, axis=1)[0]
		confidence = float(np.max(predictions))

		class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

		return {
			'prediction': class_names[pred_class],
			'class_index': int(pred_class),
			'confidence': confidence,
			'all_probabilities': {
				class_names[i]: float(predictions[0][i])
				for i in range(len(class_names))
			},
		}
	except Exception as err:
		return {'error': f'Failed to process image: {str(err)}'}
