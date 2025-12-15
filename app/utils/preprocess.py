import numpy as np
from PIL import Image

def preprocess_image(img: Image.Image):
    return np.expand_dims(np.array(img.resize((224, 224))) / 255.0, axis=0)
