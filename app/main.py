from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Politician Classifier")
model = tf.keras.models.load_model("models/vgg16_politician_final.h5")

CLASSES = ['imran_khan', 'nawaz_sharif', 'maryam_nawaz', 'shehbaz_sharif'] # Add all 16

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    preds = model.predict(img_array)
    return {"prediction": CLASSES[np.argmax(preds)], "confidence": float(np.max(preds))}