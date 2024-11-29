from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1/chili_pepper_trained.keras")

CLASS_NAMES = ["chilli_anthacnose", "chilli_healthy", "chilli_red"]

# Suitability mapping for each class
SUITABILITY_MAPPING = {
    "chilli_healthy": "Healthy chili peppers. These peppers are in optimal condition and meet the standards for agricultural production or processing.",
    "chilli_red": "Overripe or non-productive red chili peppers. These peppers are unsuitable for production due to their overripe state, which compromises quality for both processing and market standards.",
    "chilli_anthacnose": "Chili peppers affected by anthracnose disease. These are unsuitable due to fungal infection (anthracnose), which affects the quality and safety of agricultural products."
}

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)
    print("Raw predictions:", predictions)  # Debugging line

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions[0])

    # Get the suitability description based on the predicted class
    suitability_description = SUITABILITY_MAPPING.get(predicted_class, "No suitability information available.")

    # Print the class probabilities
    for idx, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {predictions[0][idx]:.4f}")
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'suitability': suitability_description
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
