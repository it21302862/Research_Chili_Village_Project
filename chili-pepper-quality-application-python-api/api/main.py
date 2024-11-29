from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

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

CLASS_NAMES = ["chili_anthacnose", "chili_healthy", "chili_red"]

# Suitability mapping for each class
SUITABILITY_MAPPING = {
    "chili_healthy": "Healthy chili peppers. These peppers are in optimal condition and meet the standards for agricultural production or processing.",
    "chili_red": "Overripe or non-productive red chili peppers. These peppers are unsuitable for production due to their overripe state, which compromises quality for both processing and market standards.",
    "chili_anthacnose": "Chili peppers affected by anthracnose disease. These are unsuitable due to fungal infection (anthracnose), which affects the quality and safety of agricultural products."
}

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def get_chili_size(image: np.ndarray) -> float:
    """Estimate the size of the chili pepper in feet (mocked size logic)."""
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Convert size in pixels to approximate feet (mock conversion factor)
        size_in_feet = h * 0.01  
        return size_in_feet
    return 0.0

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

    # If the class is "chilli_healthy", analyze size
    market_recommendation = None
    if predicted_class == "chilli_healthy":
        size_in_feet = get_chili_size(image)
        print(f"Detected size: {size_in_feet:.2f} feet")

        if 3.0 <= size_in_feet <= 5.0:
            market_recommendation = "High-level market"
        else:
            market_recommendation = "Lower-level market"

    # Print the class probabilities
    for idx, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {predictions[0][idx]:.4f}")
        
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'suitability': suitability_description,
        'market_recommendation': market_recommendation if market_recommendation else "N/A",
        'detected_size': size_in_feet if predicted_class == "chilli_healthy" else "N/A"
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
