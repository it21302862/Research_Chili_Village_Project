from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
from pymongo import MongoClient
import base64
import hashlib

app = FastAPI()

# CORS Configuration
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

# Load TensorFlow Model
MODEL = tf.keras.models.load_model("../saved_models/1/chili_pepper_trained.keras")

CLASS_NAMES = ["chili_anthacnose", "chili_healthy", "chili_red"]

# Suitability mapping for each class
SUITABILITY_MAPPING = {
    "chili_healthy": "Healthy chili peppers. These peppers are in optimal condition and meet the standards for agricultural production or processing.",
    "chili_red": "Overripe or non-productive red chili peppers. These peppers are unsuitable for production due to their overripe state, which compromises quality for both processing and market standards.",
    "chili_anthacnose": "Chili peppers affected by anthracnose disease. These are unsuitable due to fungal infection (anthracnose), which affects the quality and safety of agricultural products."
}

# MongoDB Connection
client = MongoClient("mongodb+srv://itpprojectdb:Pass123itp@itp.jet5d6f.mongodb.net/")
db = client["chili_pepper_database"]
collection = db["chili_pepper_production"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    """Convert file bytes to a numpy array image."""
    image = np.array(Image.open(BytesIO(data)))
    return image

def get_chili_size(image: np.ndarray) -> float:
    """Estimate the size of the chili pepper in feet."""
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to extract contours
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        _, _, _, h = cv2.boundingRect(largest_contour)

        # Convert height in pixels to feet (mock conversion factor)
        size_in_feet = h * 0.01  
        return size_in_feet
    return 0.0

def calculate_image_hash(image_data: bytes) -> str:
    """Generate a unique hash for the given image data."""
    return hashlib.md5(image_data).hexdigest()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = read_file_as_image(image_data)
    img_batch = np.expand_dims(image, axis=0)

    # Generate image hash
    image_hash = calculate_image_hash(image_data)

    # Check if image already exists in the database
    existing_record = collection.find_one({"image_hash": image_hash})
    if existing_record:
        return {
            "message": "Duplicate image detected. This image has already been processed.",
            "record": {
                "class": existing_record['class'],
                "confidence": existing_record['confidence'],
                "suitability": existing_record['suitability'],
                "market_recommendation": existing_record['market_recommendation'],
                "detected_size": existing_record['detected_size']
            }
        }

    # Make predictions using the model
    predictions = MODEL.predict(img_batch)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions[0])

    # Get suitability description based on the predicted class
    suitability_description = SUITABILITY_MAPPING.get(predicted_class, "No suitability information available.")

    # If the class is "chili_healthy", analyze size
    market_recommendation = None
    size_in_feet = None
    if predicted_class == "chili_healthy":
        size_in_feet = get_chili_size(image)
        print(f"Detected size: {size_in_feet:.2f} feet")

        if 3.0 <= size_in_feet <= 5.0:
            market_recommendation = "High-level market"
        else:
            market_recommendation = "Lower-level market"

    # Save data and image in MongoDB
    image_base64 = base64.b64encode(image_data).decode('utf-8')  # Encode image as base64
    record = {
        'image_hash': image_hash,  # Unique hash for the image
        'class': predicted_class,
        'confidence': float(confidence),
        'suitability': suitability_description,
        'market_recommendation': market_recommendation if market_recommendation else "N/A",
        'detected_size': size_in_feet if size_in_feet else "N/A",
        'image': image_base64
    }
    collection.insert_one(record)

    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'suitability': suitability_description,
        'market_recommendation': market_recommendation if market_recommendation else "N/A",
        'detected_size': size_in_feet if size_in_feet else "N/A"
    }

@app.get("/report")
async def get_report():
    # Dynamically calculate the total count for each class
    chili_healthy_count = collection.count_documents({'class': 'chili_healthy'})
    chili_red_count = collection.count_documents({'class': 'chili_red'})
    chili_anthacnose_count = collection.count_documents({'class': 'chili_anthacnose'})

    # Get the market recommendations for each class
    healthy_market_recommendations = {
        "High-level market": collection.count_documents({'class': 'chili_healthy', 'market_recommendation': 'High-level market'}),
        "Lower-level market": collection.count_documents({'class': 'chili_healthy', 'market_recommendation': 'Lower-level market'})
    }
    red_market_recommendations = {
        "Not suitable for market": chili_red_count  
    }
    anthacnose_market_recommendations = {
        "Not suitable for market": chili_anthacnose_count  
    }

    # Return the report
    return {
        'chili_healthy': {
            'total_count': chili_healthy_count,
            'market_recommendations': healthy_market_recommendations,
        },
        'chili_red': {
            'total_count': chili_red_count,
            'market_recommendations': red_market_recommendations,
        },
        'chili_anthacnose': {
            'total_count': chili_anthacnose_count,
            'market_recommendations': anthacnose_market_recommendations,
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
