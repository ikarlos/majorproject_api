from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
app = FastAPI()

  # Define class names
class_names =['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

  # Load the SavedModel
model = tf.keras.models.load_model('rice_grain_resnet50_model.h5')

  # Show the model architecture
model.summary()

# Default root endpoint
@app.get("/")
async def root():
    return {"message": "Hello world"}
# Example path parameter
@app.get("/name/{name}")
async def name(name: str):
    return {"message": f"Hello {name}"}
# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((180, 180)).convert('RGB')  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    max_index = np.argmax(prediction)
    return {"class_name": class_names[max_index]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)