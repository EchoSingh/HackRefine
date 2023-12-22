from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import os

def load_model():
    # Load the model architecture from the JSON file
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # Create the model from the JSON file
    loaded_model = model_from_json(loaded_model_json)
    
    # Load the weights into the model
    loaded_model.load_weights("model.h5")
    
    print("Model loaded successfully.")
    
    return loaded_model

def predict_disease(model, image_path):
    # Load and preprocess the input image using Pillow
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize the image if needed
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    result = model.predict(img_array)
    
    # Map the prediction to a human-readable label
    labels = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust",
    "Corn_(maize)___Healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight"]
    
    predicted_class = labels[np.argmax(result)]
    
    return predicted_class

def get_image_path():
    # Take input from the user for the image path
    image_path = input("Enter the path to the input image: ")

    return image_path

if __name__ == "__main__":
    # Load the pre-trained model
    model = load_model()

    if model:
        # Take input image address from the user
        image_path = get_image_path()

        try:
            # Make a prediction
            result = predict_disease(model, image_path)
            
            # Display the result
            print("Predicted disease: ", result)
        except Exception as e:
            print("Error: ", str(e))
