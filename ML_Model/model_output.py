import os
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

# loads model
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
    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    result = model.predict(img_array)
    
    # Map the prediction to a human-readable label
    labels = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust",
        "Corn_(maize)___Healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)", "Grape___Healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight", "Tomato___Bacterial_spot",
        "Tomato___Early_blight", "Tomato___Healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"
    ]
    
    predicted_class = labels[np.argmax(result)]
    
    return predicted_class

folder_path = "test set"
def predict_images_in_folder(model, folder_path):
    try:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Construct the full path to the image
                image_path = os.path.join(folder_path, filename)

                # Make a prediction for each image
                result = predict_disease(model, image_path)

                # Display the result for each image
                print(f"Image: {filename}, Predicted disease: {result}")

    except Exception as e:
        print("Error: ", str(e))

if __name__ == "__main__":
    # Load the pre-trained model
    model = load_model()
    
    # Take input folder path from the user
    folder_path = input("Enter the path to the folder containing images: ")
    
    # Make predictions for all images in the folder
    predict_images_in_folder(model, folder_path)
