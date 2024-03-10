import pickle 
import model
import numpy as np
from PIL import Image
from lr_utils import load_dataset
import streamlit as st
#import model2
import L_layer_nn

loaded_model = pickle.load(open('my_neural_network_model.sav', 'rb'))
print(len(loaded_model))
num_px = 64

st.title("Cat classifier application")
image = None
image = st.file_uploader("upload an image")

imageref = image


if image is not None:




    # change this to the name of your image file
    #my_image = "gargouille.jpg"   
    # We preprocess the image to fit your algorithm.
    #fname = "images/" + my_image
    image = np.array(Image.open(image).resize((num_px, num_px)))

    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T

    prediction = L_layer_nn.predict(image, 1, loaded_model[0])
    # Use the loaded model for prediction
    #my_predicted_image = L_layer_nn.predict(loaded_model["w"], loaded_model["b"], image)
    #print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    
    classified = int(np.squeeze(prediction))
    if classified == 0:
        print("Not a cat")
        st.text("This picture is not a cat")
        st.image(imageref)

    else:
        print("cat")
        
        st.text("This picture is a cat")
        st.image(imageref)