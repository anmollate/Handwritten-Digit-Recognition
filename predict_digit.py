import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

st.title("🖼️ HANDWRITTEN DIGIT RECOGNITION")
st.markdown("Upload a handwritten digit image and get the model's prediction.")

#Loading The Model
model=load_model("mnist_model.h5")

uploaded_img=st.file_uploader("Upload Your Handwritten Digit Image",type=["png","jpeg","jpg"])
#Loading The Image Coverting It Into Grayscale And Resizing It Into 28x28 Pixel Frame
# img=Image.open("imageone.jpg").convert("L").resize((28,28))

if uploaded_img:
    img=Image.open(uploaded_img).convert("L").resize((28,28))
    img=np.array(img)
    img=255-img
    img=img.astype("float32")/255.0
    img=img.reshape(1,28,28,1)
    pred=model.predict(img)
    digit=np.argmax(pred)
    st.subheader(f"🧠 Predicted Digit: **{digit}**")

    st.subheader("📊Probability Distribution Graph:")
    st.bar_chart(pred[0])

else:
    st.info("Please upload an image to see the prediction.")


# # The Multidimensional Pixel Values Of Image Is Converted Into Numpy Array
# img=np.array(img)

# #We invert the image so that the digit looks white on a black background, just like the images the model was trained on (like in the MNIST dataset), which helps the model recognize the digit more accurately.
# img=255-img

# #Normalizing The Pixel Values
# img=img.astype("float32")/255.0

# #Reshaping .reshape(dataset_size,width,height,Grayscale(1)/RGB(3))
# img=img.reshape(1,28,28,1)

# #Predicting The Number
# pred=model.predict(img)
# digit=np.argmax(pred)

# print("Predicted Digit:",digit)