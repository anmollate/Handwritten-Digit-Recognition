import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

#Loading The Model
model=load_model("mnist_model.h5")

#Loading The Image Coverting It Into Grayscale And Resizing It Into 28x28 Pixel Frame
img=Image.open("imageone.jpg").convert("L").resize((28,28))

# The Multidimensional Pixel Values Of Image Is Converted Into Numpy Array
img=np.array(img)

#We invert the image so that the digit looks white on a black background, just like the images the model was trained on (like in the MNIST dataset), which helps the model recognize the digit more accurately.
img=255-img

#Normalizing The Pixel Values
img=img.astype("float32")/255.0

#Reshaping .reshape(dataset_size,width,height,Grayscale(1)/RGB(3))
img=img.reshape(1,28,28,1)

#Predicting The Number
pred=model.predict(img)
digit=np.argmax(pred)

print("Predicted Digit:",digit)