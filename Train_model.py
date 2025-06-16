import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical


(x_train,y_train),(x_test,y_test)=datasets.mnist.load_data()

#Normalizing And Reshaping The Images For More Details Refer CNN_INTRODUCTION.ipynb
# .asfloat() to increase the effeciency and speed as after division the result would be float its faster and easier to divide float value with a float value
x_train=x_train.reshape(-1,28,28,1).astype("float32")/255.0
x_test=x_test.reshape(-1,28,28,1).astype("float32")/255.0

#Converts class labels (like 3) into one-hot vectors (like [0,0,0,1,...]).
y_train=tf.keras.utils.to_categorical(y_train, 10)
y_test=tf.keras.utils.to_categorical(y_test, 10)

# Builds A Sequential Model One Layer After Other In Which Output Of One Layer Is Influenced By The Previous Layer
"""
Model Building Procedure:
Convolution 2D: 32 Filters-Applies The 32 Filters Onto The 28x28 Image With The Help Of 3x3 Frame Recognizes Edges, Lines etc
MaxPooling2D: Takes The Maximum Pixel Value From The Convoluted Result
Convolution 2D: 64 Filters-Applies The 64 Filters Onto The 28x28 Image With The Help Of 3x3 Frame Recognizes Curves, Loops etc
MaxPooling2D: Takes The Maximum Pixel Value From The Convoluted Result
Flatten(): Flattens The Pixel Multi Dimensional Matrix Into One Dimensional Array, Assume It As Converting A 2x2 Matrix Into 4 Elements List
Dense(128): All 128 neurons see the entire 784 input. Each one learns a different feature combination.                       
Dense(10): All 10 neurons see the 128 outputs and learn to score each class (0–9). Softmax turns scores into probabilities.

"""
model=Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    
    Conv2D(64,(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Uses the Adam optimizer to update weights, use categorical crossentropy to calculate how wrong you are, and show me how accurate you're getting during training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Finally Train The Model On Trained Data And Validate It On Test Data That Is Test The Model On Validation_data, "epochs" Refers To How Many Times The Model Loops On The Train Data,  Batch size controls how many samples the model sees at once before updating its weights.
model.fit(x_train,y_train, epochs=5, batch_size=32, validation_data=(x_test,y_test))

#Saving The Model
model.save("mnist_model.h5")
