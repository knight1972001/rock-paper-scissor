import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the input shape of the images
input_shape = (224, 224, 3)

# Define the CNN architecture
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Compile the model with the appropriate loss function, optimizer, and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the batch size and number of epochs for training
batch_size = 20
epochs = 10

# Define the data generators for loading the images from the folders
trainDatagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valDatagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

testDatagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

trainGenerator = trainDatagen.flow_from_directory(
    'training/',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

valGenerator = valDatagen.flow_from_directory(
    'validation/',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

testGenerator = testDatagen.flow_from_directory(
    'testing/',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model on the training data
model.fit(
    trainGenerator,
    epochs=epochs,
    validation_data=valGenerator
)

# Evaluate the model on the testing data
score = model.evaluate(testGenerator, verbose=0)

# Print the test accuracy
print('Test accuracy:', score[1])

# Save the trained model to disk
model.save('rock_paper_scissors_model.h5')
