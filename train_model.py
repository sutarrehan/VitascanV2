import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

train_dir = "dataset/train"
val_dir = "dataset/val"

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=(227,227), class_mode='categorical'
)
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(227,227), class_mode='categorical'
)

print("Class indices:", train_gen.class_indices)

# Class balancing
class_counts = np.bincount(train_gen.classes)
total = sum(class_counts)
class_weight = {i: total/(4*class_counts[i]) for i in range(4)}
print("Class weights:", class_weight)

# Simple AlexNet
model = Sequential([
    Conv2D(96,(11,11),strides=(4,4),activation='relu',input_shape=(227,227,3)),
    MaxPooling2D((3,3),strides=(2,2)),
    Conv2D(256,(5,5),padding='same',activation='relu'),
    MaxPooling2D((3,3),strides=(2,2)),
    Flatten(),
    Dense(1024,activation='relu'),
    Dropout(0.5),
    Dense(4,activation='softmax')
])

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=15, class_weight=class_weight)

model.save("alexnet_model.h5")
print("MODEL TRAINED & SAVED!")
