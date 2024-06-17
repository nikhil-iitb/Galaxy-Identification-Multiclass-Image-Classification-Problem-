import io
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D
from keras import backend as K
import numpy as np
import tensorflow as tf
import tensorflow as tf
from keras.applications import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers.legacy import Adam
import torch.nn as nn
import torch.nn.functional as F

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def save_model_info(model, path='inferences/model_info/model_summary_and_params.txt'):
    # Save model summary and parameters to a file
    with io.StringIO() as stream:
        model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
        summary_str = stream.getvalue()

    # Count the number of trainable parameters
    trainable_params = int(np.sum([K.count_params(w) for w in model.trainable_weights]))

    # Write the summary and parameters to a file
    with open(path, "w") as f:
        f.write("Model Summary:\n")
        f.write(summary_str)
        f.write("\n")
        f.write(f"Total Trainable Parameters: {trainable_params}\n")

def CNN_model():
    ORIG_SHAPE = (424,424)
    CROP_SIZE = (256,256)
    IMG_SHAPE = (64,64)

    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(64, (3, 3), input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())

    # Dense layers
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Output layer
    model.add(Dense(37))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    # Save the model as well
    save_model_info(model)
    return model

def InceptionV3_transfer_model(input_shape=(75, 75, 3), learning_rate=0.001):
    # Load InceptionV3 pretrained on ImageNet data
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75,75,3))
    
    # Freeze the layers in the base InceptionV3 model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers for our regression task
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='linear')(x)  # Use a single output node with linear activation for regression
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', root_mean_squared_error])
    
    # Save model architecture (if save_model_info is a valid function in your context)
    save_model_info(model, 'inferences/model_info/inception_model_info')
    
    return model


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def q1model():
    model = Net()
    return model
