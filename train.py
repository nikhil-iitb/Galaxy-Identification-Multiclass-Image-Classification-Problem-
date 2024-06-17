import pickle
import numpy as np
import tensorflow as tf
from src.models import CNN_model, InceptionV3_transfer_model
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from src.utils import plot_training_history
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# def train_q1():
#     # Load data
#     criterion = nn.CrossEntropyLoss()
#     model = q1model()
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_cnn(epochs, lr):
    # Load the processed dataset
    X_train = np.load('./inferences/model_input_data/X_train.npy')
    y_train = np.load('./inferences/model_input_data/y_train.npy')
    X_test = np.load('./inferences/model_input_data/X_test.npy')
    y_test = np.load('./inferences/model_input_data/y_test.npy')
    print('Successfully loaded pre-processed data X_train and y_train and test sample')
    model = CNN_model()
    # Define callbacks
    checkpoint = ModelCheckpoint('best_model_cnn.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # early stopping will make sure it stops earlier when required
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),use_multiprocessing=True, callbacks=[checkpoint, early_stopping])
    # Save training history using pickle
    with open('models/training_history_cnn.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("Training history saved successfully.")
    model.save('models/cnn_model.h5')
    plot_training_history(history)

def train_inception(epochs, lr, batch_size=32):
    X_train_file = './inferences/model_input_data/X_train_inception.npy'
    y_train_file = './inferences/model_input_data/y_train_inception.npy'
    X_test_file = './inferences/model_input_data/X_test_inception.npy'
    y_test_file = './inferences/model_input_data/y_test_inception.npy'
    
    print('Loading pre-processed data...')
    X_train_shape = np.load(X_train_file, mmap_mode='r').shape
    y_train_shape = np.load(y_train_file, mmap_mode='r').shape
    
    total_train_samples = X_train_shape[0]
    total_test_samples = np.load(X_test_file, mmap_mode='r').shape[0]

    print(f'Total training samples: {total_train_samples}, Total test samples: {total_test_samples}')
    
    model = InceptionV3_transfer_model(lr)
    checkpoint = ModelCheckpoint('models/best_model_inception.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    X_test = np.load(X_test_file)
    y_test = np.load(y_test_file)
    
    # Training loop with batch processing
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Training
        for i in range(0, total_train_samples, batch_size):
            X_batch = np.load(X_train_file, mmap_mode='r')[i:i+batch_size]
            y_batch = np.load(y_train_file, mmap_mode='r')[i:i+batch_size]
            
            history = model.fit(X_batch, y_batch, batch_size=batch_size, epochs=1, validation_data=(X_test, y_test), use_multiprocessing=True, callbacks=[checkpoint, early_stopping], verbose=1)
        
    print("Training completed successfully.")
    model.save('models/final_inception_model.h5')
    plot_training_history(history, 'inception')

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 5:
        arg1 = sys.argv[1] #q1 or all
        if(arg1=='q1'):
            # epochs = int(sys.argv[3])
            # lr = float(sys.argv[4])
            # train_q1(epochs, lr)
            sys.exit()
        arg2 = sys.argv[2] #model: cnn or inception
        arg3 = sys.argv[3] #epochs
        arg4 = sys.argv[4] # lr
        if(arg2=='cnn'):
            train_cnn(int(arg3), float(arg4))
        else:
            train_inception(int(arg3), float(arg4))
    else:
        print("Usage: Please put 'all' - model - epochs - learningRate")
