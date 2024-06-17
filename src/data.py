import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_all_images, plot_random_galaxy, understand
from paths import get_labels_path

def prepare_data():
    df = pd.read_csv(get_labels_path())
    df_train, df_test = train_test_split(df, test_size=0.2)
    print("Shape of the train set: ",df_train.shape)
    print("Shape of the test set: ",df_test.shape)

    # Uncomment to understand data
    # understand(df)

    # Uncomment to generate random galaxy plots
    # plot_random_galaxy() and get the orig shape, used in next function

    # Generate dataset for all images and save for further use
    # create_and_save_model_input(df_train, df_test)
    create_data_inception(df_train, df_test)


def create_and_save_model_input(df_train, df_test):
    ORIG_SHAPE = (424,424)
    CROP_SIZE = (256,256)
    IMG_SHAPE = (64,64)
    # Preprocess image 
    X_train, y_train = get_all_images(df_train, IMG_SHAPE, CROP_SIZE, ORIG_SHAPE)
    X_test, y_test = get_all_images(df_test, IMG_SHAPE, CROP_SIZE, ORIG_SHAPE)
    print("Shape of the X-train set: ",X_train.shape)
    print("Shape of the Y-train set: ",y_train.shape)
    np.save('../inferences/model_input_data/X_train.npy', X_train)
    np.save('../inferences/model_input_data/y_train.npy', y_train)
    np.save('../inferences/model_input_data/X_test.npy', X_test)
    np.save('../inferences/model_input_data/y_test.npy', y_test)
    print("X_train, X_test, y_train, and y_test have been saved as .npy files.")

def create_data_inception(df_train, df_test):
    ORIG_SHAPE = (424,424)
    CROP_SIZE = (256,256)
    IMG_SHAPE = (75,75)
    X_train, y_train = get_all_images(df_train, IMG_SHAPE, CROP_SIZE, ORIG_SHAPE)
    X_test, y_test = get_all_images(df_test, IMG_SHAPE, CROP_SIZE, ORIG_SHAPE)
    np.save('../inferences/model_input_data/X_train_inception.npy', X_train)
    np.save('../inferences/model_input_data/y_train_inception.npy', y_train)
    np.save('../inferences/model_input_data/X_test_inception.npy', X_test)
    np.save('../inferences/model_input_data/y_test_inception.npy', y_test)
    print("X_train, X_test, y_train, and y_test have been saved as .npy files.")


    