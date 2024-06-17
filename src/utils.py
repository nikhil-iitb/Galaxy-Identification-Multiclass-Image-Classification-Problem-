import os
import random
import numpy as np
from scipy import ndimage
import skimage.transform
import skimage.io
import skimage.filters
import pandas as pd
import tensorflow as tf
from .paths import get_train_images_path, get_labels_path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from skimage.transform import resize
from tqdm import tqdm

output_dir = '../plots'
output_dir_inf = '../inferences'

# Function to look in the data and its relevant plots and distributions
def understand(df):
    df_analysis = df.iloc[:, 1:]
    save_basic_statistics(df_analysis)
    save_missing_values(df_analysis)
    save_distribution_plots(df_analysis)
    save_correlation_heatmap(df_analysis)
    save_pairwise_relationships(df_analysis)

def save_basic_statistics(df):
    stats_df = df.describe()
    stats_df.to_csv(os.path.join(output_dir_inf, 'basic_statistics.csv'))
    print("Basic statistics saved to 'basic_statistics.csv'")

# Check for Missing Values
def save_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values.to_csv(os.path.join(output_dir_inf, 'missing_values.csv'))
    print("Missing values report saved to 'missing_values.csv'")

# Distribution Plot for Probabilities
def save_distribution_plots(df):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns[1:]):  # Skip the first column 'galaxyId'
        plt.subplot(6, 7, i+1)
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(column)
        plt.tight_layout()
    plt.suptitle('Distributions of Probabilistic Scores', y=1.02)
    plt.savefig(os.path.join(output_dir, 'distributions.png'))
    plt.close()
    print("Distribution plots saved to 'plots/distributions.png'")

# Correlation Heatmap
def save_correlation_heatmap(df):
    plt.figure(figsize=(15, 10))
    corr = df.iloc[:, 1:].corr()  # Correlation of the probabilistic scores
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Heatmap of Galaxy Classes')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Correlation heatmap saved to 'correlation_heatmap.png'")

# Pairwise Scatter Plots
def save_pairwise_relationships(df):
    # Plotting a pairplot for the first few classes due to visualization constraints
    selected_columns = df.columns[1:6]  # Selecting first 5 columns for brevity
    sns.pairplot(df[selected_columns])
    plt.suptitle('Pairwise Relationships of Selected Galaxy Classes', y=1.02)
    plt.savefig(os.path.join(output_dir, 'pairwise_relationships.png'))
    plt.close()
    print("Pairwise relationships plot saved to 'pairwise_relationships.png'")

# Get the images of galaxies
def plot_random_galaxy(sample = 5):
    path = get_train_images_path()
    random_image=random.sample(os.listdir(path),sample)
    plt.figure(figsize=(16,5))
    for i in range(sample):
        plt.subplot(1,sample,i+1)
        img=tf.io.read_file(os.path.join(path,random_image[i]))
        img=tf.io.decode_image(img)
        plt.imshow(img)
        plt.title(f'Name: {random_image[i]}\nShape: {img.shape}')
        plt.savefig(os.path.join(output_dir, 'random_galaxy_plot.png'))
        plt.axis(False)
        print('Plotted random galaxies in original form')

# Rotates image by a given angle => used for perturbation
def img_rotate(img, angle):
    return skimage.transform.rotate(img, angle, mode='reflect')

# Flips image horizontally or vertically or both
def img_flip(img, flip_h, flip_v):
    if flip_h:
        img = img[::-1]
    if flip_v:
        img = img[:, ::-1]
    return img

# Translates an image to given coordinates
def img_translate(img, shift_x, shift_y):
    translate_img = np.zeros_like(img, dtype=img.dtype)

    if shift_x >= 0:
        slice_x_src = slice(None, img.shape[0] - shift_x, None)
        slice_x_tgt = slice(shift_x, None, None)
    else:
        slice_x_src = slice(- shift_x, None, None)
        slice_x_tgt = slice(None, img.shape[0] + shift_x, None)

    if shift_y >= 0:
        slice_y_src = slice(None, img.shape[1] - shift_y, None)
        slice_y_tgt = slice(shift_y, None, None)
    else:
        slice_y_src = slice(- shift_y, None, None)
        slice_y_tgt = slice(None, img.shape[1] + shift_y, None)

    translate_img[slice_x_tgt, slice_y_tgt] = img[slice_x_src, slice_y_src]
    return translate_img

# Rescales an image by a given scaling factor
def img_rescale(img, scale_factor):
    zoomed_img = np.zeros_like(img, dtype=img.dtype)
    zoomed = skimage.transform.rescale(img, scale_factor)

    if scale_factor >= 1.0:
        shift_x = (zoomed.shape[0] - img.shape[0]) // 2
        shift_y = (zoomed.shape[1] - img.shape[1]) // 2
        zoomed_img[:,:] = zoomed[shift_x:shift_x+img.shape[0], shift_y:shift_y+img.shape[1]]
    else:
        shift_x = (img.shape[0] - zoomed.shape[0]) // 2
        shift_y = (img.shape[1] - zoomed.shape[1]) // 2
        zoomed_img[shift_x:shift_x+zoomed.shape[0], shift_y:shift_y+zoomed.shape[1]] = zoomed

    return zoomed_img

# Downsamples an image by given downsampling factor
def img_downsample(img, ds_factor):
    return img[::ds_factor, ::ds_factor]

# Returns the image
def get_image(path, x1,y1, shape, crop_size):
    x = plt.imread(path)
    x = x[x1:x1+crop_size[0], y1:y1+crop_size[1]]
    x = resize(x, shape)
    # Normalising image
    x = x/255.
    return x

def get_all_images(dataframe, img_shape, crop_size, orig_shape):
    x1 = (orig_shape[0]-crop_size[0])//2
    y1 = (orig_shape[1]-crop_size[1])//2
   
    sel = dataframe.values
    ids = sel[:,0].astype(int).astype(str)
    y_batch = sel[:,1:]
    x_batch = []
    for i in tqdm(ids):
        x = get_image(get_train_images_path()+'/'+i+'.jpg', x1,y1, shape=img_shape, crop_size=crop_size)
        x_batch.append(x)
    x_batch = np.array(x_batch)
    return x_batch, y_batch

# Crop an image as per given downsamping factor
def img_crop(img, ds_factor):
    size_x = img.shape[0]
    size_y = img.shape[1]

    cropped_size_x = img.shape[0] // ds_factor
    cropped_size_y = img.shape[1] // ds_factor

    shift_x = (size_x - cropped_size_x) // 2
    shift_y = (size_y - cropped_size_y) // 2

    return img[shift_x:shift_x+cropped_size_x, shift_y:shift_y+cropped_size_y]

# Perform local normalisation of image
def img_normalise(img, sigma_mean, sigma_std):
    means = ndimage.gaussian_filter(img, sigma_mean)
    img_centered = img - means
    stds = np.sqrt(ndimage.gaussian_filter(img_centered**2, sigma_std))
    return img_centered / stds

# Check authenticity and correctness of data
def check_authenticity():
    TOLERANCE = 0.00001
    d = pd.read_csv(get_labels_path())
    targets = d.iloc[:, 1:].to_numpy().astype('float32')
    ids = d.iloc[:, 0].to_numpy().astype('int32')
    questions = [
    targets[:, 0:3], # 1.1 - 1.3,
    targets[:, 3:5], # 2.1 - 2.2
    targets[:, 5:7], # 3.1 - 3.2
    targets[:, 7:9], # 4.1 - 4.2
    targets[:, 9:13], # 5.1 - 5.4
    targets[:, 13:15], # 6.1 - 6.2
    targets[:, 15:18], # 7.1 - 7.3
    targets[:, 18:25], # 8.1 - 8.7
    targets[:, 25:28], # 9.1 - 9.3
    targets[:, 28:31], # 10.1 - 10.3
    targets[:, 31:37], # 11.1 - 11.6
    ]
    sums = [
    np.ones(targets.shape[0]), # 1, # Q1
    questions[0][:, 1], # Q2
    questions[1][:, 1], # Q3
    questions[1][:, 1], # Q4
    questions[1][:, 1], # Q5
    np.ones(targets.shape[0]), # 1, # Q6
    questions[0][:, 0], # Q7
    questions[5][:, 0], # Q8
    questions[1][:, 0], # Q9
    questions[3][:, 0], # Q10
    questions[3][:, 0], # Q11
    ]
    num_total_violations = 0
    affected_ids = set()

    for k, desired_sums in enumerate(sums):
        print(f"QUESTION {k+1}")
        actual_sums = questions[k].sum(1)
        difference = abs(desired_sums - actual_sums)
        indices_violated = difference > TOLERANCE
        ids_violated = ids[indices_violated]
        num_violations = len(ids_violated)
        if num_violations > 0:
            print(f"{num_violations} constraint violations.")
            num_total_violations += num_violations
            for id_violated, d_s, a_s in zip(ids_violated, desired_sums[indices_violated], actual_sums[indices_violated]):
                print(f"violated by {id_violated}, sum should be {d_s} but it is {a_s}")
                affected_ids.add(id_violated)
        else:
            print("No constraint violations.")

        print

    print
    print(f"{num_total_violations} violations in total.")
    print(f"{len(affected_ids)} data points violate constraints.")


def plot_training_history(history, name='cnn'):
    # Retrieve values from history object
    # accuracy = history.history.get('accuracy')
    # val_accuracy = history.history.get('val_accuracy')
    # loss = history.history.get('loss')
    # val_loss = history.history.get('val_loss')
    # mse = history.history.get('mse')
    # val_mse = history.history.get('val_mse')
    loss = [0.0604, 19.3039, 1.8351, 2.6661, 8.8902, 8.1097, 3.4875, 0.3545, 0.5307, 2.5609, 3.9247, 3.3965, 1.7750, 0.3083, 0.1319, 0.9323, 1.8444, 1.9802, 1.3209]
    accuracy = [0.4831, 0.4797, 0.5152, 8.4459e-04, 0.0025, 0.0017, 0.0025, 0.0017, 0.4713, 0.5144, 0.4932, 0.4958, 0.4848, 0.5093, 0.4671, 0.0017, 0.0025, 0.0025, 0.0025]
    mse = [0.2458, 4.3936, 1.3547, 1.6328, 2.9816, 2.8477, 1.8675, 0.5954, 0.7285, 1.6003, 1.9811, 1.8430, 1.3323, 0.5553, 0.3631, 0.9656, 1.3581, 1.4072, 1.1493]
    val_loss = [19.3316, 1.8594, 2.6480, 8.8229, 8.0175, 3.4306, 0.3514, 0.5268, 2.5771, 3.9520, 3.4295, 1.7403, 0.3320, 0.1370, 0.9590, 1.8431, 1.9703, 1.3258, 0.4880]
    val_accuracy = [0.4915, 0.4915, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.4915, 0.4915, 0.4915, 0.4915, 0.4915, 0.4915, 0.4813, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]
    val_mse = [4.3968, 1.3635, 1.6272, 2.9703, 2.8315, 1.8522, 0.5928, 0.7257, 1.6053, 1.9879, 1.8519, 1.3192, 0.5761, 0.3701, 0.9792, 1.3576, 1.4037, 1.1514, 0.6985]

    # Plot Accuracy
    if accuracy and val_accuracy:
        plt.figure()
        plt.plot(accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, name+'_accuracy.png'))
        plt.show()

    # Plot Loss
    if loss and val_loss:
        plt.figure()
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, name+'_loss.png'))
        plt.show()

    # Plot Mean Squared Error
    if mse and val_mse:
        plt.figure()
        plt.plot(mse, label='Training RMSE')
        plt.plot(val_mse, label='Validation RMSE')
        plt.title('Model Root Mean Squared Error')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(os.path.join(output_dir, name+'_rmse.png'))
        plt.show()








