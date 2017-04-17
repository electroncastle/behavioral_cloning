#!/usr/bin/env python3
"""
- Combines multiple driving logs from the Udacity driving simulator
- preprocessed images and labels for the CNN training
- Implemented CNN model for autonomous driving 
- Trains the CNN for autonomous driving
- Evaluates the CNN
"""

__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__version__ = "0.0.4"
__status__ = "research"


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'

import sys
import time
from collections import namedtuple
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

from keras.models import load_model, Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.convolutional import Convolution2D
from keras.layers import Activation, Dense, Dropout, Flatten, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
import keras.backend.tensorflow_backend as K

# workaround for bug in the dropout API (see http://stackoverflow.com/a/40066895)
import tensorflow as tf
tf.python.control_flow_ops = tf

import matplotlib.pyplot as plt
from PIL import Image
import pickle
import cv2


"""
Hyper paramaters to train the CNN
"""
HParams = namedtuple('HParams',
                     'batch_size, ' # number of samples in a batch
                     'epochs, '     # number of epochs
                     'learning_rate, '  # learning rate
                     'l2_weight_decay') # L2 weight decay


class Frame:
    """
    Class that hold information about each video frame, specifically the image filename, steering angle and speed
    It also performs image preprocessing and augmentation.
    """

    LEFT = 0    # Left camera index
    CENTER = 1  # Center camera index
    RIGHT = 2   # Right camera index

    STEERING_TO_DEGREES = 25.0

    def __init__(self, filename='', steering=0.0, speed=0.0, camera=1):
        """
        :param filename: image filename
        :param steering:  steering angle in radians
        :param speed: speed in mph
        :param camera: camera index Frame.LEFT, Frame.RIGHT or Frame.CENTER
        """

        self.image = None
        self.filename = filename
        self.steering = steering
        self.camera = camera # 0=left, 1=center, 2=right

        # Normalize speed to range [-0.5, 0.5]
        self.speed = speed/31.0 - 0.5

        self.use_image_cache = True

        # Add this offset to left camera and subtract from right camera steering angles
        self.steering_offset = 0.2

        # (width, height)
        self.new_size = (64,64)

        # Crop 60 pixels from top and 23 from the bottom
        self.crop = (60,23)

        # Probability of the train image being flipped right/left
        self.right_left_flip_p = 0.5




    def get_frame_shape(self):
        """
        :return: Shape of the image in format (height, width, channels)
        """
        image = self.get_image()
        return image.shape

    def get_label(self):
        """
        :return: Steering angle for this frame. Steering angle for the right and left camera get adjusted by
         the steering offset
        """
        label = self.steering
        if self.camera == Frame.LEFT: # left
            label += self.steering_offset
        elif self.camera == Frame.RIGHT: # Right
            label -= self.steering_offset

        return label

    def preprocess(self, image):
        """
        Crops and resizes given image
        :param image:
        :return:
        """
        h, w = image.shape[0], image.shape[1]

        # Crop
        image = image[self.crop[0]:h - self.crop[1], 0:w]

        # Resize
        image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_AREA)

        return image

    def get_image(self):
        """
        Loads and preprocesses image and optionally stores it in a memory
        Preprocessing consists of cropping followed by resizing according
        to parameters given in the Frame class constructor

        :param image: numpy array with raw image data
        :return: numpy array or shape (h,w,c) with raw image data encoded as np.float32
        """

        # Load image directly from a filesystem if it doesn't exist in the image cache
        if not self.use_image_cache or self.image is None:
            image = Image.open(self.filename).convert('RGB')
            image = np.asarray(image)

            image = self.preprocess(image)

            # Store image if caching is on
            if self.use_image_cache:
                self.image = image.copy()
        else:
            # Load image from cache
            image = self.image.copy()


        # Convert to float and normalize
        image = np.asarray(image, dtype=np.float32)
        image = image/255.0 - 0.5

        return image



    def get_image_label(self, augment=True):
        """
        Returns image and steering angle. Implicitly the data is augmented by right-left flipping
        with probability 50%
        :param augment:
        :return: image, label
        """
        label = self.get_label()
        image = self.get_image()

        if augment:
            if np.random.uniform(0, 1.0) > self.right_left_flip_p:
                image = np.fliplr(image)
                label = -label

        return image, np.asarray([label, self.speed])


    # def show(self, original=False):
    #     if original:
    #         image = self.load_image(image_str=None)
    #     else:
    #         image = self.preprocess()
    #
    #     if image.shape[2] == 1:
    #         plt.imshow(image, clmap='gray')
    #     else:
    #         plt.imshow(image)
    #
    #     plt.show(block=False)
    #     plt.waitforbuttonpress()


class DriverTrainer:
    """
    Holds the CNN model and training routines. It is also responsible for reading recorded driving data from multiple
    csv files and combine them. Lastly it allows to run an evaluation of the trained model on the validataion
    dataset and plot the ground truth and predicted data for the steering angles and speed
    """

    def __init__(self):

        # Some hyper parameters
        self.hp = HParams(batch_size=512,
                          epochs=30,
                          learning_rate=1e-4,
                          l2_weight_decay=1e-5)

        # default CPU/GPU device on which to run the training
        self.device = '/gpu:0'

        # Whether to run low pass filter on the steering angles and speed data
        self.filter_data = False

        # Filter kerne size for a box LP filter to smooth the steering angles and speed data
        self.filter_size = 5

        # Splits data to 90% training and 10% validation samples
        self.test_val_split = 0.10

    def filter(self, data):
        """
        Runs a low pass box filter on the steering and speed data. Data must not have been shuffeled yet !!
        :param data: a list of Frame objects
        """
        angles = []
        speed = []
        for sample in data:
            angles.append(sample.steering)
            speed.append(sample.speed)

        window = np.ones(int(self.filter_size)) / float(self.filter_size)
        angles_filtered= np.convolve(angles, window, 'same')
        speed_filtered= np.convolve(speed, window, 'same')

        for i, sample in enumerate(data):
            sample.steering = angles_filtered[i]
            sample.speed = speed_filtered[i]


    def parse_log_csv(self, filename):
        """
        Loads csv file with the drive data and for each record creates a Frame object with corect
        camera indexes. It then filters steering angles and speed and finally returns a continuous list
        of the Frame objects for all right, left and center cameras
        :param filename: csv file
        :return: a list of Frame objects for all images from all cameras
        """

        root, basefilename = os.path.split(filename)
        f = open(filename, mode='rt')

        left_frames=[]
        center_frames=[]
        right_frames=[]
        for i, line in enumerate(f):
            if i == 0: continue
            items = line.split(',')
            steering = float(items[3])
            speed = float(items[6])

            # center camera
            path, file = os.path.split(items[0].strip())
            center_img_filename = os.path.join(root,'IMG',file)
            center_frames.append(Frame(center_img_filename, steering, speed, Frame.CENTER))

            # left camera
            path, file = os.path.split(items[1].strip())
            left_img_filename = os.path.join(root,'IMG',file)
            left_frames.append(Frame(left_img_filename, steering, speed, Frame.LEFT))

            # right camera
            path, file = os.path.split(items[2].strip())
            right_img_filename = os.path.join(root,'IMG',file)
            right_frames.append(Frame(right_img_filename, steering, speed, Frame.RIGHT))

        # Run low pass filters on all samples. The filter has a box kernel with size = self.filter_size
        if self.filter_data:
            self.filter(center_frames)
            self.filter(left_frames)
            self.filter(right_frames)

        # Join frames from all cameras into a single list
        return center_frames+left_frames+right_frames



    def assemble_dataset(self, root, parts):
        """
        Iterates through the list of directories with recorded drive data, loads the csv files. Finally it splits the
        data into training and validatation datasets and stores them into self.train_data_filename and self.test_data_filename
        :param root: root directory used for all subdirectories given in the parts list
        :param parts: list of directory with individual drive data logs
        :return: train_dataset, validataion_dataset
        """
        data = []
        for part in parts:
            part = os.path.join(root, part)
            for subdir, dirs, files in os.walk(part):
                for file in files:
                    if file == 'driving_log.csv':
                        print("processing :", os.path.join(subdir, file))
                        if dirs[0] != 'IMG':
                            print("Missing IMG directory in ", subdir)
                            break

                        data += self.parse_log_csv(os.path.join(subdir, file))

        train = np.asarray(data)

        print("\nPickling all data")
        with open('data.p', mode='wb') as f:
            pickle.dump(train, f, protocol=4)

        print("Splitting data into training and validation datasets with test size being ",self.test_val_split, " of the total size.")
        train, val = train_test_split(train, test_size=self.test_val_split, random_state=42)
        print("Training dataset size: ", len(train))
        print("Validation dataset size: ",len(val))

        return train, val


    def pickle_datasets(self, train, val, store_image_data=False):
        """
        Stores the train and val dataset to pickle file. The datasets are lists of
        Frame object. If store_image_data=True all images are loaded from the filesystem
        in the memory by each Frame object before dumping the data into the pickle files.
        :param train: training dataset, a list of Frame objects
        :param val: validation dataset, a list of Frame objects
        :param store_image_data: if True all images will be first loaded to memory and then stored along with
        other Frame members for each sample
        """

        if store_image_data:
            print("Loading images for training dataset")
            n = len(train)
            for i, sample in enumerate(train):
                sample.use_image_cache = True
                # Calling get_image() will load the image and store within the object
                sample.get_image()

                # show progress
                sys.stdout.write('\r')
                sys.stdout.write("[%-50s] %d%%" % ('=' * int(50.0 * i / n), int(100.0 * i / n)))
                sys.stdout.flush()

        print("\nPickling training dataset")
        with open('train_images.p', mode='wb') as f:
            pickle.dump(train, f, protocol=4)

        if store_image_data:
            print("Loading images for validation dataset")
            n = len(val)
            for i, sample in enumerate(val):
                sample.use_image_cache = True
                sample.get_image()

                # show progress
                sys.stdout.write('\r')
                sys.stdout.write("[%-50s] %d%%" % ('=' * int(50.0 * i / n), int(100.0 * i / n)))
                sys.stdout.flush()

        print("\nPickling training dataset")
        with open('val_images.p', mode='wb') as f:
            pickle.dump(val, f, protocol=4)

        print("Done")

    def unpickle_datasets(self):
        """
        Loads pickeled datasets from  filesystem and returns them. The datasets are list of Frame object
        :return: a tuple of (train, val) dataset data. If the pickle files don't exist a tuple (None, None)
        gets returned
        """
        try:
            print("Loading training dataset from train_images.p")
            with open('train_images.p', 'rb') as f:
                train_data = pickle.load(f)

            print("Loading validation dataset from val_images.p")
            with open('val_images.p', 'rb') as f:
                val_data = pickle.load(f)
        except:
            return None, None

        return train_data, val_data

    def batch_generator(self, data, train=True, shuffle_data=True):
        """
        Prepares batch data training/validation. The batch generator runs in a separate thread in parallel with the
        thread running the tensorflow graph thus it doesn't slow down the training proceess unless it's slower than
        processing the batch on GPU which is likely when the images are loaded from the filesystem. Due to this fact
        the images are cached in RAM which significanlty (~5x on a machine with Titan X, 32GB RAM and SSD drive)
        reduces training time. The batch generator produces data infinitely. When it reaches the end of dataset it
        it shuffles it and starts from begining.

        :param data: a list of Frame objects
        :param train: if true full image agumentation is performed
        :param shuffle_data: if true samples will get shuffled at the beginning of each epoch
        :return: the python generator data link
        """
        start = 0
        n_samples = len(data)
        while True:

            # When we reach the end of the training dataset start from the beginning again.
            if start>n_samples:
                start = 0

            # Shuffle training data every time we are starting new epoch
            if start == 0 and shuffle_data:
                data = shuffle(data)

            end = start + self.hp.batch_size

            batch = data[start:end]
            x = []
            y = []
            for sample in batch:
                # Read the image and label. get_image_label() also preprocesses and augment the image and label data
                image, label = sample.get_image_label(augment=train)
                x.append(image)
                y.append(label)

            start += self.hp.batch_size

            x = np.asarray(x, dtype='float32')
            y = np.asarray(y, dtype='float32')
            yield (x, y)


    def long_model(self, input_shape):
        model = Sequential()

        model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='same', W_regularizer=l2(self.hp.l2_weight_decay),
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='same', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode='same', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode='same', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.5))

        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Activation('elu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.5))

        model.add(Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Activation('elu'))
        model.add(Convolution2D(128, 3, 3, subsample=(2, 2), border_mode='same', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.5))

        model.add(Flatten())
        model.add(Dense(128, activation='elu', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Dropout(.5))
        model.add(Dense(96, activation='elu', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Dropout(.5))
        model.add(Dense(64, activation='elu', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Dense(10, activation='elu', W_regularizer=l2(self.hp.l2_weight_decay)))
        model.add(Dense(2, activation='linear', W_regularizer=l2(0.0)))

        model.compile(optimizer=Adam(lr=self.hp.learning_rate), loss='mean_squared_error')

        return model


    def train(self, train_data, test_data, tune_weights_file=None):
        """
        Loads and trains the CNN model according to hyperparameters set in the constructor. Training is done
        on self.device specified in the constructor. A model_diagram.pdf file with a diagram of the Keras model
        will be stored in current directory. After the training finishes model_weights.h5, model.h5 and
        train_history.p will be stored. Finally plot with the validation and training loss will be shown and stored
        in train_val_loss_plot.pdf

        :param train_data: training dataset, a list of Frame objects
        :param test_data: validation dataset, a list of Frame objects
        :param tune_weights_file: a h5 filename with CNN weights. If given the CNN will be initialized with the stored
         weights
        """

        # Creates training and validation generators
        train_batch_generator = self.batch_generator(train_data)
        val_batch_generator = self.batch_generator(test_data, train=False)
        input_shape = train_data[0].get_frame_shape()

        # Print some information
        print("Train samples: ", len(train_data))
        print("Validation samples: ", len(test_data))
        print("CNN input shape: ", input_shape)
        print("Hyper parameters: ", self.hp)

        # Selects a device to tran the CNN on
        with K.tf.device(self.device):
            K.set_session(
                K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))

        # Loads model definition
        model = self.long_model(input_shape)
        #model = self.simple_model(input_shape)
        model.summary()

        plot(model, to_file='model_diagram.pdf', show_shapes=True, show_layer_names=True)

        if tune_weights_file is not None:
            model.load_weights(tune_weights_file)
            #model = load_model(tune_weights_file)


        callbacks=[
            # This callback will save model data after every epoch if the validation loss improved
            ModelCheckpoint('model.{epoch:02d}-{val_loss:.4f}.h5', verbose=1, save_best_only=True, save_weights_only=False)
            #TensorBoard(log_dir='./tflog', histogram_freq=0, write_graph=False, write_images=False)
        ]

        # Starts training
        start_time = time.time()
        h = model.fit_generator(generator=train_batch_generator, samples_per_epoch=train_data.shape[0],
                                validation_data=val_batch_generator, nb_val_samples=test_data.shape[0],
                                nb_epoch=self.hp.epochs,
                                max_q_size=10,
                                callbacks=callbacks)

        took = int(time.time() - start_time)
        print('Training took: {}:{} min:sec'.format(took // 60, took % 60))

        # Save model, model weights and training history
        model.save_weights('model_weights.h5')
        model.save('model.h5')

        history = h.history
        with open('train_history.p', 'wb') as f:
            pickle.dump(history, f)

        self.plot_train_history('train_history.p')


    def plot_train_history(self, history_file):
        """
        Plots training and validation history from a given history file
        :param history_file: pickeled history data
        """

        with open(history_file, 'rb') as f:
            history = pickle.load(f)

        # Plot and save the training and validation loss for each epoch
        fig = plt.figure(1)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        fig.savefig('train_val_loss_plot.pdf')
        plt.show(block=True)


    def evaluate(self, model_filename, data):
        """
        Evaluates a given model on given dataset and stores results in predicted_steering.p and predicted_speed.p
        :param model_filename: h5 filename representing Keras model
        :param data: validation dataset, a list of Frame objects
        """

        # Select CPU/GPU device
        with K.tf.device(self.device):
            K.set_session(
                K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))

        n = len(data)
        n = 20*self.hp.batch_size
        model = load_model(model_filename)

        print("Evaluating steering angle and speed with model: ",model_filename)
        print("Validation dataset size: ", n)

        gt_angles = []
        predicted_angles = []
        gt_speed = []
        predicted_speed = []

        batch_generator = self.batch_generator(data, train=False, shuffle_data=False)
        predictions = model.predict_generator(generator=batch_generator, val_samples=n)

        for i in range(n):
            val_sample = data[i]
            steering_angle, target_speed = predictions[i]
            gt_angles.append(val_sample.steering)
            gt_speed.append((val_sample.speed+0.5)*31.0)

            predicted_angles.append(steering_angle)
            predicted_speed.append((target_speed+0.5)*31.0)


        np.save('predicted_steering', predicted_angles )
        np.save('gt_steering', gt_angles)

        np.save('predicted_speed', predicted_speed)
        np.save('gt_speed', gt_speed)

        self.plot_evaluation_result()


    def show_steering_histogram(self, data):
        """
         Plots a histogram of the steering wheel angles and stores
         the plot in steering_angles_histogram.pdf
         :param data: a list of Frame objects
        """
        print("Showing histogram of steering wheel angles ")
        angles = []
        for sample in data:
            angles.append(sample.get_label())

        angles = np.asarray(angles)
        n_bins = 100
        steering_range = angles.max() - angles.min()
        print("Steering range ",steering_range)
        bin_width_degrees = steering_range*Frame.STEERING_TO_DEGREES/n_bins
        print("Histogram bin width in degrees ", bin_width_degrees)

        fig = plt.figure(1)
        plt.hist(angles, bins=n_bins)
        plt.title('Histogram of steering wheel angles. # samples {}'.format(len(angles)))
        plt.xlabel('Steering angle bins (1 bin={:.2} degrees)'.format(bin_width_degrees))
        plt.ylabel('Samples per bin')
        plt.plot()
        print("Saving histogram in steering_angles_histogram.pdf")
        fig.savefig('steering_angles_histogram.pdf')
        plt.show(block=True)


    def plot_evaluation_result(self):
        """
        Loads predicted and ground truth steering wheel angles and plots them against each other.
        The plot shows 150 following samples every time a key is pressed. Root Mean Square Error (RMSE) is also
        calculated and shown in the plot title. Every plot is saved in file ground_truth_predicted_steering_speed.pdf
        """
        predicted_angles = np.load('predicted_steering.npy')
        gt_angles = np.load('gt_steering.npy')

        predicted_speed = np.load('predicted_speed.npy')
        gt_speed = np.load('gt_speed.npy')

        n = len(gt_angles)
        gt_predicted_sum = (gt_angles - predicted_angles)
        rmse = np.sqrt(np.dot(gt_predicted_sum, gt_predicted_sum) / n)
        print("Steering angle RMSE: ", rmse)

        gt_predicted_sum = (gt_speed - predicted_speed)
        rmse_speed = np.sqrt(np.dot(gt_predicted_sum, gt_predicted_sum) / n)
        print("Speed RMSE: ", rmse_speed)


        fig = plt.figure(1)
        start = 5000
        count = 250
        while True:
            plt.subplot(2, 1, 1)
            plt.hold(False)
            plt.plot(gt_angles[start:start+count], 'b')
            plt.hold(True)
            plt.plot(predicted_angles[start:start+count], 'r')
            plt.ylim((-1.2, 1.5))

            plt.title('Ground truth vs predicted steering angle RMS={:.5}'.format(rmse))
            plt.xlabel('Time (sample index)')
            plt.ylabel('Steering angle in radians')
            plt.legend(['Ground Truth', 'Predicted'], loc='upper right')

            plt.subplot(2, 1, 2)
            plt.hold(False)
            plt.plot(gt_speed[start:start+count], 'b')
            plt.hold(True)
            plt.plot(predicted_speed[start:start+count], 'r')
            plt.ylim((0, 40))

            plt.title('Ground truth vs predicted speed RMS={:.5}'.format(rmse_speed))
            plt.xlabel('Time (sample index)')
            plt.ylabel('Speed in mph')
            plt.legend(['Ground Truth', 'Predicted'], loc='upper right')
            plt.tight_layout()

            fig.savefig('ground_truth_predicted_steering_speed.pdf')
            plt.show(block=False)
            plt.waitforbuttonpress()
            start += count

############################################################################################################
# Entry point
############################################################################################################
def main():
    parser = argparse.ArgumentParser(description='Trains CNN to predict steering wheel angle and speed '
                                                 'from an on board camera images')
    parser.add_argument('--model-weights', type=str,
        help='Path to h5 file with model weights for fine tuning.')

    parser.add_argument('--plot-steering-histogram', default=False,  action='store_true',
        help='Plots histogram of steering angles and stores it in steering_angles_histogram.pdf')

    parser.add_argument('--eval-model', type=str,
        help='Evaluate a given model on validation dataset and stores result in ground_truth_predicted_steering_speed.pdf')

    parser.add_argument('--plot-history', type=str, default=None,
        help='Plots validataion and training los from a give history file. Plot is saved in train_val_loss_plot.pdf')

    parser.add_argument('--store-images', default=False,  action='store_true',
        help='Store all preprocessed images in the pickle dataset files to speed up the training. Note even if the '
             '--store-imaegs is turned off (default) images will be cached in the memory during the first epoch and '
             'subsequently used during following epochs')

    parser.add_argument('--device',  type=str, nargs='?', default='/gpu:0',
        help='GPU/CPU device to use for training. Default is: /gpu:0'
    )

    args = parser.parse_args()

    trainer = DriverTrainer()
    trainer.device = args.device

    # Training data for track 1
    data_dirs_t1 = [
                 '/home/jiri/AAA/t1-fwd-center',
                 '/home/jiri/AAA/t1-fwd-center-2',
                 '/home/jiri/AAA/t1-fwd-center-3',
                 '/home/jiri/AAA/t1-fwd-center-4',
                 '/home/jiri/AAA/t1-fwd-center-5',
                 '/home/jiri/AAA/t1-fwd-center-6',
                 '/home/jiri/AAA/t1-fwd-center-8',
                 '/home/jiri/AAA/t1-fwd-center-9',
                 '/home/jiri/AAA/t1-bkwd-center',
                 '../data'  # original Udacity dataset
                 ]

    # Training data for track 2 (Jungle)
    data_dirs_t2 = [
                    '/home/jiri/AAA/t2-fwd-center-vs-0',
                    '/home/jiri/AAA/t2-fwd-center-vs-1',
                    '/home/jiri/AAA/t2-fwd-center-vs-2',
                    '/home/jiri/AAA/t2-fwd-center-vs-3',
                    '/home/jiri/AAA/t2-fwd-center-vs-4',
                    '/home/jiri/AAA/t2-fwd-center-vs-5',
                    '/home/jiri/AAA/t2-fwd-center-vs-6',
                    '/home/jiri/AAA/t2-fwd-center-vs-7',
                    '/home/jiri/AAA/t2-fwd-center-vs-14',
                    '/home/jiri/AAA/t2-fwd-center-vs-15',
                    '/home/jiri/AAA/t2-bkwd-center-vs-0',
                    '/home/jiri/AAA/t2-fwd-corner-1',
                    '/home/jiri/AAA/t2-fwd-corner-2'
                    ]

    if args.plot_history is not None:
        trainer.plot_train_history(args.plot_history)
        return

    if args.eval_model is not None:
        print("Loading complete, not shuffeled dataset from data.p")
        with open('data.p', 'rb') as f:
            data = pickle.load(f)

        # Predicts steering angles and speed on the validation dataset, calculated RMSE
        # and plots the ground truth vs predicted data. The plot is stored in
        # ground_truth_predicted_steering_speed.pdf
        trainer.evaluate(args.eval_model, data)
        return

    # Try to load training and validation dataset from pickle files.
    # This data include image data, steering wheel angles and speed
    train_data, val_data = trainer.unpickle_datasets()

    if train_data is None or val_data is None:
        # If the pickeled dataset are not available create them
        data = data_dirs_t2+data_dirs_t1

        # Go through all individual drive datasets, create a Frame object for
        # each frame and split the dataset to train/val parts.
        train_data, val_data = trainer.assemble_dataset('', data)

        # The train and validation datasets are stored in pickle files
        # train_images.p and val_images.p
        # If store_image_data=True the datasets will also include the preprocessed image data
        trainer.pickle_datasets(train_data, val_data, store_image_data=args.store_images)


    if args.plot_steering_histogram:
        # Plots a histogram of the steering wheel angles and stores
        # the plot in steering_angles_histogram.pdf
        trainer.show_steering_histogram(train_data)
        return


    trainer.train(train_data, val_data, tune_weights_file=args.model_weights)


if __name__ == '__main__':
    main()
