#!/usr/bin/env python3
"""
Change log:
20/02/2017 Jiri Fajtl<ok1zjf@gmail.com>
	 Added a common image preprocessing function call with the training code in model.py
	 Added a PI controller to actuate the throttle according to the actual and desired (predicted) speed.
	 Added a command line argument to add a given speed offset to the desired speed value
	 Added an option to specify a GPU/CPU device to run the prediction on.
	 Added an option to show the raw and preprocessed image from the center camera.

"""


import os

# Stops producing the tensorflow detail messages in the log
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import argparse
import base64
from datetime import datetime
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import keras.backend.tensorflow_backend as K

import cv2

# Import the Frame class from the model.py
# We will use only the Frame class to preprocess the images for the CNN input
from model import Frame

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# Let to access the command line arguments globally
args = None

# Integration buffer for the PI speed controller
ibuf=0

# CNN input data shape. It is set during initialization
input_size = []

# Create a single Frame object that we will use for image preprocessing
frame = Frame()


@sio.on('telemetry')
def telemetry(sid, data):
    global  ibuf
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image_src = Image.open(BytesIO(base64.b64decode(imgString)))
        image = frame.preprocess(np.asarray(image_src))

        image_array = np.asarray(image, dtype=np.float32)
        image_array = image_array/255.0 - 0.5

        # Run a forward pass on the CNN to predict the steering and speed
        steering_angle, target_speed = model.predict(image_array[None, :, :, :], batch_size=1)[0]

        # PI controller
        Kp = 1/30.0 * 2.5   # P gain
        Ki = 0.01           # I gain

        # Convert the normalized speed to mph
        target_speed = (target_speed+0.5) * 31.0

        # Add a given speed offset if specified on the command line
        target_speed += args.add_speed

        # Get the difference between the desired speed and the current car speed
        speed_err = target_speed - float(speed)

        # Integrate the speed errors
        ibuf+=speed_err
        ibuf = max(0, ibuf) # windup limiter

        # Calculate the throttle value and decide when to go to neutral or break.
        # Break is activated by negative throttle
        if speed_err < - 3:
            # Break
            throttle = -1
        elif speed_err < 0:
            # Neutral
            throttle = 0
        else:
            throttle = Kp*(speed_err + ibuf*Ki)

        print("sa: {:.4f}  \tt: {:.4f}  \tve: {:.4f}  \ttv: {:.4f}\tib: {:.4f}".format(steering_angle, throttle, speed_err, target_speed, ibuf))

        # Send the data to the simulator
        send_control(steering_angle, throttle)

        if args.extra_gui:
            cv2.imshow('Center camera', cv2.cvtColor(np.asarray(image_src), cv2.COLOR_RGB2BGR))
            cv2.imshow('CNN input', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image_src.save('{}.jpg'.format(image_filename))
            #cv2.imwrite('{}-cnn.jpg'.format(image_filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    parser.add_argument(
        '--add-speed',
        type=int,
        nargs='?',
        default=0,
        help='Increases/decreases speed by given amount'
    )
    parser.add_argument(
        '--device',
        type=str,
        nargs='?',
        default='/gpu:1',
        help='GPU/CPU device to use. Default is: /gpu:1'
    )
    parser.add_argument(
        '--extra-gui',
        default=False,
        action='store_true',
        help='Shows the camera and CNN input video feeds'
    )
    args = parser.parse_args()

    # Selects GPU device
    with K.tf.device(args.device):
        K.set_session(
            K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))

    model = load_model(args.model)

    # Initialize OpenCV image windows
    if args.extra_gui:
        cv2.namedWindow('Center camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('CNN input', cv2.WINDOW_NORMAL)

    # Get input data shape that our CNN model accepts
    input_size = model.layers[0].input.get_shape().as_list()[1:]
    print("CNN input shape: ",input_size)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
