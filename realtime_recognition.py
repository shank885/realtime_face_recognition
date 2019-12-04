from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

K.set_image_data_format('channels_first')
import cv2
import json
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from triplet_loss import triplet_loss
from inception_blocks_v2 import *



def create_encoding(image, model):
    img = image[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding



def who_is_it(image_path, database, model):
    """
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ### START CODE HERE ###

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = create_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###

    if min_dist > 0.85:
        print("Not in the database.")
        print("distance", min_dist)
        identity = "Unknown"
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


def main():

    embedding_path = "./database/embeddings/embeddings.json"
    face_detector_path = "./classifiers/haarcascade_frontalface_default.xml"

    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())

    # load trained model
    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(FRmodel)

    with open(embedding_path, 'r') as infile:
        database = json.load(infile)

    #who_is_it("images/camera_0.jpg", database, FRmodel)
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    face_detector = cv2.CascadeClassifier(face_detector_path)

    print("above while")
    while True:
        # capture frame
        if video_capture.isOpened():
            ret, frame = video_capture.read()

        raw_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,
                                               scaleFactor=1.5,
                                               minNeighbors=5,
                                               minSize=(30, 30))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cropped = raw_frame[y:y+h, x:x+w]
                image = cv2.resize(cropped,
                                   (96, 96),
                                   interpolation=cv2.INTER_LINEAR)
                min_dist, identity = who_is_it(image, database, FRmodel)

                if identity == 'Unknown':
                    box_color = (0, 0, 255)
                    text_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 0)
                    text_color = (255, 0, 0)

                cv2.rectangle(frame,
                              (x, y),
                              (x+w, y+h),
                              box_color,
                              2)
                cv2.putText(frame,
                            identity,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            text_color,
                            thickness=2,
                            lineType=2)

        cv2.imshow('Realtime Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
