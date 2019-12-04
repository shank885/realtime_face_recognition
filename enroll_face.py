from keras import backend as K
from fr_utils import *
from inception_blocks_v2 import *
from triplet_loss import  triplet_loss
import numpy as np
import json
import cv2
import sys
import os
import argparse
K.set_image_data_format('channels_first')

def main(args):
    image_path = "./database/images/"
    embedding_path = "./database/embeddings/embeddings.json"
    face_detector_path = "./classifiers/haarcascade_frontalface_default.xml"

    image_path = os.path.join(image_path, args.name + ".png")

    video_capture = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(face_detector_path)

    print("*********Initializing Face Enrollment*************\n")


    while True:
        while True:
            if video_capture.isOpened():
                ret, frame = video_capture.read()

            raw_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray,
                                                   scaleFactor=1.5,
                                                   minNeighbors=5,
                                                   minSize=(30, 30))
            print("length of faces: ", len(faces))
            print("faces:\n", faces)
            if len(faces) == 0:
                continue
            else:
                areas = [w*h for x, y, w, h in faces]
                i_biggest = np.argmax(areas)
                bb = faces[i_biggest]

                cv2.rectangle(frame,
                              (bb[0], bb[1]),
                              (bb[0]+bb[2], bb[1]+bb[3]),
                              (0, 255, 0),
                              2)

                cropped = raw_frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
                image = cv2.resize(cropped,
                                   (96, 96),
                                   interpolation=cv2.INTER_LINEAR)

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print("Face Captured for: {}".format(args.name))
                break

        print("Press 'C' to confirm this image")
        print("Press 'R' to retake the picture")

        response = input("\nEnter Your Response: ")

        if response == "C" or response == "c":
            print("\nImage finalized\n")
            video_capture.release()
            cv2.destroyAllWindows()
            break
        if response == "R" or response == "r":
            cv2.destroyAllWindows()
            continue

    if os.path.exists(image_path):
        print("Member with name: {} already exists!!".format(args.name))
        print("Press 'C' to overwrite or 'R' to return")
        val = input("Enter response:")
        if val == 'r' or val == 'R':
            return
        elif val == 'c' or val == 'C':
            cv2.imwrite(image_path, image)
            print("image saved")
    else:
        cv2.imwrite(image_path, image)
        print("image saved _")

    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())

    # load trained model
    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(FRmodel)

    embedding = img_to_encoding(image_path, FRmodel)[0].tolist()
    print(type(embedding))
    print(embedding)
    print(len(embedding))
    print("embedding created")

    try:
        with open(embedding_path, 'r') as rf:
            base_emb = json.load(rf)
    except IOError:
        print("Embeddibg file empty!! Creating a new embedding file")
        with open(embedding_path, 'w+') as rf:
            base_emb = {}
    with open(embedding_path, 'w') as wf:
        base_emb[args.name] = embedding
        json.dump(base_emb, wf)
        print("embedding written")

    print("face enrolled with name => {}".format(args.name))


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('name',
                        type=str,
                        help='Add the name of member to be added.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
