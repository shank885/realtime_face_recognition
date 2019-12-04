# Realtime Face Recognition

### Platform Secification:

  * Ubuntu 18.04


### Requirements:
  
  * tensorflow==1.15.0
  * sklearn==0.21.3
  * Python==3.7.4
  * OpenCV==4.1.2
  * NumPy==1.17.2
  
  
### Setup

  * Clone this [repository](https://github.com/SHANK885/realtime_face_recognition.git)
  
  
### Enroll a new face using webcam.

  1. Go inside realtime_face_recognition directory.
  2. run "python enroll_face.py <name_of_new_member>
  3. Webcam will open up with a window to capture face.
  4. Press 's' by selectiong the video window to capture the image.
  5. If you want to recapture the image:
        select the terminal window and enter "R" or "r" else enter "C" or "c".

  It will enroll the new face with the name provided in the command line.

  
  The cropped and aligned face will be saved to:
        realtime_face_recognition/database/images/ directory
  
  The 512 D face embedding vector will be added to:
        realtime_face_recognition/database/embeddings/face_embeddings.json


### Where the image is stored ?

  * The cropped faces of all te enrolled members is stored in:
    [realtime_face_recognition/database/images/ directory](https://github.com/SHANK885/realtime_face_recognition/tree/master/database/images)
  * The embeddings of all the enrolled faces is present in:
    [realtime_face_recognition/database/embeddings/<emb> directory](https://github.com/SHANK885/realtime_face_recognition/tree/master/database/embeddings)
  

