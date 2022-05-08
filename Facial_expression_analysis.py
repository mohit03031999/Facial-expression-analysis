import cv2
import mediapipe as mp
import time
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Creating a mediapipe face detector model and
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Model for Face Emotion detection
classifier =load_model(r'C:\SDSU\Spring 2022\CS549 Machine Learning\Project\model.h5')

# Labels for emotions
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Read a video file(Change the path for your own input)
cap = cv2.VideoCapture("C:\SDSU\Spring 2022\CS549 Machine Learning\Project\Video Files\\00001.mp4")

# Giving the model parameters
with mp_facedetector.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    # read until the video file is not completed
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        start = time.time()

        # Convert the BGR image to Gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        results = face_detection.process(image)

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If a face is detected
        if results.detections:
            # For each face detected in a frame
            for id, detection in enumerate(results.detections):

                # Printing the rectangle around the detected face
                mp_draw.draw_detection(image, detection)

                # Getting the face bounding box
                bBox = detection.location_data.relative_bounding_box
                h, w, c = image.shape

                # Calculating the face bounding box
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                # Getting the gray frame for the detected face
                roi_gray = gray[boundBox[1]:boundBox[1] + boundBox[3], boundBox[0]:boundBox[0] + boundBox[2]]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # Predicting the probability for all emotions on the detected face using the pre-trained model
                    prediction = classifier.predict(roi)[0]

                    # Getting the emotion for the face
                    label = emotion_labels[prediction.argmax()]

                # Adding the emotion on the frame for the output
                cv2.putText(image, f'{label}', (boundBox[0], boundBox[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        end = time.time()
        totalTime = end - start

        # Showing the output with emotion
        cv2.imshow('Face Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()