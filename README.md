# Emotion detection using deep learning

## Introduction

Using a pretrained deep convolutional neural networks model, this research seeks to identify the emotion on a person's face into one of **seven** categories. The model was trained using the **FER-2013** dataset from the International Conference on Machine Learning (ICML). This dataset contains 35887 grayscale, 48x48 sized facial photos of people who are furious, disgusted, afraid, pleased, neutral, sad, or astonished.

## Dependencies

* Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/)
* Install the required packages, run `pip install mediapipe`.

## Algorithm

* First, the **Mediapie** method is used to detect faces in each frame of the video feed.

* The frame area containing the face is scaled to **48x48** and fed into the pre-trained model.

* The model generates a list of **softmax scores** for each of the seven emotion classes.

* The emotion with the highest score is shown on the screen.

## Example Output

![Output](https://user-images.githubusercontent.com/39363730/167279963-b323b945-23a6-4807-90f5-230a2d8ba604.png)
