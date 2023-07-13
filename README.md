# Face Recognition App
## Overview
- This project involves the use a a trained convolutional neural network (CNN) to perform facial detection and recognition
- The idea of this project is to tune a neural network to detect faces then using the bounding box to crop and feed into a face recognition model. This lowers the signal to noise ratio.
- If you want to check out how I built the CNN check out my repository [here](https://github.com/IVIOIST/Face_Recognition_Auto_Lock)
## Getting Started
As a prerequisite, please make sure that:
- All dependencies outlined in the requirements.txt file are installed in you preferred directory and environment.
- The latest version of Git is installed
- Python 3.9+  is installed and dded to the system path
- A working GPU environment is not required but the project does work with one, for more information checck out the [tensorflow documentation](https://www.tensorflow.org/install/source)
## Initial Setup
### Cloning Repository and Installing Dependencies
#### 1. Clone the repository
``` [Terminal]
C:\> git clone https://github.com/IVIOIST/Face_Recognition_App
```
#### 2. Installing dependencies via requirements.txt
``` [Terminal]
C:\> pip install -r requirements.txt
```
## Running the Program
#### There are two ways to run the program: 
1. Open the file in VSCode and run the program by clicking on the play button
![Alt text](/readmedata/main.png?raw=true "Title")
1. Open the project directory and double click on `main.py`
## Overview of Controls
![Alt text](/readmedata/appimage.png?raw=true "Title")
#### **Console Output:** Outputs the current status of the application and other information abou the facial recognition process
#### **`Tolerance Slider`:** Allows for the tuning of the  sensitivity (tolerance) of the face recognition system. 1 is the most lenient and 0.1 is the most strict. In General you should tune this so that only you can trigger the authorised output in the console
#### **`Start Verification`:** Starts the face recognition process
#### **`Stop Verification`:** Stops the face recognition process
#### **`Collect`:** Starts the face data collection process for use in facial recognition
#### **`Select Action`:** Allows you to select what happens after 3 consecutive unauthorized faces are detected
![Alt text](/readmedata/notifyselection.png?raw=true "Title")
## Start Facial Recognition
#### 1. Press the `collect` button to start collecting your face for the system to recognize.
#### 2. This process will take a series of photos with a live feed presented to you on the screen.
![Alt text](/readmedata/collectiondemo.png?raw=true "Title")
#### 3. Please tilt your head whilst facing the camera for multiple angles to be taken.
#### 4. Once the collection process has finished, you can press the `start verfication` button to start the face recognition process and then press the `stop verfication` button to stop the face recognition.
#### 5. 3 Unauthorised faces detected within a window of 60 seconds will trigger the device to lock or a system notification depending on how you select. The counter resets every 60 secconds
## Notes

The application expects a speccific directory structure for storing enccodings and other data. Please ensure that required directories exist.