#Importing dependencies

#Dpendencies for kivy
from kivy.config import Config
from kivy.core.text import LabelBase
from kivy.utils import get_color_from_hex
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.slider import Slider
from kivy.uix.dropdown import DropDown

#OpenCV2
import cv2

#Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import face_recognition

#Miscellaneous
import ctypes
import pickle
import time
import numpy as np
import os
from winotify import Notification, audio

# Load the icon file
Config.set('kivy', 'window_icon', 'data\kivy\icon.ico')

# Register a custom font
LabelBase.register(name='Roboto', fn_regular=os.path.join('data', 'font', 'Roboto-Regular.ttf'))

#Defining class for the application
class FaceApp(App):
    #The build function is the __init__ function for kivy based applications
    def build(self):
        # Layout Configuration
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Live Feed Configuration
        self.livefeed = Image(size_hint=(1, 0.7))

        # Button and Label Configuration
        self.verifybutton = Button(text="Start Verification", on_press=self.start_face_detection, size_hint=(1, 0.05),
                                   background_color=get_color_from_hex('#007ACC'), font_name='Roboto', font_size=18)
        self.stopverifybutton = Button(text="Stop Verification", on_press=self.stop_face_detection, size_hint=(1, 0.05),
                                       background_color=get_color_from_hex('#007ACC'), font_name='Roboto', font_size=18)
        self.collectbutton = Button(text="Collect", on_press=self.datacollection, size_hint=(1, 0.05),
                                    background_color=get_color_from_hex('#007ACC'), font_name='Roboto', font_size=18)
        self.consoleoutput = Label(text="Console Output", size_hint=(1, 0.1), font_name='Roboto', font_size=24)
        self.counter = 0
        self.last_reset_time = time.time()
        self.last_notification_time = np.subtract(time.time(), 60)
        self.face_detector = load_model('VGG19_REV1.h5')
        self.tolerance = float(0.5)
        slider = Slider(min=0.1, max=1, value=self.tolerance, step=0.1, size_hint=(1, 0.05))
        slider.bind(value=self.on_value_change)
        self.label = Label(text=f'Tolerance: {format(self.tolerance, ".1f")}', size_hint=(1, 0.03))
        self.dropdown = DropDown()
        btn_lock = Button(text='Lock System', size_hint_y=None, height=44)
        btn_lock.bind(on_release=lambda btn: self.dropdown.select(btn.text))
        self.dropdown.add_widget(btn_lock)
        btn_notify = Button(text='Just Notify', size_hint_y=None, height=44)
        btn_notify.bind(on_release=lambda btn: self.dropdown.select(btn.text))
        self.dropdown.add_widget(btn_notify)
        self.mainbutton = Button(text='Select Action', size_hint=(1, None), height=44)  # Set size_hint_y=None and specify a height
        self.mainbutton.bind(on_release=self.dropdown.open)
        self.dropdown.bind(on_select=lambda instance, x: setattr(self.mainbutton, 'text', x))
        self.bboxcolour = (255, 0, 0)
        #Cheking to see if the files structure is correct and building them if necessary
        try:
            with open(os.path.join('data', 'encodings', 'face_encodings.pickle'), 'rb') as openfile:
                self.known_faces = pickle.load(openfile)
        except FileNotFoundError:
            self.consoleoutput.text = 'face encodings not found, please run collect first when opening'

        if not os.path.exists(os.path.join('data', 'encodings')):
            os.makedirs(os.path.join('data', 'encodings'))

        if not os.path.exists(os.path.join('data', 'collected_images')):
            os.makedirs(os.path.join('data', 'collected_images'))
        #Defining the order and orientation of the kivy widgets
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.livefeed)
        layout.add_widget(self.consoleoutput)
        layout.add_widget(slider)
        layout.add_widget(self.label)
        layout.add_widget(self.verifybutton)
        layout.add_widget(self.stopverifybutton)
        layout.add_widget(self.collectbutton)
        layout.add_widget(self.mainbutton)
        #Telling OpenCV to flip the camera output along the vertical axis and use a 720,720 resolution
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.face_detection_running = False
        Clock.schedule_interval(self.face_detection, 1.0 / 33.0)
        return layout
    #Defining the functions of the buttons
    def start_face_detection(self, *args):
        self.face_detection_running = True
        self.verifybutton.background_color = get_color_from_hex('#7BE495')

    def stop_face_detection(self, *args):
        self.face_detection_running = False
        self.consoleoutput.text = 'Face Detection Stopped'
        self.verifybutton.background_color = get_color_from_hex('#007ACC')

    def on_value_change(self, instance, value):
        self.tolerance = value
        self.label.text = f'Tolerance: {format(self.tolerance, ".1f")}'

    def face_detection(self, *args):
        if not self.face_detection_running:
            return
        
        ret, frame = self.capture.read() #Telling OpenCV to access the webcam
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resized = tf.image.resize(rgb, (224, 224)) #Using a built in tensorflow function to resize the image to fit the input of the CNN, it is resized and not cropped so the information about the geometry is still there
        
        yhat = self.face_detector.predict(np.expand_dims(np.divide(resized, 255), 0)) #Pretty proud of this code, this is to normalise the values of each of the pixel valus from 0-255 to 0-1 for use in the CNN
        
        xmin, ymin, xmax, ymax = yhat[1][0] #Utlilising unpacking to save the bounding box coordinates into their respective variables
        
        abs_xmin, abs_ymin, abs_xmax, abs_ymax = np.multiply([xmin, ymin, xmax, ymax], 720).astype(int) #Converting the normalised coordinates back into pixel coordinates
        #Drawing a bounding box if the confidence is greater than 0.5 (50%)
        if yhat[0] > 0.5:
            #Some manipulation of the bounding box
            cv2.rectangle(frame, (abs_xmin - 20, abs_ymin - 20), (abs_xmax - 50, abs_ymax - 50), self.bboxcolour, 2)
            cv2.circle(frame, (abs_xmin - 20, abs_ymin - 20), radius=2, color=(0, 0, 255), thickness=2)
            cv2.circle(frame, (abs_xmax - 50, abs_ymax - 50), radius=2, color=(0, 0, 255), thickness=2)
            self.consoleoutput.text = 'Face Detected'
            self.bboxcolour = (255, 0, 0)
            #If no faces are recognised the function will return nothing, then all indexes will be out of range and the application will fail, to prevent this a try/except block is used
            try:
                
                cropped_frame = frame[abs_ymin - 20:abs_ymax - 50, abs_xmin - 20:abs_xmax - 50] #This is cool, using indexing to crop the webcam feed be only the face and saving it as another variable for use later
                
                rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB) #Manipulatin of colour and OpenCV uses BGR format
                unknown_face_encoding = face_recognition.face_encodings(rgb_cropped_frame)
                #Creating a flag to represent the state of the system for use later
                flag = False
                #Looping through the collection of face encodings of a presaved known face
                for known_face in self.known_faces:
                    #Comparing faces
                    results = face_recognition.compare_faces(known_face, unknown_face_encoding,
                                                             tolerance=self.tolerance)
                    #If an authorised face is deteced
                    if results[0] == True:
                        counter = 0
                        #set the flag to true and print it in the console for debuggin purposes
                        flag = True
                        print(flag)
                        self.consoleoutput.text = 'Face Detected | Authorised'
                        #Set bounding box colour to green for it to be intuitive
                        self.bboxcolour = (0, 255, 0)
                        print(counter)
                        break
                #If an unauthorised face is deteced
                if flag == False:
                    #Add one to the counter of failed attempts
                    self.counter += 1
                    print(flag)
                    self.consoleoutput.text = 'Face Detected | Not Authorised'
                    #Set the bounding box to red
                    self.bboxcolour = (0, 0, 255)
                    print(counter)
            except:
                #This took a long time to figure out, pass ensures that the next bit of the code still runs in case the loop is paused mid way
                pass
            #Defining what to do if there are three failed attempts within 1 minute
            if self.counter == 3:
                print('fail limit reached')
                #Resetting the counter so that an authorised user can log in and still use the system
                self.counter = 0
                if self.mainbutton.text == 'Lock System':
                    #Using ctypes to send a lock command
                    ctypes.windll.user32.LockWorkStation()
                elif self.mainbutton.text == "Just Notify":
                    current_time = time.time()
                    print(f"Time since last notification: {current_time - self.last_notification_time}")
                    #Sending notification only if the last one was sent over a minute ago
                    if current_time - self.last_notification_time > 60:
                        print("Sending notification...")
                        toast = Notification(app_id='Face Recognition', title='Unauthorized Access', msg='You are not authorized to access this system', duration='long', icon=r'C:\Users\Administrator\Documents\Code_Files\Facial_Recognition_App\data\kivy\icon.ico')
                        toast.set_audio(audio.Reminder, loop=False)
                        toast.show()
                        self.last_notification_time = current_time  # Update the last notification time
                    else:
                        print("Not enough time has passed since the last notification.")

            #Resetting the time currrent_time variables which governs time elapsed for the current counter
            current_time = time.time()
            #This if statement resets the counter every 60 seconds
            if current_time - self.last_reset_time > 60:
                self.counter = 0
                self.last_reset_time = current_time
        #Prints no face detected if nothing at all is detected
        else:
            self.consoleoutput.text = 'No Face Detected'
        #Very hacking way to display the live camera feed, there is no native function for it in kivy, this is a texture that is mapped onto a box that gets updated as soon as another frame is received from the webcam
        buf = cv2.flip(frame, 0).tostring()
        video_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        video_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.livefeed.texture = video_texture
    #Defining the function of collecting images of an authorised face and converting them into face encodings for use in facial recognition
    def datacollection(self, *args):

        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        #Defining that we would collect 9 images
        self.consoleoutput.text = 'Data Collection Started'
        for imgnum in range(9):
            ret, frame = capture.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb, (224, 224))
            yhat = self.face_detector.predict(np.expand_dims(np.divide(resized, 255), 0))
            #Utilising the face detection CNN to collect only the face to improve performance and accuracy of the face recognition model
            xmin, ymin, xmax, ymax = yhat[1][0]
            abs_xmin, abs_ymin, abs_xmax, abs_ymax = np.multiply([xmin, ymin, xmax, ymax], 720).astype(int)

            cropped_frame = frame[abs_ymin - 20:abs_ymax - 50, abs_xmin - 20:abs_xmax - 50]
            rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            #Naming the images intuitively using an f string
            image_name = os.path.join('data', 'collected_images', f'{imgnum}.jpg')
            time.sleep(1)
            cv2.imwrite(image_name, cropped_frame)
            cv2.imshow("Live Feed", cropped_frame)
            cv2.waitKey(1)
        capture.release()
        cv2.destroyAllWindows()

        known_face_encodings = []
        #Starting the loop to convert the collected images of faces into face encodings
        for i in range(9):
            my_face = face_recognition.load_image_file(os.path.join('data', 'collected_images', f'{i}.jpg'))
            my_face_encoding = face_recognition.face_encodings(my_face)
            if len(my_face_encoding) > 0:
                #I put them in a list so that I can index them in a loop instead of having to call each individual face encoding seperately
                known_face_encodings.append(my_face_encoding[0])
            else:
                print(f'No face detected in image {i}.jpg')
        #Pickling them into a binary file a sort of pseudoencryption algorithm
        with open(os.path.join('data', 'encodings', 'face_encodings.pickle'), 'wb') as f:
            pickle.dump(known_face_encodings, f)
        #Waiting 5 seconds to ensure that the encodings have been pickled
        time.sleep(5)
        #Deleting all the unnesecary files generated in the process
        for file in os.listdir(os.path.join('data', 'collected_images')):
            if os.path.isfile(os.path.join('data', 'collected_images', file)):
                os.remove(os.path.join('data', 'collected_images', file))

#Calling the main application
if __name__ == '__main__':
    FaceApp().run()
