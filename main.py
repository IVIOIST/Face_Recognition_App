from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import face_recognition
import ctypes
import pickle
import time
import numpy as np
import os


class FaceApp(App):

    def build(self):
        self.livefeed = Image(size_hint=(1, 0.7))
        self.verifybutton = Button(text="Start Verification", on_press=self.start_face_detection, size_hint=(1,0.05))
        self.stopverifybutton = Button(text="Stop Verification", on_press=self.stop_face_detection, size_hint=(1, 0.05))
        self.collectbutton = Button(text="Collect", on_press=self.datacollection, size_hint=(1,0.1))
        self.consoleoutput = Label(text="Console Output", size_hint=(1,0.1))
        self.counter = 0
        self.last_reset_time = time.time()
        self.face_detector = load_model('VGG19_REV1.h5')
        try:
            with open(os.path.join('temp', 'encodings', 'face_encodings.pickle'), 'rb') as openfile:
                self.known_faces = pickle.load(openfile)
        except FileNotFoundError:
            print('face encodings not found, please run collect first when opening')
        
        if not os.path.exists(os.path.join('temp', 'encodings')):
            os.makedirs(os.path.join('temp', 'encodings'))

        if not os.path.exists(os.path.join('temp', 'collected_images')):
            os.makedirs(os.path.join('temp', 'collected_images'))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.livefeed)
        layout.add_widget(self.consoleoutput)
        layout.add_widget(self.verifybutton)
        layout.add_widget(self.stopverifybutton)
        layout.add_widget(self.collectbutton)
        
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.face_detection_running = False
        Clock.schedule_interval(self.face_detection, 1.0/33.0)
        return layout
    
    def start_face_detection(self, *args):
        self.face_detection_running = True

    def stop_face_detection(self, *args):
        self.face_detection_running = False
        self.consoleoutput.text = 'Face Detection Stopped'

    def face_detection(self, *args):
        if not self.face_detection_running:
            return

        ret, frame = self.capture.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (224, 224))
        yhat = self.face_detector.predict(np.expand_dims(np.divide(resized, 255), 0))

        xmin, ymin, xmax, ymax = yhat[1][0]
        abs_xmin, abs_ymin, abs_xmax, abs_ymax = np.multiply([xmin, ymin, xmax, ymax], 720).astype(int)

        if yhat[0] > 0.5:
            cv2.rectangle(frame, (abs_xmin - 20, abs_ymin - 20), (abs_xmax - 50, abs_ymax - 50), (255,0,0), 2)
            cv2.circle(frame, (abs_xmin - 20, abs_ymin - 20), radius=2, color=(0, 0, 255), thickness=2)
            cv2.circle(frame, (abs_xmax - 50, abs_ymax - 50), radius=2, color=(0, 0, 255), thickness=2)
            self.consoleoutput.text = 'Face Detected'
            try:
                cropped_frame = frame[abs_ymin - 20:abs_ymax - 50, abs_xmin - 20:abs_xmax - 50]
                rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                unknown_face_encoding = face_recognition.face_encodings(rgb_cropped_frame)
                flag = False
                for known_face in self.known_faces:
                    results = face_recognition.compare_faces(known_face, unknown_face_encoding, tolerance=0.6)
                    if results[0] == True:  
                        counter = 0
                        flag = True
                        print(flag)
                        self.consoleoutput.text = 'Face Detected | Authorised'
                        break
                if flag == False:
                    self.counter += 1 
                    print(flag)
                    self.consoleoutput.text = 'Face Detected | Not Authorised'  
            except:
                pass
            if self.counter == 3:
                print('fail limit reached')
                self.counter = 0
                ctypes.windll.user32.LockWorkStation()
            current_time = time.time()
            if current_time - self.last_reset_time > 60:
                self.counter = 0
                self.last_reset_time = current_time
        else:
            self.consoleoutput.text = 'No Face Detected'


        buf = cv2.flip(frame, 0).tostring()
        video_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        video_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.livefeed.texture = video_texture
        

    def datacollection(self, *args):
        
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

        self.consoleoutput.text = 'Data Collection Started'
        for imgnum in range(9):
            ret, frame = capture.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb, (224, 224))
            yhat = self.face_detector.predict(np.expand_dims(np.divide(resized, 255), 0))

            xmin, ymin, xmax, ymax = yhat[1][0]
            abs_xmin, abs_ymin, abs_xmax, abs_ymax = np.multiply([xmin, ymin, xmax, ymax], 720).astype(int)

            cropped_frame = frame[abs_ymin - 20:abs_ymax - 50, abs_xmin - 20:abs_xmax - 50]
            rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

            image_name = os.path.join('temp', 'collected_images', f'{imgnum}.jpg')
            time.sleep(1)
            cv2.imwrite(image_name, cropped_frame)
            cv2.imshow("Live Feed", cropped_frame)
            cv2.waitKey(1)
        capture.release()
        cv2.destroyAllWindows()

        known_face_encodings = []

        for i in range(9):
            my_face = face_recognition.load_image_file(os.path.join('temp', 'collected_images', f'{i}.jpg'))
            my_face_encoding = face_recognition.face_encodings(my_face)
            if len(my_face_encoding) > 0:
                known_face_encodings.append(my_face_encoding[0])
            else:
                print(f'No face detected in image {i}.jpg')

        with open(os.path.join('temp', 'encodings', 'face_encodings.pickle'), 'wb') as f:
            pickle.dump(known_face_encodings, f)

        time.sleep (5)
        for file in os.listdir(os.path.join('temp', 'collected_images')):
            if os.path.isfile(os.path.join('temp', 'collected_images', file)):
                os.remove(os.path.join('temp', 'collected_images', file))
        

if __name__ == '__main__':
    FaceApp().run()
    

        