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
        self.verifybutton = Button(text="Start Verification", on_press=self.face_detection, size_hint=(1,0.05))
        self.collectbutton = Button(text="Collect", on_press=self.datacollection, size_hint=(1,0.1))
        self.consoleoutput = Label(text="Console Output", size_hint=(1,0.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.livefeed)
        layout.add_widget(self.consoleoutput)
        layout.add_widget(self.collectbutton)
        
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        return layout

if __name__ == '__main__':
    FaceApp().run()