from google.protobuf import text_format
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from skimage.feature import greycomatrix, greycoprops
from kivy.properties import ObjectProperty, Clock
from skimage.feature import local_binary_pattern
from kivy.uix.progressbar import ProgressBar
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from object_detection.utils import config_util
from stat import S_ISDIR

import kivy.uix.image as kIm
import numpy as np
import cv2
import os
import time, threading
import pandas as pd
import config
import tfannotation
from markerInitialClass import *
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import slim

from markerLabelClass import *
from kivy.graphics.texture import Texture
from kivy.uix.scatter import Scatter
from kivy.properties import StringProperty
import paramiko


selectClassDropDownID = 0

listOfCheckBoxId = []


class Picture(Scatter):
    '''Picture is the class that will show the image with a white border and a
    shadow. They are nothing here because almost everything is inside the
    picture.kv. Check the rule named <Picture> inside the file, and you'll see
    how the Picture() is really constructed and used.

    The source property will be the filename to show.
    '''

    source = StringProperty(None)

class LoadDialog(FloatLayout):
        load = ObjectProperty(None)
        cancel = ObjectProperty(None)


class MarkerScreen(Screen):
    imagesPath = ''

    def __init__(self, **kwa):
            super(MarkerScreen, self).__init__(**kwa)

            self.pathText = []
            self.progressBarTime = 0;
            self.progressLen=0;
            self.box = BoxLayout()
            self.box.add_widget(kIm.Image(source='kvFiles/appImages/gifs/heimdall_OMG.gif'))
            self.popup = Popup(
                    title='Por Odin!',
                    content=self.box,
                    size_hint=(0.4, 0.40)
            )

            # self.popup.bind(on_open=self.puopen)

    def textInputActivate(self):
            if (os.path.isdir(self.ids.text_input.text)):
                    self.ids.boxLayoutFineTuningScreenSelectId.disabled = False;
                    self.ids.boxLayoutFineTuningScreenRunId.disabled = False;
                    self.ids.boxLayoutSCM.disabled = True;
                    self.ids.checkbox_HU.disabled = True;
                    self.ids.checkbox_GLCM.disabled = True;
                    self.ids.checkbox_LBP.disabled = True;

            else:
                    self.ids.text_input.text = 'Invalid Folder';
                    self.ids.boxLayoutFineTuningScreenSelectId.disabled = True;
                    self.ids.boxLayoutFineTuningScreenRunId.disabled = True;

    def disable_input(self, checkbox1, checked):
            self.ids.text_input.disabled = not checked






    def setPath(self,textIn):
        self.imagesPath = textIn


    def waitTime(self):
            time.sleep(3)
            self.popup.dismiss()

    def next(self, dt):
            if self.progress_bar.value >= 100:
                    return False
            self.progress_bar.value = self.progressLen

    def puopen(self, instance):
            Clock.schedule_interval(self.next, 1 / 25)

