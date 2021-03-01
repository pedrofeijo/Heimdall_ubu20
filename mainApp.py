#!/usr/bin/python
# -*- coding: utf-8 -*-


'''
Circle Example
==============

This example exercises circle (ellipse) drawing. You should see sliders at the
top of the screen with the Kivy logo below it. The sliders control the
angle start and stop and the height and width scales. There is a button
to reset the sliders. The logo used for the circle's background image is
from the kivy/data directory. The entire example is coded in the
kv language description.
'''
import re
from kivy.app import App
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.properties import ObjectProperty, Clock
from functools import partial
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from PIL import Image
import kivy.uix.image as kIm
from kivy.graphics import *
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.progressbar import ProgressBar
from skimage.feature import local_binary_pattern

from markerInitialClass import *
from markerLabelClass import *
from fineTuningClass import *
from videotestClass import *
from fineTuningPipelineClass import *
from markerClass import *
from plotMarkerClass import *
from imagetestYoloClass import *
from videotestYoloClass import *
from csvConverterClass import *
from csvConverterYoloClass import *

from kivy.storage.jsonstore import JsonStore
import numpy as np
import time, threading
import cv2
import os
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
import tkinter as tk
from os.path import join

#root = tk.Tk()
#screen_width = root.winfo_screenwidth()
#screen_height = root.winfo_screenheight()
#print(root.winfo_screenwidth(), root.winfo_screenheight())
#Window.size = (root.winfo_screenwidth(), root.winfo_screenheight())



class MainScreen(Screen):

        pass


class ClassifyScreen(Screen):
        pass


class CnnScreen(Screen):
        pass


class Manager(ScreenManager):
        mainScreen = ObjectProperty(None)
        fineTuningScreen = ObjectProperty(None)
        fineTuningScreenPipeline = ObjectProperty(None)
        classifyScreen = ObjectProperty(None)
        cnnScreen = ObjectProperty(None)
        videotestScreen = ObjectProperty(None)
        markerInitialScreen = ObjectProperty(None)
        markerLabelScreen = ObjectProperty(None)
        markerScreen = ObjectProperty(None)
        plotMarkerScreen = ObjectProperty(None)
        imageTestYoloScreen = ObjectProperty(None)
        videoTestYoloScreen = ObjectProperty(None)
        csvConverterScreen = ObjectProperty(None)
        csvConverterYoloScreen = ObjectProperty(None)
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        print(root.winfo_screenwidth(), root.winfo_screenheight())


layout = AnchorLayout(
        anchor_x='right', anchor_y='bottom')


def load_all_kv_files(start="./kvFiles"):
        pattern = re.compile(r".*?\.kv")
        kv_files = []
        for root, dirs, files in os.walk(start):
                kv_files += [root + "/" + file_ for file_ in files if pattern.match(file_)]

        for file_ in kv_files:
                print(file_)
                Builder.load_file(file_)


class ResGenApp(App):
        title = 'Heimdall'
        App.icon  = 'lapiscoIcon.png'
        data_dir = App().user_data_dir
        print(data_dir)
        store = JsonStore(join(data_dir, 'storage.json'))
        print(App.icon)

        def build(self):

                return Manager()


load_all_kv_files();
ResGenApp().run()