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
from markerInitialClass import *
from markerLabelClass import *
from markerClass import *
from kivy.graphics.texture import Texture
from kivy.uix.scatter import Scatter
from kivy.properties import StringProperty, ListProperty
import paramiko
from kivy.graphics import Color, Rectangle, Point, GraphicException, Line
from random import random
from kivy.core.window import Window
import tkinter as tk
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


class PlotMarkerScreen(Screen):
    pressed = []
    dist = 1000
    flag_poly = False
    X = []
    Y = []
    X_Poly = []
    Y_Poly = []
    X_upper_left = 0
    Y_upper_left = 0
    X_lower_right = 0
    Y_lower_right = 0
    element = {}
    regions = []
    colorR = {}
    colorG = {}
    colorB = {}
    root = tk.Tk()
    screen_width = 800
    screen_height = 600

    flag_next = False
    def __init__(self, **kwa):
            super(PlotMarkerScreen, self).__init__(**kwa)

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

    def disable_input(self, checkbox1, checked):
            self.ids.text_input.disabled = not checked

    def setImagesPath(self, Path):
        self.imagePath = Path

    def setCenterViewer(self, Center):
        self.imageCenterViewer = Center

    def setRegionLabel(self, Label):
        self.element['label'] = Label
        self.element['colorR'] = self.colorR[Label]
        self.element['colorG'] = self.colorG[Label]
        self.element['colorB'] = self.colorB[Label]

    def setImage(self, Image):
        self.image = Image


    def setRegionColor(self, Label):
        self.colorR[Label] = random()
        self.colorG[Label] = random()
        self.colorB[Label] = random()

    def setFlagNext(self, Flag):
        self.flag_next = Flag

    def resetRegions(self):
        self.regions = []

    def setImageInMarker(self, imagePath):
        self.imageInMarker = imagePath

    def setImageToMarker(self, imagePath ):
        self.ids.imagePlotViewerID.source = imagePath
        self.ids.imagePlotViewerID.reload()


    def on_touch_down(self, touch):
        if self.flag_poly:
            self.element['region'] = self.pressed
            self.element['all_x'] = [x for x in self.X]
            self.element['all_y'] = [int(self.ids.imagePlotViewerID.texture_size[1] - y) for y in self.Y]
            self.element['X_upper_left'] = int(self.X_upper_left)
            self.element['Y_upper_left'] = int(self.Y_upper_left)
            self.canvas.children = self.canvas.children[:3]

            self.element['X_lower_right'] = int(self.X_lower_right)
            self.element['Y_lower_right'] = int(self.Y_lower_right)
            self.element['Yolo_X'] = ((self.X_upper_left + self.X_lower_right)/2)/self.ids.imagePlotViewerID.texture_size[0]
            self.element['Yolo_Y'] = ((self.Y_upper_left + self.Y_lower_right)/2)/self.ids.imagePlotViewerID.texture_size[1]
            self.element['Yolo_Width'] = (self.X_lower_right - self.X_upper_left)/self.ids.imagePlotViewerID.texture_size[0]
            self.element['Yolo_Height'] = (self.Y_upper_left - self.Y_lower_right)/self.ids.imagePlotViewerID.texture_size[1]
            self.regions.append(self.element)
            self.manager.get_screen('markerScreenName').activeNextButton()
            self.manager.get_screen('markerScreenName').setRegions(self.regions)
            self.manager.get_screen('markerScreenName').setPointsOfMark(self.pressed)

            seg_img = np.zeros_like(self.image)

            for i, region in enumerate(self.regions):
                colorR = int(region['colorR']*255); colorG = int(region['colorG']*255); colorB = int(region['colorB']*255)
                cv2.polylines(self.image, [np.array([[x, y] for x, y in zip(region['all_x'], region['all_y'])]).reshape((-1,1,2))], True, (colorR, colorG, colorB), thickness = 3)
                cv2.fillPoly(seg_img, [np.array([[x, y] for x, y in zip(region['all_x'], region['all_y'])]).reshape((-1,1,2))], (255, 255, 255))
                cv2.rectangle(self.image, (region['X_upper_left'], region['Y_upper_left']), (region['X_lower_right'], region['Y_lower_right']), (colorR, colorG, colorB), 3)

            out_img = seg_img
            kernel = np.ones((5, 5), np.uint8)
            int_img = cv2.erode(seg_img, kernel, iterations=4)
            between_img = (out_img - int_img).astype(np.uint8)

            out_img[np.where((out_img == [0, 0, 0]).all(axis=2))] = [2, 2, 2]
            out_img[np.where((out_img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

            int_img[np.where((int_img == [255, 255, 255]).all(axis=2))] = [1, 1, 1]

            between_img[np.where((between_img == [255, 255, 255]).all(axis=2))] = [3, 3, 3]

            png_mask = (out_img + int_img + between_img).astype(np.uint8)
            normalized_img = np.zeros_like(png_mask)
            cv2.normalize(png_mask, normalized_img, 0, 255, cv2.NORM_MINMAX)

            cv2.imwrite('temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (self.imageInMarker.split('/')[-1])[:-4] + '.png', png_mask)

            cv2.imwrite('./temp/markedImage.jpg', self.image)

            self.manager.get_screen('markerScreenName').setImageToView('./temp/markedImage.jpg')

            self.pressed = []
            self.X = []
            self.Y = []
            self.X_Poly = []
            self.Y_Poly = []
            self.element = {}
            self.manager.current = 'markerScreenName'

            self.flag_poly = False

        else:
            self.rCenterX = int(self.imageCenterViewer[0] - self.ids.imagePlotViewerID.center_x)

            self.rCenterY = int(self.imageCenterViewer[1] - self.ids.imagePlotViewerID.center_y)
            self.manager.get_screen('markerScreenName').setDiffCenter([self.rCenterX, self.rCenterY])

            # self.flag_poly = False


            self.X.append(int(touch.spos[0]*self.ids.imagePlotViewerID.texture_size[0])); self.pressed.append(touch.spos[0]*self.ids.imagePlotViewerID.size[0]); self.Y.append(int(touch.spos[1]*self.ids.imagePlotViewerID.texture_size[1])); self.pressed.append(int(touch.spos[1]*self.ids.imagePlotViewerID.size[1]))

            self.X_Poly.append(int(touch.spos[0] * self.ids.imagePlotViewerID.size[0]));
            self.Y_Poly.append(int(touch.spos[1] * self.ids.imagePlotViewerID.size[1]));



            self.X_upper_left = np.min(self.X); self.Y_upper_left = self.ids.imagePlotViewerID.texture_size[1] - np.min(self.Y)
            self.X_lower_right = np.max(self.X); self.Y_lower_right = self.ids.imagePlotViewerID.texture_size[1] - np.max(self.Y)

            self.X_min = np.min(self.X); self.X_max = np.max(self.X); self.Y_min = np.min(self.Y); self.Y_max = np.max(self.Y)

            self.X_min_Poly = np.min(self.X_Poly);self.X_max_Poly = np.max(self.X_Poly);self.Y_min_Poly = np.min(self.Y_Poly); self.Y_max_Poly = np.max(self.Y_Poly)

            ud = touch.ud
            ud['group'] = g = str(touch.uid)

            # print(self.ids.imagePlotViewerID.size[0],self.ids.imagePlotViewerID.size[1],self.ids.imagePlotViewerID.texture_size)
            # print(self.screen_width, self.screen_height)
            if len(self.pressed) >= 4 and self.flag_poly == False:
                dist = np.sqrt((self.pressed[0]-self.pressed[-2])**2 + (self.pressed[1] - self.pressed[-1])**2)
                if dist < 10:
                    self.pressed[-2] = self.pressed[0]
                    self.pressed[-1] = self.pressed[1]
                    self.flag_poly = True
                    self.manager.get_screen('markerScreenName').setFlagPoly(self.flag_poly)


            with self.canvas:
                if self.flag_poly:
                    Color(self.element['colorB'], self.element['colorG'], self.element['colorR'], mode='rgb', group=g)
                    ud['lines'] = [Line(points=self.pressed, width=1),
                                   Line(points=(self.X_min_Poly, self.Y_max_Poly, self.X_max_Poly, self.Y_max_Poly, self.X_max_Poly, self.Y_min_Poly, self.X_min_Poly, self.Y_min_Poly), width=2, close=True)]
                Color(self.element['colorB'], self.element['colorG'], self.element['colorR'], mode='rgb', group=g)
                if len(self.pressed) == 2:
                    ud['lines'] = [Point(points=self.pressed, pointsize=3)]
                else:
                    ud['lines'] = [Line(points=self.pressed, width=2)]
                if(self.flag_poly == True):
                    j = self.on_touch_down(touch)


        return True

    def waitTime(self):
            time.sleep(2)
            self.popup.dismiss()