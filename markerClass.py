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
from kivy.graphics.texture import Texture
from kivy.uix.scatter import Scatter
from kivy.properties import StringProperty, ListProperty
import paramiko
from kivy.graphics import Color, Rectangle, Point, GraphicException, Line
from random import random
import socket
import shutil
from yattag import Doc, indent

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
    PATH_DATA = '/home/lapiscoatlanta03/Datasets'
    imagesPath = ''
    imagePath = ''
    RCenter = ''
    X = []
    Y = []
    # image_center = []
    X_upper_left = 0
    Y_upper_left = 0
    X_lower_right = 0
    Y_lower_right = 0
    flag_poly = False
    pointsOfMark = []
    listOfLabelsToSelectForImage_CheckBoxs_Ids = {}
    listOfLabelsToSelectForImage_Boxes_Ids = {}
    listOfLabelsToSelectForImage_Labels_Ids = {}
    listOfLastSelectLabel_CheckBoxs_Ids={}
    listOfLastSelectLabel_Boxes_Ids={}
    listOfLastSelectLabel_Labels_Ids={}

    host_name = socket.gethostname()
    markIds = []
    markLabels = []
    labelsSelected = []
    imageShape = []
    nMarkedImages = 0
    iconsPath = []
    lastSelectedList = []

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

    def setImagesPathList(self, List):
        self.imagesPathlist = List
        print(self.imagesPathlist)

    def setLastSelectedList(self, label):

        if label in self.lastSelectedList:
            print(label,self.lastSelectedList)

        else:

            if len(self.lastSelectedList)>= 5:
                print(self.lastSelectedList.pop(0))
            self.lastSelectedList.append(label)

            for idxRM, wgtRM in self.listOfLastSelectLabel_Labels_Ids.items():
                # Precisa Remover a BOXLAYOUT, não somente o checkbox
                self.ids.boxLastClassSelectedThumbnailId.remove_widget(wgtRM)
            for idxRM, wgtRM in self.listOfLastSelectLabel_CheckBoxs_Ids.items():
                # Precisa Remover a BOXLAYOUT, não somente o checkbox
                self.ids.boxLastClassSelectedThumbnailId.remove_widget(wgtRM)
            for idxRM, wgtRM in self.listOfLastSelectLabel_Boxes_Ids.items():
                # Precisa Remover a BOXLAYOUT, não somente o checkbox
                self.ids.boxLastClassSelectedThumbnailId.remove_widget(wgtRM)

            labelIconFlag = 0
            for label in self.lastSelectedList:
                self.manager.get_screen('plotMarkerScreenName').setRegionColor(label)
                newBox = BoxLayout()
                for s in self.iconsPath:
                    if label in s:
                        labelIconFlag = 1
                        try:

                            rt = s.replace(self.imagePath + '/icons/', "")

                            newLabel = kIm.Image(source='Label_Icons/' + rt)
                            newBox.add_widget(newLabel)

                            break

                        except ValueError as varerr:
                            print('ERRO ', varerr)
                            newLabel = Label()
                            newLabel.text = label
                            newLabel.text_size = (self.width, None)
                            newLabel.halign = 'center'
                            newLabel.valign = 'middle'
                            newBox.add_widget(newLabel)
                            break

                if labelIconFlag != 1:
                    newLabel = Label()
                    newLabel.text = label
                    newLabel.text_size = (self.width, None)
                    newLabel.halign = 'center'
                    newLabel.valign = 'middle'
                    newBox.add_widget(newLabel)
                labelIconFlag = 0
                newCheckBox = CheckBox()
                newCheckBox.group = 'AvailableCheckbox'
                newCheckBox.color = (1, 1, 1, 1)
                newBox.size_hint = (1, None)
                self.listOfLastSelectLabel_CheckBoxs_Ids["CheckBox_" + str(label)] = newCheckBox;
                self.listOfLastSelectLabel_Boxes_Ids["Box_" + str(label)] = newBox;
                self.listOfLastSelectLabel_Labels_Ids["Label_" + str(label)] = newLabel;

                newBox.add_widget(newCheckBox)
                self.ids.boxLastClassSelectedThumbnailId.add_widget(newBox)



    def setLabelsSelected(self, listLabels):
        self.labelsSelected = np.copy(listLabels)

    def get_selected_labels(self):
        sftp_client = MarkerInitialScreen.ssh_client.open_sftp()

        self.imageInMarker = self.imagesPathlist[0].replace(self.imagePath + '/', "")
        self.nMarkedImages += 1
        self.ids.imageInMarkerNameId.text = '/'.join(map(str, self.imageInMarker.split('/')[1:])) + '    ' + str(self.nMarkedImages) + 'ª de ' + str(
            len(self.imagesPathlist))
        self.manager.get_screen('plotMarkerScreenName').setImageInMarker(self.imageInMarker)
        try:
            os.makedirs('./temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])))
            MarkerInitialScreen.ssh_client.exec_command(
                "mkdir -p " + self.imagePath + '/.temp/lapisco_' + self.host_name)
        except:
            MarkerInitialScreen.ssh_client.exec_command(
                "mkdir -p " + self.imagePath + '/.temp/lapisco_' + self.host_name)
        flag = True
        while flag:
            try:
                sftp_client.get(self.imagesPathlist[0], './temp/' + self.imageInMarker)
                try:
                    MarkerInitialScreen.ssh_client.exec_command(
                        "mkdir -p " + self.imagePath + '/.temp/lapisco_' + self.host_name)
                    sftp_client.rename(self.imagesPathlist[0],
                                       self.imagePath + '/.temp/lapisco_' + self.host_name + '/' +
                                       self.imageInMarker.split('/')[-1])
                except:
                    sftp_client.rename(self.imagesPathlist[0],
                                       self.imagePath + '/.temp/lapisco_' + self.host_name + '/' +
                                       self.imageInMarker.split('/')[-1])
                flag = False
            except:
                os.remove('./temp/' + self.imageInMarker)
                self.imagesPathlist.pop(0)
                self.imageInMarker = self.imagesPathlist[0].replace(self.imagePath + '/', "")
                self.nMarkedImages += 1
                self.ids.imageInMarkerNameId.text = '/'.join(
                    map(str, self.imageInMarker.split('/')[1:])) + '    ' + str(self.nMarkedImages) + 'ª de ' + str(
                    len(self.imagesPathlist))
                self.manager.get_screen('plotMarkerScreenName').setImageInMarker(self.imageInMarker)
                flag = True

        with sftp_client.open(self.imagePath + '/.temp/lapisco_' + self.host_name + '/' + self.imageInMarker.split('/')[-1]) as f:
            img = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)

        self.imageShape = img.shape

        self.ids.imageViewerID.source = 'temp/' + self.imageInMarker
        self.manager.get_screen('plotMarkerScreenName').setImage(img)

        for idxRM, wgtRM in self.listOfLabelsToSelectForImage_CheckBoxs_Ids.items():
            # Precisa Remover a BOXLAYOUT, não somente o checkbox
            self.ids.boxLayoutLabelsToSelectForImageId.remove_widget(wgtRM)
        for idxRM, wgtRM in self.listOfLabelsToSelectForImage_Labels_Ids.items():
            # Precisa Remover a BOXLAYOUT, não somente o checkbox
            self.ids.boxLayoutLabelsToSelectForImageId.remove_widget(wgtRM)
        for idxRM, wgtRM in self.listOfLabelsToSelectForImage_Boxes_Ids.items():
            # Precisa Remover a BOXLAYOUT, não somente o checkbox
            self.ids.boxLayoutLabelsToSelectForImageId.remove_widget(wgtRM)

        stdin_F2, stdout_F2, stderr_F2 = MarkerInitialScreen.ssh_client.exec_command("find " + self.imagePath + '/icons'+" -type f")
        foldersSftp = stdout_F2.readlines()
        for iconPathIm in foldersSftp:
            if any(ext in iconPathIm for ext in ('.jpg', '.png', '.JPG', '.PNG')):
                self.iconsPath.append(iconPathIm.replace("\n", ""))
                rtN = iconPathIm.replace("\n", "")
                rt = rtN.replace(self.imagePath + '/icons/', "")
                print(iconPathIm.replace("\n", ""),rt)
                sftp_client.get(iconPathIm.replace("\n", ""), 'Label_Icons/' + rt)


        def update_height(img, *args):
            img.height = img.width / img.image_ratio

        labelIconFlag=0
        for label in self.labelsSelected:
            self.manager.get_screen('plotMarkerScreenName').setRegionColor(label)
            newBox = BoxLayout()
            for s in self.iconsPath:
                if label in s:
                    labelIconFlag = 1
                    try:

                        rt = s.replace(self.imagePath + '/icons/', "")

                        newLabel = kIm.Image(source = 'Label_Icons/'+rt  )
                        newBox.add_widget(newLabel)

                        break

                    except ValueError as varerr:
                        print('ERRO ', varerr)
                        newLabel = Label()
                        newLabel.text = label
                        newLabel.text_size = (self.width, None)
                        newLabel.halign = 'center'
                        newLabel.valign = 'middle'
                        newBox.add_widget(newLabel)
                        break

            if labelIconFlag != 1:
                newLabel = Label()
                newLabel.text = label
                newLabel.text_size = (self.width, None)
                newLabel.halign = 'center'
                newLabel.valign = 'middle'
                newBox.add_widget(newLabel)
            labelIconFlag = 0

            newCheckBox = CheckBox()
            self.listOfLabelsToSelectForImage_CheckBoxs_Ids["CheckBox_" + str(label)] = newCheckBox;
            self.listOfLabelsToSelectForImage_Boxes_Ids["Box_" + str(label)] = newBox;
            self.listOfLabelsToSelectForImage_Labels_Ids["Label_"+str(label)] = newLabel;

            newCheckBox.group = 'AvailableCheckbox'
            newCheckBox.color = (1, 1, 1, 1)
            newBox.size_hint = (1, None)

            newBox.add_widget(newCheckBox)

            self.ids.boxLayoutLabelsToSelectForImageId.add_widget(newBox)

    def createXMLfile(self, filename, width, height, depth, labels = [], Xs_upper_left = [], Ys_upper_left = [], Xs_lower_right = [], Ys_lower_right = []):
        doc, tag, text = Doc().tagtext()

        with tag('annotation'):
            with tag('folder'):
                text('OXIIIT')
            with tag('filename'):
                text(str(filename))

            with tag('source'):
                with tag('database'):
                    text('DATABASE')
                with tag('annotation'):
                    text('OXIIIT')
                with tag('image'):
                    text('flickr')

            with tag('size'):
                with tag('width'):
                    text(str(width))
                with tag('height'):
                    text(str(height))
                with tag('depth'):
                    text(str(depth))

            with tag('segmented'):
                text('0')

            for label, X_upper_left, Y_upper_left, X_lower_right, Y_lower_right in zip(labels, Xs_upper_left, Ys_upper_left, Xs_lower_right, Ys_lower_right):
                with tag('object'):
                    with tag('name'):
                        text(label)

                    with tag('pose'):
                        text('Frontal')

                    with tag('truncated'):
                        text('0')

                    with tag('occluded'):
                        text('0')

                    with tag('bndbox'):
                        with tag('xmin'):
                            text(str(X_upper_left))

                        with tag('ymin'):
                            text(str(Y_upper_left))

                        with tag('xmax'):
                            text(str(X_lower_right))

                        with tag('ymax'):
                            text(str(Y_lower_right))

                    with tag('difficult'):
                        text('0')

        return indent(doc.getvalue(), indentation=' ' * 2, newline='\r\n')

    def setPointsOfMark(self, Points):
        Points_x = [Points[i] + self.RCenter[0] for i in range(0, len(Points), 2)]
        Points_y = [Points[j] + self.RCenter[1] for j in range(1, len(Points), 2)]
        self.pointsOfMark = [[Points_x[k], Points_y[k]] for k in range(len(Points_x))]

    def setRegions(self, Regions):
        self.regions = Regions

    def backToMarkerLabel(self):
        self.regions = []
        self.ids.imageViewerID.source = 'temp/' + self.imageInMarker
        filelist = os.listdir('./temp/')
        for file in filelist:
            try:
                os.remove('./temp/' + file)
            except:
                shutil.rmtree('./temp/' + file)
        self.manager.current = 'markerLabelScreenName'
        self.manager.get_screen('plotMarkerScreenName').resetRegions()

    def setFlagPoly(self, Flag):
        self.flag_poly = Flag

    def setDiffCenter(self, Center):
        self.RCenter = Center

    def setImagesPath(self, Path):
        self.imagePath = Path

    def activeNextButton(self):
        self.ids.buttonToNextID.disabled = False

    def showMarkes(self, X_min = 0, X_max = 0, Y_min = 0, Y_max = 0, color = 0):
        if self.flag_poly:
            self.childrenImageViewerId = self.ids.imageViewerID.canvas.children
            self.markIds.append(str(random()))
            X_min += self.RCenter[0];
            X_max += self.RCenter[0]
            Y_min += self.RCenter[1];
            Y_max += self.RCenter[1]
            with self.ids.imageViewerID.canvas:
                Color(color, 1, 1, mode='rgb')
                Line(points=self.pointsOfMark, width=2)
                Line(points=(X_min, Y_max, X_max, Y_max, X_max, Y_min, X_min, Y_min), width=2, close=True)
            self.markLabels.append(Label(size_hint=(None, None)))
            self.flag_poly = False
        else:
            self.ids.imageViewerID.canvas.children = self.childrenImageViewerId[:3]


    def setImageToView(self, Image):
        self.ids.imageViewerID.source = Image
        self.ids.imageViewerID.reload()

    def marker_image(self):
        for idx, wgt in self.listOfLabelsToSelectForImage_CheckBoxs_Ids.items():
            if wgt.active:
                image_label = idx.replace("CheckBox_", "")
                self.manager.get_screen('plotMarkerScreenName').setRegionLabel(image_label)
                self.setLastSelectedList(image_label)

        for idx, wgt in self.listOfLastSelectLabel_CheckBoxs_Ids.items():
            if wgt.active:
                image_label = idx.replace("CheckBox_", "")
                self.manager.get_screen('plotMarkerScreenName').setRegionLabel(image_label)
                self.setLastSelectedList(image_label)

        self.manager.current = 'plotMarkerScreenName'
        self.manager.get_screen('plotMarkerScreenName').setCenterViewer([int(self.ids.imageViewerID.center_x), int(self.ids.imageViewerID.center_y)])
        self.manager.get_screen('plotMarkerScreenName').setImagesPath(self.imagePath)
        self.manager.get_screen('plotMarkerScreenName').setRegionLabel(image_label)
        self.manager.get_screen('plotMarkerScreenName').setImageToMarker(self.ids.imageViewerID.source)


    def nextImage(self):
        sftp_client = MarkerInitialScreen.ssh_client.open_sftp()
        self.flag_poly = False
        self.flag_next = True
        self.manager.get_screen('plotMarkerScreenName').setFlagNext(self.flag_next)

        try:
            with open('./temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (self.imageInMarker.split('/')[-1])[:-4] + '.txt', 'a', encoding='utf-8') as f:
                for region in self.regions:
                    f.write('{}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}\n'.format('/marcado' + str('/'.join(map(str, self.imageInMarker.split('/')[1:-1])) + '/' + self.imageInMarker.split('/')[-1]),
                                                                            str(region['label']), str(region['X_upper_left']), str(region['Y_upper_left']), str(region['X_lower_right']), str(region['Y_lower_right']),
                                                                            str(region['all_x'][:-1]), str(region['all_y'][:-1]),
                                                                            str(region['Yolo_X']), str(region['Yolo_Y']), str(region['Yolo_Width']), str(region['Yolo_Height'])))
            f.close()

            labels = [region['label'] for region in self.regions]
            Xs_upper_left = [region['X_upper_left'] for region in self.regions]
            Ys_upper_left = [region['Y_upper_left'] for region in self.regions]
            Xs_lower_right = [region['X_lower_right'] for region in self.regions]
            Ys_lower_right = [region['Y_lower_right'] for region in self.regions]

            xml = self.createXMLfile(self.imageInMarker.split('/')[-1], self.imageShape[1], self.imageShape[0], self.imageShape[2],
                                labels, Xs_upper_left, Ys_upper_left, Xs_lower_right, Ys_lower_right)

            with open('./temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (self.imageInMarker.split('/')[-1])[:-4] + '.xml', 'w') as file:
                file.write(xml)
            file.close()

            self.regions = []
            self.manager.get_screen('plotMarkerScreenName').resetRegions()

        except:
            print('Not there files!')

        try:
            MarkerInitialScreen.ssh_client.exec_command("mkdir -p " + self.imagePath + '/.temp/lapisco_' + self.host_name)
            sftp_client.put('temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (self.imageInMarker.split('/')[-1])[:-4] + '.txt',
                            self.imagePath + '/.temp/lapisco_' + self.host_name + '/' + (self.imageInMarker.split('/')[-1])[:-4] + '.txt')
        except:
            sftp_client.put('temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (
                                                                                                     self.imageInMarker.split(
                                                                                                         '/')[-1])[
                                                                                                     :-4] + '.txt',
                            self.imagePath + '/.temp/lapisco_' + self.host_name + '/' + (self.imageInMarker.split(
                                '/')[-1])[:-4] + '.txt')

        try:
            print('temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (
                                                                                               self.imageInMarker.split(
                                                                                                   '/')[
                                                                                                   -1])[
                                                                                           :-4] + '.xml')
            print(self.imagePath + '/marcado/xmls/' + (self.imageInMarker.split('/')[-1])[:-4] + '.xml')
            MarkerInitialScreen.ssh_client.exec_command("mkdir -p " + self.imagePath + '/marcado/xmls')
            sftp_client.put('temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (
                                                                                                     self.imageInMarker.split(
                                                                                                         '/')[-1])[
                                                                                                     :-4] + '.xml',
                            self.imagePath + '/marcado/xmls/' + (self.imageInMarker.split('/')[-1])[:-4] + '.xml')
        except:
            print('temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (
                                                                                                         self.imageInMarker.split(
                                                                                                             '/')[
                                                                                                             -1])[
                                                                                                     :-4] + '.xml')
            print(self.imagePath + '/marcado/xmls/' + (self.imageInMarker.split('/')[-1])[:-4] + '.xml')
            sftp_client.put('temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (
                                                                                                         self.imageInMarker.split(
                                                                                                             '/')[
                                                                                                             -1])[
                                                                                                     :-4] + '.xml',
                            self.imagePath + '/marcado/xmls/' + (self.imageInMarker.split('/')[-1])[:-4] + '.xml')

        try:
            MarkerInitialScreen.ssh_client.exec_command("mkdir -p " + self.imagePath + '/marcado/masks')
            sftp_client.put('temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (
                                                                                                     self.imageInMarker.split(
                                                                                                         '/')[-1])[
                                                                                                     :-4] + '.png',
                            self.imagePath + '/marcado/masks/' + (self.imageInMarker.split('/')[-1])[:-4] + '.png')
        except:
            sftp_client.put('temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])) + '/' + (
                                                                                                         self.imageInMarker.split(
                                                                                                             '/')[
                                                                                                             -1])[
                                                                                                     :-4] + '.png',
                            self.imagePath + '/marcado/masks/' + (self.imageInMarker.split('/')[-1])[:-4] + '.png')

        try:
            MarkerInitialScreen.ssh_client.exec_command("mkdir -p " + self.imagePath + '/marcado/' + '/'.join(map(str, self.imageInMarker.split('/')[1:-1])))
            sftp_client.rename(self.imagePath + '/.temp/lapisco_' + self.host_name + '/' + self.imageInMarker.split('/')[-1], self.imagePath + '/marcado/' + '/'.join(
                map(str, self.imageInMarker.split('/')[1:-1])) + '/' + self.imageInMarker.split('/')[-1])

        except:
            sftp_client.rename(self.imagePath + '/.temp/lapisco_' + self.host_name + '/' + self.imageInMarker.split('/')[-1], self.imagePath + '/marcado/' + '/'.join(
                map(str, self.imageInMarker.split('/')[1:-1])) + '/' + self.imageInMarker.split('/')[-1])

        filelist = os.listdir('./temp/')

        for file in filelist:
            try:
                os.remove('./temp/' + file)
            except:
                shutil.rmtree('./temp/' + file)


        self.imagesPathlist.pop(0)
        if len(self.imagesPathlist) > 0:
            self.imageInMarker = self.imagesPathlist[0].replace(self.imagePath + '/', "")
            self.nMarkedImages += 1
            self.ids.imageInMarkerNameId.text = '/'.join(map(str, self.imageInMarker.split('/')[1:])) + '    ' + str(
                self.nMarkedImages) + 'ª de ' + str(
                len(self.imagesPathlist))
            self.manager.get_screen('plotMarkerScreenName').setImageInMarker(self.imageInMarker)

            try:
                os.makedirs('./temp/' + '/'.join(map(str, self.imageInMarker.split('/')[:-1])))
                MarkerInitialScreen.ssh_client.exec_command(
                    "mkdir -p " + self.imagePath + '/.temp/lapisco_' + self.host_name)
            except:
                MarkerInitialScreen.ssh_client.exec_command(
                    "mkdir -p " + self.imagePath + '/.temp/lapisco_' + self.host_name)
            flag = True
            while flag:
                try:
                    sftp_client.get(self.imagesPathlist[0], './temp/' + self.imageInMarker)
                    try:
                        MarkerInitialScreen.ssh_client.exec_command(
                            "mkdir -p " + self.imagePath + '/.temp/lapisco_' + self.host_name)
                        sftp_client.rename(self.imagesPathlist[0],
                                           self.imagePath + '/.temp/lapisco_' + self.host_name + '/' +
                                           self.imageInMarker.split('/')[-1])
                    except:
                        sftp_client.rename(self.imagesPathlist[0],
                                           self.imagePath + '/.temp/lapisco_' + self.host_name + '/' +
                                           self.imageInMarker.split('/')[-1])
                    flag = False
                except:
                    os.remove('./temp/' + self.imageInMarker)
                    self.imagesPathlist.pop(0)
                    self.imageInMarker = self.imagesPathlist[0].replace(self.imagePath + '/', "")
                    self.nMarkedImages += 1
                    self.ids.imageInMarkerNameId.text = '/'.join(
                        map(str, self.imageInMarker.split('/')[1:])) + '    ' + str(self.nMarkedImages) + 'ª de ' + str(
                        len(self.imagesPathlist))
                    self.manager.get_screen('plotMarkerScreenName').setImageInMarker(self.imageInMarker)
                    flag = True

            with sftp_client.open(
                    self.imagePath + '/.temp/lapisco_' + self.host_name + '/' + self.imageInMarker.split('/')[-1]) as f:
                img = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)


            self.ids.imageViewerID.source = 'temp/' + self.imageInMarker
            self.manager.get_screen('plotMarkerScreenName').setImage(img)

        else:
            print('Not there more files to marker!!!')
            self.manager.current = 'mainScreenName'


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

    def touch_down(self, touch):
        if self.collide_point(*touch.pos):
            print (self.source)

