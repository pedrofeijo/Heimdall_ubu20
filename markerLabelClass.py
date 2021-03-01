
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

import paramiko


# imagesPath_list =  []
# selectClassDropDownID = 0

listOfCheckBoxId = []
imagePath = ''
class LoadDialog(FloatLayout):
        load = ObjectProperty(None)
        cancel = ObjectProperty(None)


class MarkerLabelScreen(Screen):
    text_input = ObjectProperty(None)
    listOfLabels_CheckBoxs_Ids = {}
    listOf_Selected_Labels_CheckBoxs_Ids = {}
    imagesPath_list = []

    Datasetlabels = []
    labels = []
    df = 0
    filesNames = []
    listOfBoxsDatasetLabels_Ids = {}
    listOfBoxsMarkerLabels_Ids = {}

    def __init__(self, **kwa):
        super(MarkerLabelScreen, self).__init__(**kwa)

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
        print(checked)
        if checked:

            for idxRM, wgtRM in self.listOfLabels_CheckBoxs_Ids.items():
                if "CheckBox_All" in idxRM:
                    continue
                else:
                    wgtRM.active = True;
        else:
            for idxRM, wgtRM in self.listOfLabels_CheckBoxs_Ids.items():
                if "CheckBox_All" in idxRM:
                    continue
                else:
                    wgtRM.active = False;


    def get_labels(self):

        for idxRM, wgtRM in self.listOfBoxsDatasetLabels_Ids.items():
            self.ids.boxLayoutLabelsAvailableId.remove_widget(wgtRM)

        newBox = BoxLayout()
        newLabel = Label()
        newCheckBox = CheckBox()
        self.listOfLabels_CheckBoxs_Ids["CheckBox_" + str("All")] = newCheckBox;
        self.listOfBoxsDatasetLabels_Ids["Box_" + str("All")] = newBox;
        newLabel.text = "All"
        newCheckBox.color = (1, 1, 1, 1)
        newBox.size_hint = (1, None)
        newBox.add_widget(newLabel)
        newBox.add_widget(newCheckBox)
        self.ids.boxLayoutLabelsAvailableId.add_widget(newBox)
        newCheckBox.bind(active=self.disable_input)

        for file in self.Datasetlabels :
            newBox = BoxLayout()
            newLabel = Label()
            newCheckBox = CheckBox()
            self.listOfLabels_CheckBoxs_Ids["CheckBox_" + str(file)] = newCheckBox;
            self.listOfBoxsDatasetLabels_Ids["Box_" + str(file)] = newBox;
            newLabel.text = file
            newLabel.text_size = (self.width, None)
            newCheckBox.color = (1, 1, 1, 1)
            newBox.size_hint = (1, None)
            newBox.add_widget(newLabel)
            newBox.add_widget(newCheckBox)

            self.ids.boxLayoutLabelsAvailableId.add_widget(newBox)


    def setImagesPath(self, Path):
        self.imagePath = Path

    def get_available_images(self):
        sftp_client = MarkerInitialScreen.ssh_client.open_sftp()
        folders_dict = {}
        # print('Home Path: ', self.imagePath)
        stdin_F, stdout_F, stderr_F = MarkerInitialScreen.ssh_client.exec_command("find " + self.imagePath + '/bruto' + " -type f")
        filesSftp = stdout_F.readlines()

        stdin_F2, stdout_F2, stderr_F2 = MarkerInitialScreen.ssh_client.exec_command("find " + self.imagePath + '/icons'+" -type f")
        foldersSftp = stdout_F2.readlines()
        temp = []
        for filePath in filesSftp:
            if any(ext in filePath for ext in ('.jpg', '.png', '.JPG', '.PNG')):
                # self.imagesPath_list.append(filePath.replace("\n", ""))
                temp.append(filePath.replace("\n", ""))
        # print(self.labels)
        self.imagesPath_list = temp
        for i, image_Path in enumerate(self.imagesPath_list):
            self.filesNames.append(image_Path.replace(self.imagePath+'/', ""))

        self.manager.current = 'markerScreenName'
        self.manager.get_screen('markerScreenName').setLabelsSelected(self.labels)
        self.manager.get_screen('markerScreenName').setImagesPath(self.imagePath)
        self.manager.get_screen('markerScreenName').setImagesPathList(self.imagesPath_list)
        self.manager.get_screen('markerScreenName').get_selected_labels()

    def setLabels(self , labelsList):
        self.Datasetlabels = np.copy(labelsList);


    def filtrar_labels(self):

        for idxRM, wgtRM in self.listOf_Selected_Labels_CheckBoxs_Ids.items():
            # Precisa Remover a BOXLAYOUT, não somente o checkbox
            self.ids.boxLayoutLabelsCheckedId.remove_widget(wgtRM)
            self.labels = []

        for idx, wgt in self.listOfLabels_CheckBoxs_Ids.items():
            if wgt.active and idx!="CheckBox_All":
                labelName = idx.replace("CheckBox_", "")
                newBox = BoxLayout()
                newLabel = Label()
                newLabel.text = labelName

                newBox.size_hint = (1, None)
                newBox.add_widget(newLabel)

                self.labels.append(labelName)
                self.listOf_Selected_Labels_CheckBoxs_Ids["Box_" + str(labelName)] = newBox;
                self.ids.boxLayoutLabelsCheckedId.add_widget(newBox)


    def waitTime(self):
            time.sleep(3)
            self.popup.dismiss()

    def next(self, dt):
            if self.progress_bar.value >= 100:
                    return False
            self.progress_bar.value = self.progressLen

    def puopen(self, instance):
            Clock.schedule_interval(self.next, 1 / 25)
