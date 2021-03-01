from google.protobuf import text_format
from kivy.app import App
from kivy.storage.jsonstore import JsonStore
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from skimage.feature import greycomatrix, greycoprops
from kivy.properties import ObjectProperty, Clock, StringProperty
from skimage.feature import local_binary_pattern
from kivy.uix.progressbar import ProgressBar
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from object_detection.utils import config_util
from stat import S_ISDIR, S_ISREG
from os.path import join

import kivy.uix.image as kIm
import numpy as np
import cv2
import os
import time, threading
import pandas as pd
import config
import tfannotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import slim

import paramiko

# ssh_client = paramiko.SSHClient()
# ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())


selectClassDropDownID = 0

listOfCheckBoxId = []

class LoadDialog(FloatLayout):
        load = ObjectProperty(None)
        cancel = ObjectProperty(None)


class MarkerInitialScreen(Screen):
    PATH_DATA = '/home/sol2/Datasets'
    # loadfile = ObjectProperty(None)
    # savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    listOfDatasets_CheckBoxs_Ids = {}
    listOfDatasets_Boxs_Ids = {}
    username = ''
    hostname = ''
    password = ''
    connected = False
    datasets_list = []
    imagesHomePath = ''

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    data_dir = App().user_data_dir
    store = JsonStore(join(data_dir, 'storage.json'))

    sftp_client = ''
    files = []
    configFilePath = ''
    initialLabels = []
    lastCredential = ""

    login = ""

    fileName = ''

    def __init__(self, **kwa):
            super(MarkerInitialScreen, self).__init__(**kwa)

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
            self.textinputtext = StringProperty()


            try:
                self.lastCredential = MarkerInitialScreen.store.get('credentials')
                print(self.lastCredential)


            except KeyError:
                print('Failed Login')
                MarkerInitialScreen.store.put('credentials', hostname='', username='',
                                              password='')
                self.lastCredential = MarkerInitialScreen.store.get('credentials')
                username = ""
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

    def read_and_login(self):

        temp = TextInput(text=self.ids.text_input_username_hostname.text).text
        try:
            temp = temp.split('@')
        except:
            print('Invalid adrress')

        try:

            self.username = temp[0]
            self.hostname = temp[1]
            self.password = TextInput(text=self.ids.text_input_password.text).text
            del temp


            self.ssh_client.connect(hostname=self.hostname, username=self.username, password=self.password, port=2203)
            self.connected = True

            self.ids.boxLayoutConnectionStatus.text = 'Status de conexção SSH: Conectado'
            self.ids.boxLayoutViewerDatasetsId.disabled = False
            self.ids.boxLayoutAvailablesDatasetsId.disabled = False

            # Apesar de ser obvio ( Flag logo acima dizendo True, essa parte abaixo foi extraida de uma função e portanto deixada intacta)
            if self.connected:

                MarkerInitialScreen.store.put('credentials', hostname=self.hostname,username=self.username, password=self.password)

                #self.PATH_DATA = '/media/'+self.username+'/Dados/Datasets'
                self.PATH_DATA = config.PATH_DATASET 
                # self.PATH_DATA = '/home/' + self.username + '/Datasets'
                self.sftp_client = self.ssh_client.open_sftp()
                self.datatasets_list = self.sftp_client.listdir(self.PATH_DATA)
                self.datatasets_list.sort()
                self.n_files = len(self.sftp_client.listdir(self.PATH_DATA))

                for idxRM, wgtRM in self.listOfDatasets_Boxs_Ids.items():
                    self.ids.boxLayoutAvailablesDatasetsId.remove_widget(wgtRM)

                for datasetName in self.datatasets_list:
                    if '_temp' not in datasetName:
                        stdin_F, stdout_F, stderr_F = self.ssh_client.exec_command(
                            "find " + self.PATH_DATA + '/' + datasetName + '/bruto' + " -type f")
                        n_files = len([file for file in stdout_F.readlines() if
                                       any(ext in file for ext in ('.jpg', '.png', '.JPG', '.PNG'))])
                        newBox = BoxLayout()
                        newLabel = Label()
                        newCheckBox = CheckBox()
                        self.listOfDatasets_CheckBoxs_Ids["CheckBox_" + str(datasetName)] = newCheckBox;
                        self.listOfDatasets_Boxs_Ids["Box_" + str(datasetName)] = newBox
                        newLabel.text = datasetName + ' (' + str(n_files) + ')'
                        newCheckBox.group = 'AvailableCheckbox'
                        newCheckBox.color = (1, 1, 1, 1)

                        newBox.size_hint = (1, None)
                        newBox.add_widget(newLabel)
                        newBox.add_widget(newCheckBox)
                        self.ids.boxLayoutAvailablesDatasetsId.add_widget(newBox)

                # print(self.datatasets_list)
            else:
                self.ids.boxLayoutConnectionStatus.text = 'Status de conexção SSH: Favor, conectar para acessar os datasets'

        except Exception as e:
            print(e)
            # self.ids.boxLayoutViewerDatasetsId.disable = True
            self.ids.boxLayoutConnectionStatus.text = 'Status de conexão SSH: Inválidado'
            self.ids.boxLayoutViewerDatasetsId.disabled = True
            self.ids.boxLayoutAvailablesDatasetsId.disabled = True


    def get_dataset(self):

        for idx, wgt in self.listOfDatasets_CheckBoxs_Ids.items():

            if wgt.active:
                self.imagesHomePath = os.path.join(self.PATH_DATA, idx.replace("CheckBox_", ""))
                # print("Image Path ", self.imagesHomePath)
                self.manager.get_screen('markerLabelScreenName').setImagesPath(self.imagesHomePath)

                self.sftp_client = self.ssh_client.open_sftp()

                self.datatasets_list = self.sftp_client.listdir(self.imagesHomePath)

                for fileName in self.datatasets_list:
                    if ".config" in fileName:
                        self.configFilePath = os.path.join(self.imagesHomePath,fileName)
                        break


                with self.sftp_client.open(self.configFilePath) as f:
                    for k in f.readlines():
                        self.initialLabels.append(k.replace("\n",""))

                self.manager.get_screen('markerLabelScreenName').setLabels(self.initialLabels)
                self.initialLabels = []
                self.manager.get_screen('markerLabelScreenName').get_labels()
                self.manager.current = 'markerLabelScreenName'


    def waitTime(self):
            time.sleep(3)
            self.popup.dismiss()

    def next(self, dt):
            if self.progress_bar.value >= 100:
                    return False
            self.progress_bar.value = self.progressLen

    def puopen(self, instance):
            Clock.schedule_interval(self.next, 1 / 25)
