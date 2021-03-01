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

import kivy.uix.image as kIm
import numpy as np
import cv2
import os
import time, threading
import pandas as pd
import config
import config_yolo
import tfannotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import slim
import csv

selectClassDropDownID = 0

listOfCheckBoxId = []




class LoadDialog(FloatLayout):
        load = ObjectProperty(None)
        cancel = ObjectProperty(None)


class CsvConverterYoloScreen(Screen):
        loadfile = ObjectProperty(None)
        savefile = ObjectProperty(None)
        text_input = ObjectProperty(None)

        #Colunas do CSV:
        listOf_Columns_Elements_TO_Select_Ids = {}
        listOfLabel_Select_CheckBoxs = []

        #CheckBoxes de cada classe disponivel na coluna selecionada
        listOfBox_Class_Ids = {}
        listOfCheckBox_Class = {}
        listOfLabel_Classes_CheckBoxs = []


        #Lista das Classes Selecionadas
        listOf_Selected_CheckBoxes_Class = {}
        listOf_Selected_Boxes_Class = {}


        countVerifyImages = 0;

        listOfOrder = []
        listofLabelText = ['Selecione a coluna de path', 'Selecione a coluna de classes',
                           'Selecione a coluna de X Upper Left',
                           'Selecione a coluna de Y Upper Left', 'Selecione a coluna de X Lower Right',
                           'Selecione a coluna de Y Lower Right','']
        count_listofLabelText = 0

        df = 0
        fileName = ''
        def __init__(self, **kwa):
                super(CsvConverterYoloScreen, self).__init__(**kwa)

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

        def show_load(self):
                content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
                self._popup = Popup(title="Selecione um CSV", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

        def waitTime(self):
                time.sleep(3)
                self.popup.dismiss()



        def load(self, path, filename):

                print(filename,filename[0][-4:])
                if (os.path.isdir(filename[0])):
                        self.progressLen = 0;

                        self.popup.title ='Grr Humanos! , selecione um ARQUIVO DATA! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()

                elif ( '.data' in filename[0][-5: ]):

                        self.ids.text_input.text = filename[0].split('/')[-1]
                        pathFile = '/'.join(filename[0].split('/')[1:-1])
                        # self.df = pd.read_csv(filename[0] , delimiter = ';')
                        self.fileName = filename[0]

                        with open(self.fileName) as f:
                            data = f.readlines()

                        num_classes = data[0].replace("classes = ","")
                        num_classes = int(num_classes.replace("\n",""))

                        train_path = data[1].replace("train = ","")
                        train_path = train_path.replace("\n","")

                        test_path = data[2].replace("valid = ", "")
                        test_path = test_path.replace("\n","")

                        names_path = data[3].replace("names = ","")
                        names_path = names_path.replace("\n","")

                        with open(train_path) as f:
                            data_train = f.readlines()

                        with open(test_path) as f:
                            data_test = f.readlines()

                        data_total = data_train+data_test

                        with open(names_path) as f:
                            data_names = f.readlines()

                        dictNames = {i: data_names[i] for i in range(0, len(data_names))}
                        outputFile = '/'+pathFile+'/converted_file_annotations.csv'
                        with open(outputFile, 'w') as csvfile:
                            filewriter = csv.writer(csvfile, delimiter=';')
                            filewriter.writerow(
                                ['path', 'label', 'X_upper_left', 'Y_upper_left',
                                 'X_lower_right', 'Y_lower_right','all_x','all_y',
                                 'Yolo_X','Yolo_Y','Yolo_Width','Yolo_Height'])

                            for filename in data_total:
                                Filename = "{}".format(filename)
                                Filename = Filename.replace("\n","")

                                _, fileExtension = os.path.splitext(Filename)
                                fileName_txt = Filename.replace(fileExtension,".txt")
                                f = open(fileName_txt, "r")

                                i = 0
                                image = cv2.imread(Filename)
                                height_img, width_img = image.shape[:2]
                                for word in f.read().split():
                                    if i == 0:
                                        annotation_tag = dictNames.get(int(word))
                                        annotation_tag = annotation_tag.replace('\n',"")
                                    elif i == 1:
                                        Yolo_x = word
                                        Yolo_x = float(Yolo_x)
                                    elif i == 2:
                                        Yolo_y = word
                                        Yolo_y = float(Yolo_y)
                                    elif i == 3:
                                        Yolo_width = word
                                        Yolo_width = float(Yolo_width)
                                    elif i == 4:
                                        Yolo_height = word
                                        Yolo_height = float(Yolo_height)
                                    i = i + 1

                                width = round(Yolo_width * width_img)
                                height = round(Yolo_height * height_img)
                                Upper_left_X = round((Yolo_x * width_img) - (width / 2))

                                Upper_left_Y = round((Yolo_y * height_img) - (height / 2))

                                Lower_right_X = round((Yolo_x * width_img) + (width / 2))

                                Lower_right_Y = round((Yolo_y * height_img) + (height / 2))

                                filewriter.writerow([Filename, annotation_tag, Upper_left_X, Upper_left_Y,
                                                     Lower_right_X,Lower_right_Y,'','',
                                                     Yolo_x,Yolo_y,Yolo_width,Yolo_height])

                        listOfCheckBox_Ids = {}
                        self.ids.labelFineTuningScreenLabelId.text = 'CSV salvo em: '+ str(outputFile)
                else:
                        self.progressLen = 0;

                        self.popup.title = 'Por Odin! Eu pedi um .DATA! '
                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()

                self.dismiss_popup()

        def next(self, dt):
                if self.progress_bar.value >= 100:
                        return False
                self.progress_bar.value = self.progressLen

        def run(self):

            for idxRM, wgtRM in self.listOfBox_Class_Ids.items():
                self.ids.boxLayoutFineTuningScreenRunCheckBoxesId.remove_widget(wgtRM)

            # Gera lista de classes no campo 3
            for idx, wgt in self.listOf_Columns_Elements_TO_Select_Ids.items():

                if wgt.active:
                    self.listOfOrder.append(str(idx).replace("CheckBox_", ""))

                    for i in self.listOfOrder:
                        newBox = BoxLayout()
                        newLabel = Label()
                        self.listOfBox_Class_Ids["Box_" + str(i)] = newBox;
                        newLabel.text = str(i)
                        newBox.size_hint = (1, None)
                        newBox.add_widget(newLabel)
                        self.ids.boxLayoutFineTuningScreenRunCheckBoxesId.add_widget(newBox)

                    # self.ids.boxLayoutFineTuningScreenSelectedCheckBoxesMasterId.disabled = False;
                    break
            self.count_listofLabelText += 1
            self.ids.labelFineTuningScreenSelectId.text = self.listofLabelText[self.count_listofLabelText]
            if self.count_listofLabelText < 6:
                self.ids.boxLayoutFineTuningScreenExtractClassesId.disabled = True
            else:
                self.ids.boxLayoutFineTuningScreenExtractClassesId.disabled = False
                self.ids.boxLayoutFineTuningScreenSelectClassesId.disabled = True



        def puopen(self, instance):
                Clock.schedule_interval(self.next, 1 / 25)


        def extractClassesSelected(self):

            newdf = self.df[self.listOfOrder]
            print(newdf)
            newdf.columns = ['Filepath', 'Annotation tag','Upper left corner X', 'Upper left corner Y', 'Lower right corner X', 'Lower right corner Y']
            newdf.to_csv("converted.csv",sep=';',index=False)


        def dismiss_popup(self):
                self._popup.dismiss()
        pass
