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

selectClassDropDownID = 0

listOfCheckBoxId = []


class LoadDialog(FloatLayout):
        load = ObjectProperty(None)
        cancel = ObjectProperty(None)


class CsvConverterScreen(Screen):
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
        pathFile = ''

        def __init__(self, **kwa):
                super(CsvConverterScreen, self).__init__(**kwa)

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

                        self.popup.title ='Grr Humanos! , selecione um ARQUIVO CSV! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()

                elif ( '.csv' in filename[0][-4: ]):

                        self.ids.text_input.text = filename[0].split('/')[-1]
                        self.pathCSV = '/'.join(filename[0].split('/')[1:-1])
                        self.df = pd.read_csv(filename[0] , delimiter = ';')
                        self.fileName = filename[0]
                        # selectColumnLabel = Label()
                        self.ids.labelFineTuningScreenSelectId.text = self.listofLabelText[0]
                        # self.ids.boxLayoutFineTuningScreenSelectId.add_widget(selectColumnLabel)
                        listOfCheckBox_Ids = {}

                        for idxRM, wgtRM in self.listOf_Columns_Elements_TO_Select_Ids.items():
                            self.remove_widget(wgtRM)

                        for headerValues in self.df.columns.values:
                                print(headerValues)
                                newBox = BoxLayout()
                                newLabel = Label()
                                newCheckBox = CheckBox()
                                self.listOf_Columns_Elements_TO_Select_Ids["CheckBox_" + str(headerValues)] = newCheckBox;
                                newLabel.text = headerValues
                                newCheckBox.group = 'SelectColumnGroup'
                                newCheckBox.color = (1,1,1,1)
                                self.listOfLabel_Select_CheckBoxs.append(headerValues)

                                newBox.add_widget(newLabel)
                                newBox.add_widget(newCheckBox)


                                self.ids.boxLayoutFineTuningScreenSelectId.add_widget(newBox)

                        self.ids.boxLayoutFineTuningScreenSelectId.disabled = False;

                        self.ids.boxLayoutFineTuningScreenRunId.disabled = False;
                        self.ids.boxLayoutFineTuningScreenExtractClassesId.disabled = True




                else:
                        self.progressLen = 0;

                        self.popup.title = 'Por Odin! Eu pedi um CSV! '
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
            newdf['all_x'] = ''
            newdf['all_y'] = ''
            newdf['Yolo_X'] = ''
            newdf['Yolo_Y'] = ''
            newdf['Yolo_Width'] = ''
            newdf['Yolo_Height'] = ''
            # df['age_half'] = df.apply(lambda row: valuation_formula(row['age']), axis=1)
            newdf.columns = ['path', 'label', 'X_upper_left', 'Y_upper_left', 'X_lower_right',
                             'Y_lower_right', 'all_x', 'all_y','Yolo_X', 'Yolo_Y', 'Yolo_Width', 'Yolo_Height']
            for index in range(0,newdf.shape[0]):
                imagePath = newdf.iloc[index,0]
                try:
                    pilImage = Image.open(imagePath)
                except:
                    pilImage = Image.open('/'+self.pathCSV+'/'+imagePath)
                (w, h) = pilImage.size[:2]
                width = newdf.iloc[index,4] - newdf.iloc[index,2]
                height = newdf.iloc[index,5] - newdf.iloc[index,3]
                newdf.iloc[index,8] = (newdf.iloc[index,2] + (width / 2)) / w
                newdf.iloc[index,9] = (newdf.iloc[index,3] + (height / 2)) / h
                newdf.iloc[index,10] = abs(width / w)
                newdf.iloc[index,11] = abs(height / h)


            outputFile = '/' + self.pathCSV + '/converted_file_annotations.csv'
            newdf.to_csv(outputFile ,sep=';',index=False)
            self.ids.labelFineTuningScreenLabelId.text = 'CSV salvo em: ' + str(outputFile)


        def dismiss_popup(self):
                self._popup.dismiss()
        pass
