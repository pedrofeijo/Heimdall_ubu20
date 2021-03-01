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
import tkinter as tk
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

def HU_FE(fileHU, image):
        moments = cv2.moments(image.astype(np.float64))
        momentsRes = cv2.HuMoments(moments).flatten()
        momentsElement = " ".join(str(x) for x in momentsRes)
        momentsElement = momentsElement.replace(" ", ",") + '\n'
        fileHU.write(momentsElement)


def LBP_FE(fileLBP, image):
        lbp_image = local_binary_pattern(image, 59, 1, "uniform")
        lbpE = np.asarray(np.histogram(lbp_image.ravel(), bins=59)).tolist()[0]
        lbpElement = " ".join(str(x) for x in lbpE)
        lbpElement = lbpElement.replace(" ", ",") + '\n'
        fileLBP.write(lbpElement)


def GLCM_FE(fileGLCM, image):
        glcm = greycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
        xs = []
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        xs.append(greycoprops(glcm, 'correlation')[0, 0])
        xs.append(greycoprops(glcm, 'homogeneity')[0, 0])
        xs.append(greycoprops(glcm, 'ASM')[0, 0])
        xs.append(greycoprops(glcm, 'energy')[0, 0])
        xs.append(greycoprops(glcm, 'correlation')[0, 0])
        glcmElement = " ".join(str(x) for x in xs)
        glcmElement = glcmElement.replace(" ", ",") + '\n'
        fileGLCM.write(glcmElement)


def SCM_FE(fileSCM, image):
        print('SCM Fault')

class LoadDialog(FloatLayout):
        load = ObjectProperty(None)
        cancel = ObjectProperty(None)


class FineTuningScreen(Screen):
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


        df = 0
        fileName = ''
        def __init__(self, **kwa):
                super(FineTuningScreen, self).__init__(**kwa)

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


                        # self.ids.text_input.text = filename[0];

                        # self.ids.boxLayoutFineTuningScreenRunId.disabled = False;
                        # self.ids.checkboxHUBox.disabled = True;
                        # self.ids.checkboxGLCMBox.disabled = True;
                        # self.ids.checkboxSCMBox.disabled = True;
                        # self.ids.checkboxLBPBox.disabled = True;
                elif ( '.csv' in filename[0][-4: ]):

                        self.ids.text_input.text = filename[0].split('/')[-1]
                        pathCSV = '/'.join(filename[0].split('/')[1:-1])
                        self.df = pd.read_csv(filename[0] , delimiter = ';')
                        self.fileName = filename[0]
                        selectColumnLabel = Label()
                        selectColumnLabel.text = 'Selecione a Coluna de Classes'
                        self.ids.boxLayoutFineTuningScreenSelectId.add_widget(selectColumnLabel)
                        listOfCheckBox_Ids = {}
                        config.BASE_PATH = os.path.join('/',pathCSV)
                        print(config.BASE_PATH)
                        try:
                            # os.mkdir(os.path.join(config.BASE_PATH,'records'),  755);
                            os.mkdir(os.path.join(config.BASE_PATH, 'records'), 0o777);
                        except:
                            print('Path already exist? ')
                        config.ANNOT_PATH = filename[0]
                        config.TRAIN_RECORD = os.path.sep.join([config.BASE_PATH,
                                                         "records/training.record"])
                        config.TEST_RECORD = os.path.sep.join([config.BASE_PATH,
                                                        "records/testing.record"])
                        config.CLASSE_FILE = os.path.sep.join([config.BASE_PATH,
                                                        "records/classes.pbtxt"])

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

                        config_yolo.RECORDS_YOLO = os.path.join(config.BASE_PATH, 'records')

                        self.ids.boxLayoutFineTuningScreenSelectId.disabled = False;
                        self.ids.boxLayoutFineTuningScreenRunId.disabled = False;


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

        def run(self ):

            count = 0

            for idxRM, wgtRM in self.listOfBox_Class_Ids.items():
                self.ids.boxLayoutFineTuningScreenRunCheckBoxesId.remove_widget(wgtRM)

            # Gera lista de classes no campo 3
            for idx, wgt in self.listOf_Columns_Elements_TO_Select_Ids.items():

                if wgt.active:
                    print(self.listOfLabel_Select_CheckBoxs[count])
                    print(self.df[str(idx).replace("CheckBox_","")].unique())
                    for uniqueClass in self.df[str(idx).replace("CheckBox_","")].unique():
                        newBox = BoxLayout()
                        newLabel = Label()

                        newCheckBox = CheckBox()
                        self.listOfBox_Class_Ids["Box_" + str(uniqueClass)] = newBox;
                        self.listOfCheckBox_Class["CheckBox_"+str(uniqueClass)] = newCheckBox;
                        newLabel.text = str(uniqueClass)

                        newCheckBox.color = (1, 1, 1, 1)
                        self.listOfLabel_Classes_CheckBoxs.append(uniqueClass)
                        newBox.size_hint  = (1,None)
                        newBox.add_widget(newLabel)
                        newBox.add_widget(newCheckBox)

                        self.ids.boxLayoutFineTuningScreenRunCheckBoxesId.add_widget(newBox)


                    self.ids.boxLayoutFineTuningScreenSelectedCheckBoxesMasterId.disabled = False;
                    break

                count = count + 1
                # self.progress_bar.value = 1
                # self.pathText = path
                # mythread = threading.Thread(target=self.extract)
                # mythread.start()
                # self.popup.open()

        def puopen(self, instance):
                Clock.schedule_interval(self.next, 1 / 25)

        def buildConfigFiles(self):
            # open the classes output file
            f = open(config.CLASSE_FILE, "w")

            # path_file_classes = os.path.dirname(config.CLASSE_FILE)
            # file_classes_yolo = open(path_file_classes + "/obj.names", "w")
            config_yolo.OBJ_NAMES_YOLO = os.path.join(config_yolo.RECORDS_YOLO,'obj.names')
            file_classes_yolo = open(config_yolo.OBJ_NAMES_YOLO, "w")

            # loop over the classes

            count_classes = 0
            for (k, v) in config.CLASSES.items():
                # construct the class information and write to file
                item = ("item {\n"
                        "\tid: " + str(v) + "\n"
                                            "\tname: '" + k + "'\n"
                                                              "}\n")
                f.write(item)
                file_classes_yolo.write(k+'\n')
                # close the output classes file
                count_classes = count_classes+1
            f.close()
            file_classes_yolo.close()


            # initialize a data dictionary used to map each image filename
            # to all bounding boxes associated with the image, then load
            # the contents of the annotations file
            D = {}
            rows = open(config.ANNOT_PATH).read().strip().split("\n")

            for row in rows[1:]:
                # break the row into components
                row = row.split(",")[0].split(";")
                try:
                    (imagePath, label, startX, startY, endX, endY) = row
                except:
                    (imagePath, label, startX, startY, endX, endY,*_) = row

                (startX, startY) = (float(startX), float(startY))
                (endX, endY) = (float(endX), float(endY))

                # if we are not interested in the label, ignore it
                if label not in config.CLASSES:
                    continue

                # build the path to the input image, then grab any other
                # bounding boxes + labels associated with the image
                # path, labels, and bounding box lists, respectively
                p = os.path.sep.join([config.BASE_PATH, imagePath])
                b = D.get(p, [])

                # build a tuple consisting of the label and bounding box,
                # then update the list and store it in the dictionary
                b.append((label, (startX, startY, endX, endY)))
                D[p] = b

            # create training and testing splits from our data dictionary
            (trainKeys, testKeys) = train_test_split(list(D.keys()),
                                                     test_size=float(self.ids.text_input_TestProportion.text), random_state=42)




            # initialize the data split files
            datasets = [
                ("train", trainKeys, config.TRAIN_RECORD),
                ("test", testKeys, config.TEST_RECORD)

            ]
            # loop over the datasets
            for (dType, keys, outputPath) in datasets:
                # inicialize the TensorFlow wirter and initialize the total
                # number of examples written to file


                # create train and test txt files for yolo
                config_yolo.TRAIN_TEST_YOLO = os.path.dirname(outputPath)
                f = open(config_yolo.TRAIN_TEST_YOLO+"/train.txt", "w")
                f.writelines("%s\n" % item for item in trainKeys)
                f.close()
                f = open(config_yolo.TRAIN_TEST_YOLO+"/test.txt", "w")
                f.writelines("%s\n" % item for item in testKeys)
                f.close()

                print("[INFO] processing '{}'...".format(dType))
                writer = tf.python_io.TFRecordWriter(outputPath)
                total = 0

                # loop over all thhe keys in the current set
                for k in keys:
                    # load the input image from disk as a TensorFlow object
                    # print(k)
                    # input("Press Enter to continue...")
                    encoded = tf.gfile.GFile(os.path.join(config.BASE_PATH,k), "rb").read()
                    encoded = bytes(encoded)

                    # load the image from disk again, this time as a PIL
                    # object
                    pilImage = Image.open(k)
                    (w, h) = pilImage.size[:2]

                    # parse the filename and encoding from input path
                    filename = k.split(os.path.sep)[-1]
                    encoding = filename[filename.rfind(".") + 1:]

                    # initialize the annotation object used to store
                    # information regarding the bounuding box + labels
                    tfAnnot = tfannotation.TFAnnotation()
                    tfAnnot.image = encoded
                    tfAnnot.encoding = encoding
                    tfAnnot.filename = filename
                    tfAnnot.width = w
                    tfAnnot.height = h

                    # loop over the bounding boxes + labels associated with
                    # the image

                    #create .txt for each image
                    image_path_yolo = k
                    filename_yolo, file_extension_yolo = os.path.splitext(image_path_yolo)
                    file_path_yolo = image_path_yolo.replace(file_extension_yolo, ".txt")

                    # Get list of classes yolo - it will be used to get the idx
                    file_classes_yolo = open(config_yolo.OBJ_NAMES_YOLO, "r")
                    labels_yolo = file_classes_yolo.readlines()
                    file_classes_yolo.close()

                    file = open(file_path_yolo, 'w')
                    for (label, (startX, startY, endX, endY)) in D[k]:
                        # TensorFlow assumes all bounding boxes are in the
                        # range [0,1] so we need to scale them

                        xMin = startX / w
                        xMax = endX / w
                        yMin = startY / h
                        yMax = endY / h

                        #YOLO format

                        width = endX - startX
                        height = endY - startY
                        Yolo_x = (startX + (width / 2)) / w
                        Yolo_y = (startY + (height / 2)) / h
                        Yolo_width = abs(width / w)
                        Yolo_height = abs(height / h)
                        # print(k)
                        # print(D[k])
                        # print('{} {:6f} {:6f} {:6f} {:6f}\n'.format(label, Yolo_x, Yolo_y, Yolo_width, Yolo_height))
                        # input("Press Enter to continue...")

                        label_yolo_idx = labels_yolo.index(label+'\n')
                        file.write('{} {:6f} {:6f} {:6f} {:6f}\n'.format(label_yolo_idx, Yolo_x, Yolo_y, Yolo_width, Yolo_height))

                        if self.countVerifyImages < 5:
                            image = cv2.imread(k)
                            startX = int(xMin * w)
                            startY = int(yMin * h)
                            endX = int(xMax * w)
                            endY = int(yMax * h)

                            cv2.rectangle(image,(startX, startY), (endX,endY),(0,255,0),2)

                            root = tk.Tk()
                            screen_width = root.winfo_screenwidth()
                            screen_height = root.winfo_screenheight()
                            image = cv2.resize(image, (screen_width, screen_height), interpolation=cv2.INTER_AREA)
                            cv2.imshow("Image",image)
                            cv2.moveWindow("Image", 20, 20);
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            self.countVerifyImages += 1;

                        # update the bounding boxes + labels lists
                        tfAnnot.xMins.append(xMin)
                        tfAnnot.xMaxs.append(xMax)
                        tfAnnot.yMins.append(yMin)
                        tfAnnot.yMaxs.append(yMax)
                        tfAnnot.textLabels.append(label.encode("utf8"))
                        tfAnnot.classes.append(config.CLASSES[label])
                        tfAnnot.difficult.append(0)

                        total += 1

                    # encode the data point attributes using the TensorFlor
                    # helper functions
                    features = tf.train.Features(feature=tfAnnot.build())
                    example = tf.train.Example(features=features)

                    # add the example to the writer
                    writer.write(example.SerializeToString())

                # close the writer and print diagnostic informationn to the
                # user
                writer.close()
                print("[INFO] {} examples saved for '{}'".format(total, dType))

            # create obj.data for yolo
            config_yolo.OBJ_DATA_YOLO=os.path.join(config_yolo.RECORDS_YOLO, 'obj.data')
            f = open(config_yolo.OBJ_DATA_YOLO, "w")
            print(f)
            f.writelines("classes = " + str(count_classes)+"\n")
            f.writelines("train = " + config_yolo.RECORDS_YOLO + "/train.txt\n")
            f.writelines("valid = " + config_yolo.RECORDS_YOLO + "/test.txt\n")
            f.writelines("names = " + config_yolo.RECORDS_YOLO + "/obj.names\n")
            f.close()
            config_yolo.NUM_CLASSES = count_classes
            self.manager.get_screen('FineTuningScreenPipelineName').buildSelectModel()
            self.manager.current = 'FineTuningScreenPipelineName'



        # def extractClassesSelected(self):
        #     count = 0
        #     classNumber = 1;
        #     for idxRM, wgtRM in self.listOf_Selected_Boxes_Class.items():
        #         self.ids.boxLayoutFineTuningScreenSelectedCheckBoxesId.remove_widget(wgtRM)
        #
        #     for idx , wgt  in self.listOfCheckBox_Class.items():
        #         if wgt.active:
        #             self.listOf_Selected_CheckBoxes_Class[self.listOfLabel_Classes_CheckBoxs[count]] = classNumber
        #
        #
        #             newBox = BoxLayout()
        #             newLabel = Label()
        #             newTextInput = TextInput(text=str(classNumber)  , size_hint_x=0.2, size_hint_y=None ,height=30 , pos_hint  = {'y':0.4})
        #
        #             self.listOf_Selected_Boxes_Class["Box_" + str(classNumber)] = newBox;
        #             config.CLASSES[str(self.listOfLabel_Classes_CheckBoxs[count])] = classNumber
        #             newLabel.text = str(self.listOfLabel_Classes_CheckBoxs[count])
        #             newBox.size_hint = (1, None)
        #
        #             newBox.add_widget(newLabel)
        #             newBox.add_widget(newTextInput)
        #
        #             self.ids.boxLayoutFineTuningScreenSelectedCheckBoxesId.add_widget(newBox)
        #             classNumber = classNumber + 1
        #
        #
        #         count = count + 1
        #     self.ids.boxLayoutFineTuningScreenSelectTestProportionId.disabled=False
        #     self.ids.boxLayoutFineTuningScreenRunId2.disabled = False

        def extractClassesSelected(self):
            count = 0
            classNumber = 1;
            for idxRM, wgtRM in self.listOf_Selected_Boxes_Class.items():
                self.ids.boxLayoutFineTuningScreenSelectedCheckBoxesId.remove_widget(wgtRM)

            for idx, wgt in self.listOfCheckBox_Class.items():
                if wgt.active:
                    self.listOf_Selected_CheckBoxes_Class[str(idx).replace("CheckBox_", "")] = classNumber

                    newBox = BoxLayout()
                    newLabel = Label()
                    newTextInput = TextInput(text=str(classNumber), size_hint_x=0.2, size_hint_y=None, height=30,
                                             pos_hint={'y': 0.4})

                    self.listOf_Selected_Boxes_Class["Box_" + str(classNumber)] = newBox;
                    config.CLASSES[str(str(idx).replace("CheckBox_", ""))] = classNumber
                    newLabel.text = str(str(idx).replace("CheckBox_", ""))
                    newBox.size_hint = (1, None)

                    newBox.add_widget(newLabel)
                    newBox.add_widget(newTextInput)

                    self.ids.boxLayoutFineTuningScreenSelectedCheckBoxesId.add_widget(newBox)
                    classNumber = classNumber + 1

                count = count + 1
            self.ids.boxLayoutFineTuningScreenSelectTestProportionId.disabled = False
            self.ids.boxLayoutFineTuningScreenRunId2.disabled = False

        def extractFeatures(self ):

                print(self.pathText)
                fileHU = open('FE_HU.dat', 'w')
                fileLBP = open('FE_LBP.dat', 'w')
                fileGLCM = open('FE_GLCM.dat', 'w')
                fileSCM = open('FE_SCM.dat', 'w')
                totalFiles = len(os.listdir(self.pathText))
                progressLenCount = int(100/totalFiles);
                self.progressLen = 0 ;

                for filename in os.listdir(self.pathText):
                        image = cv2.imread(os.path.join(self.pathText, filename))
                        try:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        except:
                                continue
                        new_width = 200
                        new_height = 300

                        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        if self.ids.checkboxFeAll.active:
                                HU_FE(fileHU, image)
                                LBP_FE(fileLBP, image)
                                GLCM_FE(fileGLCM, image)
                                SCM_FE(fileSCM, image)
                        else:
                                break
                        self.progressLen += progressLenCount;

                self.progressLen = 100;
                fileHU.close()
                fileLBP.close()
                fileGLCM.close()
                fileSCM.close()
                self.popup.dismiss()

        def dismiss_popup(self):
                self._popup.dismiss()
        pass
