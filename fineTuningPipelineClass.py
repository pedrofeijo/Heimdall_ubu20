from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty, Clock

import getpass
import shutil
import kivy.uix.image as kIm
import numpy as np
import cv2
import os
import time, threading
import pandas as pd
import copy
import config
import config_yolo
import tfannotation
import tensorflow as tf
import tkinter as tk
from typing import List
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from object_detection.utils import config_util
from ast import literal_eval
from tensorflow.core.framework import graph_pb2
from tensorflow.python.estimator import run_config
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.contrib import slim
# from google.protobuf import text_format, unittest_pb2
from datetime import date
from shutil import copyfile

selectClassDropDownID = 0

listOfCheckBoxId = []
# homePath = '/home/lcktiroot/Heimdall/models/'
homePath = '/home/'+getpass.getuser()+'/Heimdall/models/'


class FineTuningScreenPipeline(Screen):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    pipeLinePath = ''
    # Lista de Modelos para FineTunning:
    listOf_Models_Ids = {}

    # CheckBoxes de cada classe disponivel na coluna selecionada
    listOfBox_Class_Ids = {}
    listOfCheckBox_Class = {}
    listOfLabel_Classes_CheckBoxs = []

    # Lista das Classes Selecionadas
    listOf_Selected_CheckBoxes_Class = {}
    listOf_Selected_Boxes_Class = {}
    listOfLabel_Select_CheckBoxs = []

    countVerifyImages = 0;

    df = 0
    fileName = ''

    def __init__(self, **kwa):
        super(FineTuningScreenPipeline, self).__init__(**kwa)

        self.pathText = []
        self.progressBarTime = 0;
        self.progressLen = 0;
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

        print(filename, filename[0][-4:])
        if (os.path.isdir(filename[0])):
            self.progressLen = 0;

            self.popup.title = 'Grr Humanos! , selecione um ARQUIVO CSV! '

            self.popup.open()
            mythread = threading.Thread(target=self.waitTime)
            mythread.start()

            # self.ids.text_input.text = filename[0];

            # self.ids.boxLayoutFineTuningScreenRunId.disabled = False;
            # self.ids.checkboxHUBox.disabled = True;
            # self.ids.checkboxGLCMBox.disabled = True;
            # self.ids.checkboxSCMBox.disabled = True;
            # self.ids.checkboxLBPBox.disabled = True;
        elif ('.csv' in filename[0][-4:]):

            self.ids.text_input.text = filename[0].split('/')[-1]
            pathCSV = '/'.join(filename[0].split('/')[1:-1])
            self.df = pd.read_csv(filename[0], delimiter=';')
            self.fileName = filename[0]
            selectColumnLabel = Label()
            selectColumnLabel.text = 'Selecione a Coluna de Classes'
            self.ids.boxLayoutFineTuningScreenSelectId.add_widget(selectColumnLabel)
            listOfCheckBox_Ids = {}
            config.BASE_PATH = os.path.join('/', pathCSV)
            print(config.BASE_PATH)
            try:
                os.mkdir(os.path.join(config.BASE_PATH, 'records'), 755);
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
                newCheckBox.color = (1, 1, 1, 1)
                self.listOfLabel_Select_CheckBoxs.append(headerValues)
                newBox.size_hint = (1, None)
                newBox.add_widget(newLabel)
                newBox.add_widget(newCheckBox)

                self.ids.boxLayoutFineTuningScreenSelectId.add_widget(newBox)

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

    def selectModel(self):

        for idx, wgt in self.listOf_Models_Ids.items():
            print(idx, wgt.active)
            if wgt.active:

                folder = os.path.join(homePath, idx)
                copyFolder = copy.deepcopy(folder)
                nameModel = copyFolder.replace(homePath, "")
                config_yolo.MODEL = nameModel

                if config_yolo.MODEL[:4] != 'yolo':
                    file = os.path.join(folder, "pipeline.config")

                    configs = config_util.get_configs_from_pipeline_file(file)

                    model_config = configs['model']
                    train_config = configs['train_config']
                    input_config = configs['train_input_config']

                    getattr(model_config, model_config.WhichOneof("model")).num_classes = len(
                        self.manager.get_screen('FineTuningScreenName').listOfLabel_Classes_CheckBoxs)
                    configs['train_config'].batch_size = 1
                    configs['train_config'].fine_tune_checkpoint = os.path.join(folder, "model.ckpt")
                    print(folder)
                    configs['train_input_config'].label_map_path = config.CLASSE_FILE  # PBTXT
                    configs['train_input_config'].tf_record_input_reader.input_path[
                        0] = config.TRAIN_RECORD  # trainingRecord
                    # configs['eval_config'].num_examples = # Precisa agora??
                    print('Num Classes: ', getattr(model_config, model_config.WhichOneof("model")).num_classes)

                    self.pipeLinePath = os.path.join(config.BASE_PATH, str(date.today()) + '_experiments_' + nameModel)
                    print(self.pipeLinePath)

                    try:
                        os.mkdir(self.pipeLinePath, 0o777);
                    except:
                        for filename in os.listdir(self.pipeLinePath):
                            os.unlink(os.path.join(self.pipeLinePath, filename))
                    try:
                        configs[
                            'train_config'].optimizer.momentum_optimizer.learning_rate.manual_step_learning_rate.schedule.pop(
                            0)
                    except:
                        continue
                    config_util._update_train_steps(configs, 500000)
                    pipelineConfig = config_util.create_pipeline_proto_from_configs(configs)
                    config_util.save_pipeline_config(pipelineConfig, self.pipeLinePath)
                    self.ids.scrlvPipelineTextInputModelID.text = text_format.MessageToString(pipelineConfig)
                    print(getattr(model_config, model_config.WhichOneof("model")))


                else:
                    # print('yolo')
                    # print(config.BASE_PATH)
                    self.pipeLinePath = os.path.join(config.BASE_PATH, str(date.today()) + '_experiments_' + nameModel)

                    try:
                        os.mkdir(self.pipeLinePath, 0o777);
                    except:
                        for filename in os.listdir(self.pipeLinePath):
                            os.unlink(os.path.join(self.pipeLinePath, filename))

                    with open(config_yolo.RECORDS_YOLO + '/obj.data', 'a') as file:
                        file.writelines('backup = ' + self.pipeLinePath)

                    config_yolo.CFG_YOLO = config_yolo.RECORDS_YOLO + "/yolo-obj.cfg"
                    copyfile(homePath + nameModel + "/darknet/yolo-obj.cfg", config_yolo.CFG_YOLO)

                    with open(config_yolo.CFG_YOLO, 'r') as f:
                        get_all = f.readlines()  # type: List[str]

                    # edit yolo config
                    if config_yolo.MODEL[:6] == "yolov2":
                        with open(config_yolo.CFG_YOLO, 'w') as f:
                            for i, line in enumerate(get_all, 1):
                                if config_yolo.MODEL == "yolov2_voc2":
                                    if i == 224:
                                        f.writelines("filters=" + str((int(config_yolo.NUM_CLASSES) + 5) * 5) + "\n")
                                    elif i == 230:
                                        f.writelines("classes=" + str(config_yolo.NUM_CLASSES) + "\n")
                                    else:
                                        f.writelines(line)
                                else:
                                    if i == 237:
                                        f.writelines("filters=" + str((int(config_yolo.NUM_CLASSES) + 5) * 5) + "\n")
                                    elif i == 244:
                                        f.writelines("classes=" + str(config_yolo.NUM_CLASSES) + "\n")
                                    else:
                                        f.writelines(line)

                    if config_yolo.MODEL[:6] == "yolov3":
                        with open(config_yolo.CFG_YOLO, 'w') as f:
                            for i, line in enumerate(get_all, 1):
                                if i == 603:
                                    f.writelines("filters=" + str((int(config_yolo.NUM_CLASSES) + 5) * 3) + "\n")
                                elif i == 610:
                                    f.writelines("classes=" + str(config_yolo.NUM_CLASSES) + "\n")
                                elif i == 689:
                                    f.writelines("filters=" + str((int(config_yolo.NUM_CLASSES) + 5) * 3) + "\n")
                                elif i == 696:
                                    f.writelines("classes=" + str(config_yolo.NUM_CLASSES) + "\n")
                                elif i == 776:
                                    f.writelines("filters=" + str((int(config_yolo.NUM_CLASSES) + 5) * 3) + "\n")
                                elif i == 783:
                                    f.writelines("classes=" + str(config_yolo.NUM_CLASSES) + "\n")
                                else:
                                    f.writelines(line)

                    with open(config_yolo.CFG_YOLO, 'r') as f:
                        get_all = f.readlines()
                    full_get_all = ''.join(elem for elem in get_all)
                    self.ids.scrlvPipelineTextInputModelID.text = full_get_all


    def buildSelectModel(self):
        for dirName in os.listdir(homePath):
            print(dirName)
            newBox = BoxLayout()
            newLabel = Label()
            newCheckBox = CheckBox()
            self.manager.get_screen('FineTuningScreenPipelineName').listOf_Models_Ids[str(dirName)] = newCheckBox;
            newLabel.text = dirName
            newLabel.text_size = (self.width, None)
            newCheckBox.group = 'PipelineSelectModelGroup'
            newCheckBox.color = (1, 1, 1, 1)
            newBox.size_hint = (1, None)
            newBox.add_widget(newLabel)
            newBox.add_widget(newCheckBox)
            self.manager.get_screen(
                'FineTuningScreenPipelineName').ids.boxLayoutFineTuningScreenPipelineSelectModelId.add_widget(newBox)

    def puopen(self, instance):
        Clock.schedule_interval(self.next, 1 / 25)

    def change_scroll_y(self, ti, scrlv):
        y_cursor = ti.cursor_pos[1]
        y_bar = scrlv.scroll_y * (ti.height - scrlv.height)
        if ti.height > scrlv.height:
            if y_cursor >= y_bar + scrlv.height:
                dy = y_cursor - (y_bar + scrlv.height)
                scrlv.scroll_y = scrlv.scroll_y + scrlv.convert_distance_to_scroll(0, dy)[1]
            if y_cursor - ti.line_height <= y_bar:
                dy = (y_cursor - ti.line_height) - y_bar
                scrlv.scroll_y = scrlv.scroll_y + scrlv.convert_distance_to_scroll(0, dy)[1]

    def train(self):
        if config_yolo.MODEL[:4] != 'yolo':
            pipeline_config_path = os.path.join(self.pipeLinePath, "pipeline.config")

            with open(pipeline_config_path, 'wb') as f:
                f.write(self.ids.scrlvPipelineTextInputModelID.text.encode())

            os.system(
                "gnome-terminal -e 'bash -c \"cd /home/"+ getpass.getuser()+"/tensorflow/models/research  ; python object_detection/legacy/train.py --logtostderr --pipeline_config_path=" + os.path.join(
                    self.pipeLinePath, 'pipeline.config') + " --train_dir=" + os.path.join(
                    self.pipeLinePath) + ";  exec bash\"'")


        else:

            with open(config_yolo.CFG_YOLO, 'wb') as f:
                f.write(self.ids.scrlvPipelineTextInputModelID.text.encode())

            os.system(
                "gnome-terminal -e 'bash -c \"cd /home/"+ getpass.getuser()+"/Heimdall/models/" + config_yolo.MODEL + "/darknet  ; ./darknet detector train " + config_yolo.OBJ_DATA_YOLO + " " + config_yolo.CFG_YOLO + ";  exec bash\"'")

    def buildConfigFiles(self):
        # open the classes output file
        f = open(config.CLASSE_FILE, "w")

        # loop over the classes
        for (k, v) in config.CLASSES.items():
            # construct the class information and write to file
            item = ("item {\n"
                    "\tid: " + str(v) + "\n"
                                        "\tname: '" + k + "'\n"
                                                          "}\n")
            f.write(item)

            # close the output classes file
        f.close()

        # initialize a data dictionary used to map each image filename
        # to all bounding boxes associated with the image, then load
        # the contents of the annotations file
        D = {}
        rows = open(config.ANNOT_PATH).read().strip().split("\n")

        for row in rows[1:]:
            # break the row into components
            row = row.split(",")[0].split(";")
            (imagePath, label, startX, startY, endX, endY, _) = row
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
                                                 test_size=float(self.ids.text_input_TestProportion.text),
                                                 random_state=42)

        # initialize the data split files
        datasets = [
            ("train", trainKeys, config.TRAIN_RECORD),
            ("test", testKeys, config.TEST_RECORD)

        ]
        # loop over the datasets
        for (dType, keys, outputPath) in datasets:
            # inicialize the TensorFlow wirter and initialize the total
            # number of examples written to file
            print("[INFO] processing '{}'...".format(dType))
            writer = tf.python_io.TFRecordWriter(outputPath)
            total = 0

            # loop over all thhe keys in the current set
            for k in keys:
                # load the input image from disk as a TensorFlow object
                # print( os.path.join(config.BASE_PATH,k))
                encoded = tf.gfile.GFile(os.path.join(config.BASE_PATH, k), "rb").read()
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

                for (label, (startX, startY, endX, endY)) in D[k]:
                    # TensorFlow assumes all bounding boxes are in the
                    # range [0,1] so we need to scale them

                    xMin = startX / w
                    xMax = endX / w
                    yMin = startY / h
                    yMax = endY / h

                    if self.countVerifyImages < 5:
                        image = cv2.imread(k)
                        startX = int(xMin * w)
                        startY = int(yMin * h)
                        endX = int(xMax * w)
                        endY = int(yMax * h)

                        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        root = tk.Tk()
                        screen_width = root.winfo_screenwidth()
                        screen_height = root.winfo_screenheight()
                        image = cv2.resize(image, (screen_width, screen_height), interpolation=cv2.INTER_AREA)

                        cv2.imshow("Image", image)
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

    def extractClassesSelected(self):
        count = 0
        classNumber = 1;
        for idx, wgt in self.listOfCheckBox_Class.items():
            if wgt.active:
                self.listOf_Selected_CheckBoxes_Class[self.listOfLabel_Classes_CheckBoxs[count]] = classNumber

                newBox = BoxLayout()
                newLabel = Label()
                newTextInput = TextInput(text=str(classNumber), size_hint_x=0.2, size_hint_y=None, height=30,
                                         pos_hint={'y': 0.4})

                self.listOf_Selected_Boxes_Class["Box_" + str(classNumber)] = newBox;
                config.CLASSES[str(self.listOfLabel_Classes_CheckBoxs[count])] = classNumber
                newLabel.text = str(self.listOfLabel_Classes_CheckBoxs[count])

                newBox.size_hint = (1, None)
                newBox.add_widget(newLabel)
                newBox.add_widget(newTextInput)

                self.ids.boxLayoutFineTuningScreenSelectedCheckBoxesId.add_widget(newBox)
                classNumber = classNumber + 1
            count = count + 1
        self.ids.boxLayoutFineTuningScreenSelectTestProportionId.disabled = False
        self.ids.boxLayoutFineTuningScreenRunId2.disabled = False

    def extractFeatures(self):

        print(self.pathText)
        fileHU = open('FE_HU.dat', 'w')
        fileLBP = open('FE_LBP.dat', 'w')
        fileGLCM = open('FE_GLCM.dat', 'w')
        fileSCM = open('FE_SCM.dat', 'w')
        totalFiles = len(os.listdir(self.pathText))
        progressLenCount = int(100 / totalFiles);
        self.progressLen = 0;

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
