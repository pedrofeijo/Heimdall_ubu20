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


class ImageTestYoloScreen(Screen):
        loadfile = ObjectProperty(None)
        savefile = ObjectProperty(None)
        text_input = ObjectProperty(None)

        imagePath = ''
        weightsPath = ''
        namesPath = ''
        cfgPath = ''
        outputPath = ''

        videoPath = ''
        weightsVideoPath = ''
        namesVideoPath = ''
        cfgVideoPath = ''
        outputVideoPath = ''

        numberLabels = 0
        countVerifyImages = 0;


        df = 0
        fileName = ''
        def __init__(self, **kwa):
                super(ImageTestYoloScreen, self).__init__(**kwa)

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

        def show_load_weights(self):
                content = LoadDialog(load=self.load_weights, cancel=self.dismiss_popup)
                self._popup = Popup(title="Selecione um Peso(.WEIGHTS)", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

        def show_load_names(self):
                content = LoadDialog(load=self.load_names, cancel=self.dismiss_popup)
                self._popup = Popup(title="Selecione as Classes(.NAMES) ", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

        def show_load_cfg(self):
                content = LoadDialog(load=self.load_cfg, cancel=self.dismiss_popup)
                self._popup = Popup(title="Selecione o arquivo de configuração(.CFG) ", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

        def show_load_image(self):
                content = LoadDialog(load=self.load_image, cancel=self.dismiss_popup)
                self._popup = Popup(title="Selecione uma Imagem", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

        def show_load_weights_video(self):
                content = LoadDialog(load=self.load_weights_video, cancel=self.dismiss_popup)
                self._popup = Popup(title="Selecione um Peso(.WEIGHTS)", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

        def show_load_names_video(self):
                content = LoadDialog(load=self.load_names_video, cancel=self.dismiss_popup)
                self._popup = Popup(title="Selecione as Classes(.NAMES) ", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

        def show_load_cfg_video(self):
                content = LoadDialog(load=self.load_cfg_video, cancel=self.dismiss_popup)
                self._popup = Popup(title="Selecione o arquivo de configuração(.CFG) ", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

        def show_load_video(self):
                content = LoadDialog(load=self.load_video, cancel=self.dismiss_popup)
                self._popup = Popup(title="Selecione um Video", content=content, size_hint=(0.9, 0.9))
                self._popup.open()

        def waitTime(self):
                time.sleep(3)
                self.popup.dismiss()

        def load_weights(self, path, filename):

                print(filename,filename[0][-8:])
                if (os.path.isdir(filename[0])):
                        self.progressLen = 0;

                        self.popup.title ='Grr, Humanos! , selecione um ARQUIVO .WEIGHTS! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()
                elif ('.weights' in filename[0][-8:]):

                    self.ids.text_input_weights.text = filename[0].split('/')[-1]
                    self.weightsPath = os.path.join('/','/'.join(filename[0].split('/')[1:]))

                else:
                    self.progressLen = 0;
                    self.popup.title = 'Por Odin! Eu pedi um .WEIGHTS! '
                    self.popup.open()
                    mythread = threading.Thread(target=self.waitTime)
                    mythread.start()

                self.dismiss_popup()

        def load_names(self, path, filename):

                print(filename, filename[0][-6:])
                if (os.path.isdir(filename[0])):
                        self.progressLen = 0;

                        self.popup.title = 'Grr, Humanos! , selecione um ARQUIVO .NAMES! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()
                elif ('.names' in filename[0][-6:]):
                    self.ids.text_input_names.text = str(filename[0].split('/')[-1])
                    self.namesPath = os.path.join('/','/'.join(filename[0].split('/')[1:]))

                else:
                    self.progressLen = 0;
                    self.popup.title = 'Por Odin! Eu pedi um .NAMES! '
                    self.popup.open()
                    mythread = threading.Thread(target=self.waitTime)
                    mythread.start()

                self.dismiss_popup()

        def load_cfg(self, path, filename):

                print(filename, filename[0][-4:])
                if (os.path.isdir(filename[0])):
                        self.progressLen = 0;

                        self.popup.title = 'Grr, Humanos! , selecione um ARQUIVO .CFG! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()
                elif('.cfg' in filename[0][-4:]):
                    self.ids.text_input_cfg.text = str(filename[0].split('/')[-1])
                    self.cfgPath = os.path.join('/','/'.join(filename[0].split('/')[1:]))

                else:
                    self.progressLen = 0;
                    self.popup.title = 'Por Odin! Eu pedi um CFG! '
                    self.popup.open()
                    mythread = threading.Thread(target=self.waitTime)
                    mythread.start()

                self.dismiss_popup()

        def load_image(self, path, filename):

                print(filename, filename[0][-4:])
                if (os.path.isdir(filename[0])):
                        self.progressLen = 0;

                        self.popup.title = 'Grr Humanos! , selecione um ARQUIVO CSV! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()
                else:

                    self.ids.text_input_image.text = filename[0].split('/')[-1]
                    self.imagePath = os.path.join('/','/'.join(filename[0].split('/')[1:]))

                self.dismiss_popup()

        def load_weights_video(self, path, filename):

                print(filename,filename[0][-8:])
                if (os.path.isdir(filename[0])):
                        self.progressLen = 0;

                        self.popup.title ='Grr, Humanos! , selecione um ARQUIVO .WEIGHTS! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()
                elif ('.weights' in filename[0][-8:]):

                    self.ids.text_input_weights_video.text = filename[0].split('/')[-1]
                    self.weightsVideoPath = os.path.join('/','/'.join(filename[0].split('/')[1:]))

                else:
                    self.progressLen = 0;
                    self.popup.title = 'Por Odin! Eu pedi um .WEIGHTS! '
                    self.popup.open()
                    mythread = threading.Thread(target=self.waitTime)
                    mythread.start()

                self.dismiss_popup()

        def load_names_video(self, path, filename):

                print(filename, filename[0][-6:])
                if (os.path.isdir(filename[0])):
                        self.progressLen = 0;

                        self.popup.title = 'Grr, Humanos! , selecione um ARQUIVO .NAMES! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()
                elif ('.names' in filename[0][-6:]):
                    self.ids.text_input_names_video.text = str(filename[0].split('/')[-1])
                    self.namesVideoPath = os.path.join('/','/'.join(filename[0].split('/')[1:]))

                else:
                    self.progressLen = 0;
                    self.popup.title = 'Por Odin! Eu pedi um .NAMES! '
                    self.popup.open()
                    mythread = threading.Thread(target=self.waitTime)
                    mythread.start()

                self.dismiss_popup()

        def load_cfg_video(self, path, filename):

                print(filename, filename[0][-4:])
                if (os.path.isdir(filename[0])):
                        self.progressLen = 0;

                        self.popup.title = 'Grr, Humanos! , selecione um ARQUIVO .CFG! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()
                elif('.cfg' in filename[0][-4:]):
                    self.ids.text_input_cfg_video.text = str(filename[0].split('/')[-1])
                    self.cfgVideoPath = os.path.join('/','/'.join(filename[0].split('/')[1:]))

                else:
                    self.progressLen = 0;
                    self.popup.title = 'Por Odin! Eu pedi um CFG! '
                    self.popup.open()
                    mythread = threading.Thread(target=self.waitTime)
                    mythread.start()

                self.dismiss_popup()

        def load_video(self, path, filename):

                print(filename, filename[0][-4:])
                if (os.path.isdir(filename[0])):
                        self.progressLen = 0;

                        self.popup.title = 'Grr Humanos! , selecione um ARQUIVO CSV! '

                        self.popup.open()
                        mythread = threading.Thread(target=self.waitTime)
                        mythread.start()
                else:

                    self.ids.text_input_video.text = filename[0].split('/')[-1]
                    self.videoPath = os.path.join('/','/'.join(filename[0].split('/')[1:]))

                self.dismiss_popup()




        def next(self, dt):
                if self.progress_bar.value >= 100:
                        return False
                self.progress_bar.value = self.progressLen


        def run(self ):
            print(self.imagePath)
            print(self.weightsPath)
            print(self.namesPath)
            print(self.cfgPath)
            self.outputPath = os.path.join(os.path.dirname(self.cfgPath), "output_image.jpg")
            print(self.outputPath)
            #print("gnome-terminal -e 'bash -c \"python predict_video_yolo.py --input "+self.videoPath + " --output " + self.outputPath +" --weights " + self.weightsPath +" --names " + self.namesPath + " --config "+ self.cfgPath+";  exec bash\"'")
            os.system("gnome-terminal -e 'bash -c \"python predict_image_yolo.py --input "+self.imagePath + " --output " + self.outputPath +" --weights " + self.weightsPath +" --names " + self.namesPath + " --config "+ self.cfgPath+";  exec bash\"'")

        def run_video(self ):
            print(self.videoPath)
            print(self.weightsVideoPath)
            print(self.namesVideoPath)
            print(self.cfgVideoPath)
            self.outputVideoPath = os.path.join(os.path.dirname(self.cfgVideoPath), "output.avi")
            print(self.outputVideoPath)
            #print("gnome-terminal -e 'bash -c \"python predict_video_yolo.py --input "+self.videoPath + " --output " + self.outputPath +" --weights " + self.weightsPath +" --names " + self.namesPath + " --config "+ self.cfgPath+";  exec bash\"'")
            os.system("gnome-terminal -e 'bash -c \"python predict_video_yolo.py --input "+self.videoPath + " --output " + self.outputVideoPath +" --weights " + self.weightsVideoPath +" --names " + self.namesVideoPath + " --config "+ self.cfgVideoPath+";  exec bash\"'")


        def puopen(self, instance):
                Clock.schedule_interval(self.next, 1 / 25)

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

                print("[INFO] processing '{}'...".format(dType))
                writer = tf.python_io.TFRecordWriter(outputPath)
                total = 0

                # loop over all thhe keys in the current set
                for k in keys:
                    # load the input image from disk as a TensorFlow object
                    # print( os.path.join(config.BASE_PATH,k))
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

                            cv2.rectangle(image,(startX, startY), (endX,endY),(0,255,0),2)

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
            self.manager.get_screen('FineTuningScreenPipelineName').buildSelectModel()
            self.manager.current = 'FineTuningScreenPipelineName'



        def dismiss_popup(self):
                self._popup.dismiss()
        pass