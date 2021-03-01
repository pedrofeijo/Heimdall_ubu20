# import the necessary packages
import cv2
import glob
import os
import argparse
import math
import csv
import numpy as np
import pandas as pd

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
df = pd.DataFrame()
df_temp = pd.DataFrame()
vec_size = 0
refPt = []
close_pol = 'NULL'
cropping = False
class_selected = 0
regions = []
file_pos = 0
NUM_IMGS = 0
MAX_WIDTH = 1280
MAX_HEIGHT = 800

class_colours = [(58, 36, 170), (218, 135, 9), (166, 206, 227), (31, 120, 180), (178, 223, 138), (51, 160, 44), (251, 154, 153), (227, 26, 28), (253, 191, 111),
                 (255, 127, 0)]


def draw_info(image):

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(image, (7, 5), (103, 23), (138, 136, 142), cv2.FILLED)
    cv2.putText(image, 'Selected: {}'.format(class_selected), (10, 20),
                font, 0.5, class_colours[class_selected], 1, cv2.LINE_AA)

    pos_y = 40
    for i in range(0, len(class_colours)):
        cv2.putText(image, 'Class {}'.format(i), (10, pos_y),
                    font, 0.5, class_colours[i], 1, cv2.LINE_4)
        pos_y = pos_y + 15

    cv2.rectangle(image, (7, pos_y+13), (95, pos_y+30),
                  (138, 136, 142), cv2.FILLED)
    cv2.putText(image, '{} of {}'.format(file_pos+1, NUM_IMGS),
                (10, pos_y+25), font, 0.5, (255, 255, 255), 1, cv2.LINE_8)

    cv2.putText(image, 'p (previus)', (10, pos_y + 70),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_8)
    cv2.putText(image, 'n (next|save)', (10, pos_y + 85),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_8)
    cv2.putText(image, 'r (reset)', (10, pos_y + 100),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_8)
    cv2.putText(image, 'q (quit)', (10, pos_y + 115),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_8)


def save_regions(image_path, regions, dimensions):
    # Replace jpg path to read txt file
    global df, df_temp
    if len(refPt) > vec_size:
        del refPt[-1]

    if regions:
        print('\nSaving ... {}'.format(regions))
        for region in regions:
            coords = np.array(region['region'])
            coords_x = coords[:, 0]
            coords_y = coords[:, 1]
            upper_x, lower_x = int(coords[:, 0].min()), int(coords[:, 0].max())
            upper_y, lower_y = int(coords[:, 1].min()), int(coords[:, 1].max())
            df_temp = pd.DataFrame(
                [[image_path, region['class'], upper_x, upper_y, lower_x, lower_y, np.array(coords_x), np.array(coords_y)]])
            df = df.append(df_temp, ignore_index=True)

            df.to_csv(csvfilename, sep=';', index=None, header=None)

def read_markers(image_path, dimensions):
    global regions, df
    df = pd.read_csv(csvfilename, header=None, sep=';')
    regions = []
    element = dict()
    for i in df.iloc[:][0][df.iloc[:][0] == image_path].index:
        # if df.shape[1] == 4:
        coords_x = df.iloc[i, -2][1:-1].split(' ')
        coords_y = df.iloc[i, -1][1:-1].split(' ')
        coords_x = [int(j) for j in coords_x if j.isdigit() == True]
        coords_y = [int(j) for j in coords_y if j.isdigit() == True]
        element['region'] = [[coords_x[j], coords_y[j]]
                                for j in range(len(coords_x))]
        element['class'] = int(df.iloc[i, 1])

        regions.append(element)
        print_regions()
        regions = []

def read_img(file_path):
    global MAX_WIDTH, MAX_HEIGHT
    image = cv2.imread(file_path)
    dimensions = image.shape
    print('{} {}'.format(file_path, dimensions))

    if dimensions[1] > MAX_WIDTH or dimensions[0] > MAX_HEIGHT:
        if math.ceil(dimensions[1] / MAX_WIDTH) > math.ceil(dimensions[0] / MAX_HEIGHT):
            denominator = math.ceil(dimensions[1] / MAX_WIDTH)
        else:
            denominator = math.ceil(dimensions[0] / MAX_HEIGHT)

        image = cv2.resize(image, None, fx=1 / denominator,
                           fy=1 / denominator, interpolation=cv2.INTER_CUBIC)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    # read_markers(file_path)
    return image


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, close_pol
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if close_pol == 'yes':
            refPt = []
            close_pol = 'no'
        refPt.append([x, y])
        cropping = True
        if len(refPt) > 1:
            if np.sqrt((refPt[-1][0] - refPt[0][0])**2 + (refPt[-1][1] - refPt[0][1])**2) <= 20:
                close_pol = 'yes'
                element = dict()
                element['region'] = refPt
                element['class'] = class_selected
                regions.append(element)
                save_regions(files[file_pos], regions, image.shape)
                refPt.append(refPt[0])
            cv2.line(image, tuple(
                refPt[-2]), tuple(refPt[-1]), class_colours[class_selected], 2)
            cv2.imshow("image", image)
            if refPt[0] == refPt[-1]:
                coords = np.array(regions[0]['region'])
                upper_x, lower_x = coords[:, 0].min(), coords[:, 0].max()
                upper_y, lower_y = coords[:, 1].min(), coords[:, 1].max()
                cv2.rectangle(image, tuple([upper_x, upper_y]), tuple(
                    [lower_x, lower_y]), class_colours[class_selected], 2)
                cv2.imshow("image", image)
    elif event == cv2.EVENT_RBUTTONDOWN:
        refPt = []
        cropping = False


def print_regions():
    # cv2.imshow("image", image)

    for region in regions:
        class_type = region['class']
        region = region['region']
        # print(region[0])
        # draw a rectangle around the region of interest
        if len(region) == 2:
            cv2.rectangle(image, tuple(region[0]), tuple(
                region[1]), class_colours[class_type], 2)
            cv2.imshow("image", image)
        elif len(region) > 2:
            region.append(region[0])
            # print(region[-1])
            for i in range(2, (len(region)+1)):
                # print(region[i-1], region[i-2])
                cv2.line(image, tuple(
                    region[i-2]), tuple(region[i-1]), class_colours[class_type], 2)
                cv2.imshow("image", image)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--path", required=True, help="Path to the image", type=str)
    # ap.add_argument('-d', '--dimension', nargs=2, help='Max width and height to show the image', required=True, type=int)
    args = vars(ap.parse_args())
    csvfilename = 'file_annotations.csv'
    # csvfilename1 = '/home/lapiscoatlanta02/Documents/yolo-marker-master/file_annotations1.csv'
    args['path'] = '/home/lapiscoatlanta02/Documents/marker-pets/imgs/*.jpg'
    args['dimension'] = [1920, 1080]

    MAX_WIDTH = args['dimension'][0]
    MAX_HEIGHT = args['dimension'][1]

    # Image path list
    files = glob.glob(args['path'])
    NUM_IMGS = len(files)

    if csvfilename not in os.listdir('.'):
        df = pd.DataFrame(
            [['Filename', 'Annotation tag', 'Upper_left_X', 'Upper_left_Y', 'Lower_right_X', 'Lower_right_Y', 'Coords X', 'Coords Y']])
        df.to_csv(csvfilename, sep=';', index=None, header=None)

    if not NUM_IMGS:
        print('No image!')
        exit(0)

    # Read the first image and its markers
    image = read_img(files[file_pos])
    read_markers(files[file_pos], image.shape)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        # image = read_img(files[file_pos])
        read_markers(files[file_pos], image.shape)
        draw_info(image)

        cv2.imshow("image", image)
        key = cv2.waitKey(-1)

        # if the '0-9' key is pressed, class is setted
        if key >= 48 and key <= 57:
            class_selected = int(chr(key))

        if key == ord("n"):
            regions = []

            if file_pos + 1 < NUM_IMGS:
                file_pos = file_pos + 1
                image = read_img(files[file_pos])
                read_markers(files[file_pos], image.shape)

        if key == ord("p"):
            if file_pos > 0:
                file_pos = file_pos - 1
                image = read_img(files[file_pos])
                read_markers(files[file_pos], image.shape)

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = read_img(files[file_pos])
            print('Cleaning regions')
            regions = []
            df = df.drop(
                df.index[df.iloc[:][0][df.iloc[:][0] == files[file_pos]].index]).dropna()
            df.to_csv(csvfilename, sep=';', index=None, header=None)
            print_regions()

        # if the 'q' key is pressed, break from the loop
        elif key == ord("q"):
            break

    # close all open windows
    cv2.destroyAllWindows()
