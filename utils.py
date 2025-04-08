import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
import cv2
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def getName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']
    data = pd.read_csv(os.path.join('data_test','driving_log.csv'), names = columns)
    data['center'] = data['center'].apply(getName)
    print('Total images imported: ',data.shape[0])
    return data

#visualization and distribution of data

def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 4000
    hist, bins = np.histogram(data['steering'],nBins)
    print(bins)
    if display:
        center = (bins[:-1]+bins[1:]*)0.5
        print(center)
        plt.bar(center,hist,width=0.06)
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
                binDataList.append(i)
            binDataList = shuffle(binDataList)
            binDataList = binDataList[samplesPerBin:]
            removeIndexList.extend(binDataList)
        print('Removed images: ', len(removeIndexList))
        data.drop(data.index[removeIndexList],inplace=True)
        print('Remaining images: ', len(data))

        if display:
            hist, _ = np.histogram(data,['steering'],nbins)
            plt.bar(center,hist,width=0.06)
            plt.plot((-1,1),(samplesPerBins))
            plt.show()

        return data

def loadData(path,data):
    imagesPath =[]
    steering =[]

    for i in range(len(data)):
        indexData = data.iloc[i]
        imagesPath.append(os.path.join(path,'IMG',indexData[0]))
        steering.append(float(iindexData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)

    return imagesPath, steering

#image augmentation(step6)

def augmentImage(ImgPath, steering):
    img = mpimg.imread(ImgPath)

    def augmentImage(ImgPath, steering):
    img = mpimg.imread(ImgPath)

    #PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    #ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale = (1,1.2))
        img = zoom.augment_image(img)
 
    #BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)
    
    #FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = -steering
    
    return img, steering 

#imgRe, st = augmentImage('data_test\\IMG\\center_2025_03_23_02_04_33_901.jpg',0)
#plt.imshow(imgRe)
#plt.show()

def preProcessing(img):
    img = img[60:135,:,:]   #crop
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255

    return img

#imgRe = preProcessing(mpimg.imread('data_test\\IMG\\center_2025_03_23_02_04_33_901.jpg'))
#plt.imshow(imgRe)
#plt.show()

def batchGen(imagesPath, steeringList, batchsize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch),np.asarray(steeringBatch))

#NVIDIA MODEL

def createModel():
    model = Sequential()

    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    
    model.compile(Adam(lr=0.0001), loss='mse')

    return model