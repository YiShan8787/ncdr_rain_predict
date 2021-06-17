# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:27:15 2021

@author: user
"""

# import the necessary packages
'''
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

from openpyxl import load_workbook


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False,
	help="path to input dataset", default = "E:\\tech\\ncdr\\ncdr_rain_predict\\data\\weather_image")
ap.add_argument("-p", "--plot", type=str, default="loss_acc.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="rain_predict.model",
	help="path to output loss/accuracy plot")
ap.add_argument("-gt", "--gt", type=str, default="data/gt/south.xlsx",
	help="gt for the data")
args = vars(ap.parse_args())
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8


'''
need to be modify
'''

print("[INFO] loading gt")
gtPaths = args["gt"]

# 讀取 Excel 檔案
wb = load_workbook(gtPaths)
sheet = wb.active

gt_time_list = [] 

for row in sheet.rows:
    #print(row[0].value)
    gt_time_list.append(str(row[0].value))
    #for cell in row:
    #    print(cell.value)


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading weather images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

last_label = None
data_weather_labels = []
data_weathers = []
tmp_weathers = []

# 裁切區域的 x 與 y 座標（左上角）
x = 100
y = 63

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-1]
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    
    #image = cv2.resize(image, (224, 224))
    image = image[y:,:]
    # update the data and labels lists, respectively
    
    #cv2.imshow(label, image)
    #cv2.waitKey()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data.append(image)
    labels.append(label)
    
    if last_label == None:
        last_label = label[3:11]
        data_weather_labels.append(last_label)
        tmp_weathers.append(image)
    elif last_label == label[3:11]:
        tmp_weathers.append(image)
        
    if len(tmp_weathers) == 4:
        data_weathers.append(tmp_weathers)
        tmp_weathers = []
        last_label = None
        
        
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)
cv2.destroyAllWindows()
data_weathers = np.array(data_weathers) / 255.0
#data_weather_labels = np.array(data_weather_labels)
print("number of videos: ", str(len(data_weathers)))
print("number of the date of the videos: ", str(len(data_weather_labels)))

#find time in labels, then buid the frames
print("[INFO] category labeling")
category_labels = []
positive_number = 0
for i in range(len(data_weather_labels)):
    if data_weather_labels[i] in gt_time_list:
        category_labels.append(1)
        positive_number = positive_number +1
    else:
        category_labels.append(0)
    
    
print("number of positive number: ", str(positive_number))
print("number of negative number: ", str(len(data_weather_labels) - positive_number))
# perform one-hot encoding on the labels
category_labels = to_categorical(category_labels)


'''

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")


# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False
    
    
# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)


# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))



# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save(args["model"], save_format="h5")
'''