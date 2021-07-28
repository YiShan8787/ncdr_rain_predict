# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:27:15 2021

@author: user
"""

# import the necessary packages

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

from openpyxl import load_workbook

#os.environ["CUDA_VISIBLE_DEVICES"] = "" # use cpu
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--weather_dataset", required=False,
	help="path to input dataset", default = "/media/ubuntu/My Passport/NCDR/ncdr_rain_predict/data/weather_image")
ap.add_argument("-p", "--plot", type=str, default="loss_acc.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="rain_predict.h5",
	help="path to output loss/accuracy plot")
ap.add_argument("-gt", "--gt", type=str, default="data/gt/south.xlsx",
	help="gt for the data")
args = vars(ap.parse_args())
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 5
BS = 1
num_folds = 3

random_st = 42

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

del wb
del sheet
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading weather images...")
imagePaths = list(paths.list_images(args["weather_dataset"]))
data = []
labels = []

last_label = None
data_weather_labels = []
data_weathers = []
tmp_weathers = []

# 裁切區域的 x 與 y 座標（左上角）
x = 220
y = 163

y_len = 340
x_len = 210

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    #print(imagePath)
    label = imagePath.split(os.path.sep)[-1]
    #print(label[-8:-4])
    time = label[-8:-4]
    if time == "1200" or time == "1800":
        continue
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    #if isinstance(image,type(None)):
    #    print("error")
    #print(type(image))
    #image = cv2.resize(image, (224, 224))
    image = image[y:y+y_len,x:x+x_len]
    #print(image.shape)
    # update the data and labels lists, respectively
    #if time == 0:
     #   cv2.imshow(label, image)
    cv2.waitKey()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data.append(image)
    labels.append(label)
    
    if label[3:11] not in data_weather_labels:
        data_weather_labels.append(label[3:11])
    
    '''
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
    '''

        
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data_weathers = np.reshape(data, (-1,2,340,210,3))
data = np.array(data) / 255.0
#labels = np.array(labels)
cv2.destroyAllWindows()
data_weathers = np.array(data_weathers) / 255.0
#data_weather_labels = np.array(data_weather_labels)
print("number of videos: ", str(len(data_weathers)))
print("number of the date of the videos: ", str(data_weathers.shape[0]))

#del labels
del data

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
print("[INFO] one-hot")
category_labels = to_categorical(category_labels)

#test_shape = np.load('/media/ubuntu/My Passport/NCDR/ncdr_rain_predict/data/station_data/2014/06/20140601/huminity_npy/2014060100_huminity_arr.npy')
#print(test_shape.shape)

print("[INFO] loading station data(huminity)")

station_path = '/media/ubuntu/My Passport/NCDR/ncdr_rain_predict/data/station_data'

#data_station_huminity =[]
tmp_huminitys = []


for year in os.listdir(station_path):
    #print(file)
    year_dir = station_path + "/" + year
    for month in os.listdir(year_dir):
        month_dir = year_dir + "/" + month
        for date in os.listdir(month_dir):
            date_dir = month_dir + "/" + date + "/huminity_npy"
            for date_file in os.listdir(date_dir):
                if not date_file.endswith(".npy"):
                    break
                #print(date_file[8:10])
                time = int(date_file[8:10])
                if time >11:
                    continue
                    #print(time)
                file_name = date_file
                date_txt = date_dir + "/" + date_file
                #print(date_txt)
                f = np.load(date_txt)
                #f[f == np.nan] = 0
                
                tmp_huminitys.append(f)
            #data_station_huminity = np.reshape()
                #print(f.shape)
data_station_huminity = np.array(tmp_huminitys)
data_station_huminity = np.reshape(data_station_huminity,(-1,12,210,340,3))
del tmp_huminitys
print("number of videos: ", data_station_huminity.shape[0])

print("[INFO] loading station data(temperature)")

station_path = '/media/ubuntu/My Passport/NCDR/ncdr_rain_predict/data/station_data'

#data_station_huminity =[]
tmp_temps = []


for year in os.listdir(station_path):
    #print(file)
    year_dir = station_path + "/" + year
    for month in os.listdir(year_dir):
        month_dir = year_dir + "/" + month
        for date in os.listdir(month_dir):
            date_dir = month_dir + "/" + date + "/temp_npy"
            for date_file in os.listdir(date_dir):
                if not date_file.endswith(".npy"):
                    break
                time = int(date_file[8:10])
                if time >11:
                    continue
                    #print(time)
                file_name = date_file
                date_txt = date_dir + "/" + date_file
                #print(date_txt)
                f = np.load(date_txt)
                #f[f == np.nan] = 0
                
                
                tmp_temps.append(f)
            #data_station_huminity = np.reshape()
                #print(f.shape)
data_station_temperature = np.array(tmp_temps)
data_station_temperature = np.reshape(data_station_temperature,(-1,12,210,340,3))
print("number of videos: ", data_station_temperature.shape[0])
del tmp_temps

print("[INFO] train-test split")

(train_weather_X, test_weather_X, train_weather_Y, test_weather_Y, index_train, index_test) = train_test_split(data_weathers, category_labels, range(data_weathers.shape[0]),
	test_size=0.20, stratify=category_labels, random_state=random_st)
#for i in range(len(train_weather_Y)):
#    if np.argmax(train_weather_Y[i]) == 1:
 #       print("there is positive set in training set")
  #      break
print("train_weather shape: ", train_weather_X.shape)
del data_weathers
print("finish split weather")

(train_temp_X, test_temp_X, train_temp_Y, test_temp_Y) = train_test_split(data_station_temperature, category_labels,
	test_size=0.20, stratify=category_labels, random_state=random_st)

del  train_temp_Y
del  test_temp_Y

del data_station_temperature
print("finish split temp")

(train_huminity_X, test_huminity_X, train_huminity_Y, test_huminity_Y) = train_test_split(data_station_huminity, category_labels,
	test_size=0.20, stratify=category_labels, random_state=random_st)

del train_huminity_Y
del test_huminity_Y

del data_station_huminity
print("finish split huminity")

print("[INFO] build model")

weather_frames, weather_channels, station_frames, station_channels, rows, columns = 2,3, 12, 3,210,340

#encode model

weather_video = Input(shape=(weather_frames,
                     columns,
                     rows,
                     weather_channels))

temp_video = Input(shape=(station_frames,
                     rows,
                     columns,
                     station_channels))

huminity_video = Input(shape=(station_frames,
                     rows,
                     columns,
                     station_channels))

#vgg model

vgg_weather = VGG16(input_shape=(rows,
                              columns,
                              weather_channels),
                 weights="imagenet",
                 include_top=False)
vgg_weather.trainable = False

vgg_temp = VGG16(input_shape=(rows,
                              columns,
                              station_channels),
                 weights="imagenet",
                 include_top=False)
vgg_temp.trainable = False

vgg_huminity = VGG16(input_shape=(rows,
                              columns,
                              station_channels),
                 weights="imagenet",
                 include_top=False)
vgg_huminity.trainable = False

#cnn out

cnn_out_weather = GlobalAveragePooling2D()(vgg_weather.output)

cnn_out_temp = GlobalAveragePooling2D()(vgg_temp.output)

cnn_out_huminity = GlobalAveragePooling2D()(vgg_huminity.output)

#cnn model 

cnn_weather = Model(vgg_weather.input, cnn_out_weather)

cnn_temp = Model(vgg_temp.input, cnn_out_temp)

cnn_huminity = Model(vgg_huminity.input, cnn_out_huminity)

#encode frame

weather_encoded_frames = TimeDistributed(cnn_weather)(weather_video)

temp_encoded_frames = TimeDistributed(cnn_temp)(temp_video)

huminity_encoded_frames = TimeDistributed(cnn_huminity)(huminity_video)

# LSTM

weather_encoded_sequence = LSTM(256)(weather_encoded_frames)

temp_encoded_sequence = LSTM(256)(temp_encoded_frames)

huminity_encoded_sequence = LSTM(256)(huminity_encoded_frames)

#concate

encoded_sequence = concatenate([weather_encoded_sequence, temp_encoded_sequence, huminity_encoded_sequence])

# dense layer

hidden_layer = Dense(1024, activation="relu")(encoded_sequence)
outputs = Dense(2, activation="softmax")(hidden_layer)

# build all model

model = Model(inputs = [weather_video, temp_video, huminity_video], outputs= [outputs])
model.summary()

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
#inputs_set = np.concatenate((train_weather_X, test_weather_X), axis=0)
#targets_set = np.concatenate((train_weather_Y, test_weather_Y), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1

for train_index, test_index in kfold.split(train_weather_X, train_weather_Y):
    print("TRAIN:", train_index, "TEST:", test_index)

    X_weather_train, X_weather_test = train_weather_X[train_index], train_weather_X[test_index]
    X_temp_train, X_temp_test = train_temp_X[train_index], train_temp_X[test_index]
    X_huminity_train, X_huminity_test = train_huminity_X[train_index], train_huminity_X[test_index]

    Y_train, Y_test = train_weather_Y[train_index], train_weather_Y[test_index]

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit([X_weather_train, X_temp_train, X_huminity_train], Y_train,
                batch_size=BS,
                epochs=EPOCHS,
                verbose=1)

    # Generate generalization metrics
    scores = model.evaluate([X_weather_test, X_temp_test, X_huminity_test], train_weather_Y[test_index], verbose=0,batch_size = BS)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
f_log = open("demofile3.txt", "w")
print('------------------------------------------------------------------------')
f_log.write('------------------------------------------------------------------------')
print('Score per fold')
f_log.write('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  f_log.write('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
  f_log.write(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

f_log.write('------------------------------------------------------------------------')
f_log.write('Average scores for all folds:')
f_log.write(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
f_log.write(f'> Loss: {np.mean(loss_per_fold)}')
f_log.write('------------------------------------------------------------------------')


# make predictions on the testing set
print("[INFO] evaluating network...")
f_log.write("[INFO] evaluating network...")
predIdxs = model.predict([test_weather_X, test_temp_X, test_huminity_X], batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
#print("test index of true")
#print(labels[np.where(np.argmax(test_weather_Y, axis=1)==1)])
#f_log(labels[np.where(np.argmax(test_weather_Y, axis=1)==1)])
print(classification_report(test_weather_Y.argmax(axis=1), predIdxs,
	target_names=['False', 'True']))

f_log.write(classification_report(test_weather_Y.argmax(axis=1), predIdxs,
	target_names=['False', 'True']))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(test_weather_Y.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

#f_log.write(cm.tostring())
f_log.write("acc: {:.4f}".format(acc))
f_log.write("sensitivity: {:.4f}".format(sensitivity))
f_log.write("specificity: {:.4f}".format(specificity))



# plot the training loss and accuracy
#print(history.history.keys())
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
#plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save(args["model"])
f_log.close()

'''
    
    
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