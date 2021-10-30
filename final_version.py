import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import random

from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Layer, BatchNormalization, Input, Reshape, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam, Adamax, SGD
from tensorflow.keras import Model, Input 
from tensorflow.keras.callbacks import EarlyStopping

import plotly.express as px
import plotly.graph_objects as go

import os

from pathlib import Path

print("\n\n******************************\n\n")







########## LOAD DATA ##########
pathname = "/Users/lilycollins/Desktop/SRA/research_project/samples/samples"

captcha_list = []
img_shape = (50, 200, 1)
symbols = list(map(chr, range(97, 123))) + list(map(chr, range(48, 58))) # the symbols that can be in captcha

len_symbols = len(symbols) # the number of symbols
nSamples = len(os.listdir(pathname)) # the number of samples 'captchas'
len_captcha = 5

X = np.zeros((nSamples, 50, 200, 1)) # 1070 * 50 * 200
y = np.zeros((5, nSamples, len_symbols)) # 5 * 1070 * 36






########## PROCESS FILENAMES ##########

for i, captcha in enumerate(os.listdir(pathname)):
    captcha_code = captcha.split(".")[0]
    captcha_list.append(captcha_code)
    captcha_cv2 = cv2.imread(os.path.join(pathname, captcha), cv2.IMREAD_GRAYSCALE)
    captcha_cv2 = captcha_cv2 / 255.0
    captcha_cv2 = np.reshape(captcha_cv2, img_shape)
    targs = np.zeros((len_captcha, len_symbols))
    
    for a, b in enumerate(captcha_code):
        targs[a, symbols.index(b)] = 1
    
    X[i] = captcha_cv2
    y[:, i] = targs


print("shape of X:", X.shape)
print("shape of y:", y.shape)






########## SPLIT DATA ##########

X_train = X[:856] 
y_train = y[:, :856]
X_test = X[856:]
y_test = y[:, 856:]

img_width = 200
img_height = 50
max_length = 5












########## DEFINE MODEL ##########
print("\n\n")
captcha = Input(shape=(50,200,1))
# print("\n\n")

x = Conv2D(16, (3,3),padding='same',activation='relu')(captcha)
x = MaxPooling2D((2,2) , padding='same')(x)

x = Conv2D(32, (3,3),padding='same',activation='relu')(x)
x = MaxPooling2D((2,2) , padding='same')(x)

x = Conv2D(32, (3,3),padding='same',activation='relu')(x)
x = MaxPooling2D((2,2) , padding='same')(x)

x = Conv2D(32, (3,3),padding='same',activation='relu')(x)
x = MaxPooling2D((2,2) , padding='same')(x)

x = BatchNormalization()(x)


flatOutput = Flatten()(x)

dense1 = Dense(64 , activation='relu')(flatOutput)
dropout1 = Dropout(0.5)(dense1)
dense1 = Dense(64 , activation='relu')(dropout1)
dropout1 = Dropout(0.5)(dense1)
output1 = Dense(len_symbols , activation='sigmoid' , name='char_1')(dropout1)

dense2 = Dense(64 , activation='relu')(flatOutput)
dropout2 = Dropout(0.5)(dense2)
dense2 = Dense(64 , activation='relu')(dropout2)
dropout2 = Dropout(0.5)(dense2)
output2 = Dense(len_symbols , activation='sigmoid' , name='char_2')(dropout2)
    
dense3 = Dense(64 , activation='relu')(flatOutput)
dropout3 = Dropout(0.5)(dense3)
dense3 = Dense(64 , activation='relu')(dropout3)
dropout3 = Dropout(0.5)(dense3)
output3 = Dense(len_symbols , activation='sigmoid' , name='char_3')(dropout3)
    
dense4 = Dense(64 , activation='relu')(flatOutput)
dropout4 = Dropout(0.5)(dense4)
dense4 = Dense(64 , activation='relu')(dropout4)
dropout4 = Dropout(0.5)(dense4)
output4 = Dense(len_symbols , activation='sigmoid' , name='char_4')(dropout4)
    
dense5 = Dense(64 , activation='relu')(flatOutput)
dropout5 = Dropout(0.5)(dense5)
dense5 = Dense(64 , activation='relu')(dropout5)
dropout5 = Dropout(0.5)(dense5)
output5 = Dense(len_symbols , activation='sigmoid' , name='char_5')(dropout5)
    
model = Model(inputs = captcha , outputs=[output1 , output2 , output3 , output4 , output5])








########## COMPLILE AND FIT MODEL ##########

maxs = Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="a")

sgd = SGD(learning_rate=0.002,
                               decay=1e-6,
                               momentum=0.9,
                               nesterov=True,
                               clipnorm=5)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
earlystopping = EarlyStopping(monitor ="val_loss",  
                             mode ="min", patience = 5,  
                             restore_best_weights = True) 

history = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=120, verbose=1, validation_split=0.2) #, callbacks =[earlystopping]






########## TEST MODEL ##########

score = model.evaluate(X_test,[y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]],verbose=1)

print('Test Loss and accuracy:', score)













########## GRAPH THE LOSS OF TRAIN AND TEST OVER EPOCHS ##########

# plt.figure(figsize=(15,8))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()






def makePredict(captcha):
    captcha = np.reshape(captcha , (50,200))
    result = model.predict(np.reshape(captcha, (1,50,200,1)))
    result = np.reshape(result ,(5,36))
    indexes =[]
    for i in result:
        indexes.append(np.argmax(i))
        
    label=''
    for i in indexes:
        label += symbols[i]
        
    return label




########## DISPLAY 4 RANDOM CAPTCHAS ##########

# randomlist = []
# for i in range(0,4):
#     n = random.randint(0,214)
#     randomlist.append(n)

# fig, axs = plt.subplots(2, 2, figsize=(15,15))
# axs[0, 0].imshow(X_test[randomlist[0]])
# axs[0, 0].set_title(makePredict(X_test[randomlist[0]]))

# axs[0, 1].imshow(X_test[randomlist[1]])
# axs[0, 1].set_title(makePredict(X_test[randomlist[1]]))

# axs[1, 0].imshow(X_test[randomlist[2]])
# axs[1, 0].set_title(makePredict(X_test[randomlist[2]]))

# axs[1, 1].imshow(X_test[randomlist[3]])
# axs[1, 1].set_title(makePredict(X_test[randomlist[3]]))
# fig.subplots_adjust(hspace=-0.6)




########## TEST MODEL ON TEST DATA ##########

actual_pred = []

for i in range(len(captcha_list[856:])):
    actual_pred.append([captcha_list[i + 856], makePredict(X_test[i])])

sameCount = 0
diffCount = 0
letterDiff = {}
correctness = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
for element in actual_pred:
    if element[0] == element[1]:
        sameCount += 1
    else:
        diffCount += 1
        correctnessPoint = 0
        for i in range(len(actual_pred[0][0])):
            if element[0][i] != element[1][i] and str(i) not in letterDiff:
                letterDiff[str(i)] = 1
            elif element[0][i] != element[1][i] and str(i) in letterDiff:
                letterDiff[str(i)] += 1
            if element[0][i] != element[1][i]:
                correctnessPoint += 1
        correctness[str(correctnessPoint)] += 1





########## PRINT ACCURACY ##########

a = sameCount/(sameCount + diffCount)
print("\n\nOverall Accuracy: " + str(a) + "\n\n")





########## PRINT GRAPH IN CHROME ##########

# x = ['True predicted', 'False predicted']
# y = [sameCount, diffCount]
# fig = go.Figure(data=[go.Bar(x = x, y = y)])
# fig.show()



print("\n\n******************************\n\n")
