import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense


clas = Sequential()

clas.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation ='relu'))
clas.add(MaxPooling2D(pool_size=(2,2)))

clas.add(Convolution2D(32,3,3, activation ='relu'))
clas.add(MaxPooling2D(pool_size=(2,2)))



clas.add(Flatten())

clas.add(Dense(output_dim=128,activation='relu'))
clas.add(Dense(output_dim=256,activation='relu'))
clas.add(Dense(output_dim=256,activation='relu'))
clas.add(Dense(output_dim=512,activation='relu'))
clas.add(Dense(output_dim=256,activation='relu'))
clas.add(Dense(output_dim=256,activation='relu'))
clas.add(Dense(output_dim=128,activation='relu'))

clas.add(Dense(output_dim=1,activation='sigmoid'))


clas.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set =train_datagen.flow_from_directory('cnn/train',
                                                target_size=(64,64),
                                                batch_size=1,
                                                class_mode='binary')

test_set =test_datagen.flow_from_directory('cnn/test',
                                                target_size=(64,64),
                                                batch_size=1,
                                                class_mode='binary')

clas.fit_generator(training_set,
                   samples_per_epoch=8000,
                   nb_epoch=100,
                   validation_data=test_set,
                   nb_val_samples=2000)


import numpy as np
import pandas as pd



test_set.reset()

pred=clas.predict_generator(test_set,verbose=1)

pred[pred >.5]=1
pred[pred <=.5]=0


test_labels=[]

for i in range(0,int(200)):
    test_labels.extend(np.array(test_set[i][1]))
    
    
    
dosyaisimleri= test_set.filenames
sonuc =pd.DataFrame()

sonuc['dosyaisimleri'] = dosyaisimleri

sonuc['tahminler'] = pred

sonuc['test']= test_labels


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(test_labels,pred)
print(cm)





























