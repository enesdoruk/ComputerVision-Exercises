from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Activation,Reshape
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import UpSampling2D,Conv2D
from tensorflow.python.keras.layers import ELU
from tensorflow.python.keras.layers import Flatten,Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.datasets import mnist
import pickle

NAME =
import os
from PIL import Image
from helper import *

def generator(input_dim =100, units=1024,activation = 'relu'):
    model= Sequential()
    model.add(Dense(input_dim=input_dim,units=units))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Reshape((7,7,128),input_shape=(128*7*7,)))
    #büyütüyorz resmi(14*14*128)
    model.add(UpSampling2D((2,2)))
    #filtreden dolayı(14*14*64)
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    #28?28?64(amaç 28*28*1)
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(1,(5,5),padding='same'))
    model.add(Activation('tanh'))
    print(model.summary())
    return model

def discriminatar(input_shape=(28,28,1),nb_filtre=64):
    model = Sequential()
    #14*14*64
    model.add(Conv2D(nb_filtre,(5,5),strides=(2,2),padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ELU())
    #5*5*128
    model.add(Conv2D(2*nb_filtre,(5,5),strides=(2,2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(4*nb_filtre))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model


batch_size=32
num_epoch=50
learning_rate=0.0002

image_path='image/'
if not os.path.exists(image_path):
    os.mkdir(image_path)

#gans oluştuturken etiketlere ihtiyaç yok onun için _ verdik
def train():
    (x_train,y_train), (_,_) =mnist.load_data()
    x_train=(x_train.astype(np.float32)-127.5)/ 127.5
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)

    g=generator()
    d=discriminatar()
    #discirimiator eğitmiyouz
    optimize= Adam(lr=learning_rate, beta_1=0.5)
    d.trainable=True
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=optimize)
    d.trainable= False
     #d.trainable false verdik onun için sadece genrator eğitiiliyor
    dcgan=Sequential([g,d])
    dcgan.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=optimize)

    num_batches =x_train.shape[0] // batch_size
    gen_img = np.array([np.random.uniform(-1,1,100) for _ in range(49)])
    y_d_true= [1]*batch_size
    y_d_gen= [0]*batch_size
    y_g= [1]*batch_size



    for epoch in range(num_epoch):
        for i in range(num_batches):
            x_d_batch =x_train[i*batch_size:(i+1)*batch_size]
            #gürültü oluşturuyoruz
            x_g= np.array([np.random.normal(0,0.5,100) for i in range(batch_size)])
            x_d_gen=g.predict(x_g)

            d_loss= d.train_on_batch(x_d_batch,y_d_true)
            d_loss= d.train_on_batch(x_d_gen,y_d_gen)

            g_loss =dcgan.train_on_batch(x_g,y_g)
            show_progress(epoch,i,g_loss[0],d_loss[0],g_loss[1],d_loss[1])

        image = combine_images(g.predict(gen_img))
        image =image*127.5+127.5
        #kaydetme
        Image.fromarray(image.astype(np.uint8)).save(image_path + "%03d.png" %(epoch))

#train fonk çağırıyoruz sadece bu dosyayı çalıştırdığımızda
if __name__ == '__main__':
    train()








