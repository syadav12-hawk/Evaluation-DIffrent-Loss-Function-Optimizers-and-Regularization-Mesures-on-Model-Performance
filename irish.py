# -*- coding: utf-8 -*-
"""
Name: Sourav Yadav
Assignment 3
AID: A20450418
Spring 2020
Deep Learning

"""

#Importing Packages and Lib
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import losses
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
from keras import regularizers


#Load the data
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data'
iris_data = pd.read_csv(url, error_bad_lines=False,header=None)

#Extracting Features and Labels 
features = iris_data.iloc[:,:-1]
label = iris_data.iloc[:,-1:]

#One Hot encoding of the labels 
encoder = OneHotEncoder()
label=np.reshape(label, (label.shape[0], 1))
label= encoder.fit_transform(label).toarray()

#Spliting of the data into train and test data. 
x_strain,x_test,y_strain,y_test = train_test_split(features, label,test_size = 0.2)

#Normalizing the data
train_mean = x_strain.mean()
train_std = x_strain.std()
x_strain = ((x_strain - train_mean )/train_std)
x_test = ((x_test - train_mean)/train_std)

#Spliting data into Train and Validation data.
x_train,x_val,y_train,y_val = train_test_split(x_strain, y_strain,test_size = 0.2)


def neural_model_plot(history_dict,name):  
    plt.clf()
    #Loss Plot
    loss_values=history_dict['loss']
    val_loss_values=history_dict['val_loss']
    epochs=range(1,len(loss_values)+1)
    plt.plot(epochs,loss_values,'bo',label='Training Loss')
    plt.plot(epochs,val_loss_values,'b',label='Validation Loss')
    plt.title(name+': Traning and Validation Loss ')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    #Clear Previous Plots
    plt.clf()
    
    #Accuracy Plot
    acc_values=history_dict['accuracy']
    val_acc_values=history_dict['val_accuracy']
    plt.plot(epochs,acc_values,'bo',label='Training Accuracy')
    plt.plot(epochs,val_acc_values,'b',label='Validation Accuracy')
    plt.title(name+' :Traning and Validation Accuracy ')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()



#Evaluating Diffrent Loss Functiuons
#Categorical_Crossentropy
#--------------------------------------------------------------------
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=rmsprop,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=20,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("ccross_entropy.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='Categorical Cross Entropy Loss')

#Loading saved Models
print("Loading Saved Model")
model_new=load_model("ccross_entropy.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))




#Kullback_leibler_divergence Loss
#-------------------------------------------------------------------------------------
#rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
#neural_model(rmsprop,losses.kullback_leibler_divergence,epoch=200,batch_size=1,name='kl_divergence.hdf5')

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=rmsprop,loss=losses.kullback_leibler_divergence,metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=40,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("kl_divergence.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='KL Divergence Loss')

#Loading saved Models
print("Loading Saved Model")
model_new=load_model("kl_divergence.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))





#Hinge Losses
#------------------------------------------------------------------------------
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(16,activation='relu'))
#model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=rmsprop,loss=losses.hinge,metrics=['accuracy'])
#neural_model(model,epoch=200,batch_size=1,name='ccross_entropy.hdf5')
history=model.fit(x_train,y_train,epochs=25,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("hinge_loss.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='Hinge Loss')

#Loading saved Models
print("Loading Saved Model")
model_new=load_model("hinge_loss.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))



#Sqaured Hinge Loss
#--------------------------------------------------------------

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(16,activation='relu'))
#model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=rmsprop,loss=losses.squared_hinge,metrics=['accuracy'])
#neural_model(model,epoch=200,batch_size=1,name='ccross_entropy.hdf5')
history=model.fit(x_train,y_train,epochs=40,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("sq_hinge_loss.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='Squared Hinge Loss')

#Loading saved Models
print("Loading Saved Model")
model_new=load_model("sq_hinge_loss.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))



#losses.categorical_crossentropy,losses.kullback_leibler_divergence,losses.hinge,losses.squared_hinge


#Evaluatinh Diffrent Optimers
#--------------------------------------------------------------------------
#Evaluating Diffrent Optimizers on the Irish Data Set




#SGD Optimizers
#---------------------------------------------------------------------
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(16,activation='relu'))
#model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))

sgd=keras.optimizers.SGD(learning_rate=0.001,momentum=0,nesterov=False)
model.compile(optimizer=sgd,loss=losses.categorical_crossentropy,metrics=['accuracy'])
#neural_model(model,epoch=200,batch_size=1,name='ccross_entropy.hdf5')
history=model.fit(x_train,y_train,epochs=150,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("sgd_optimizer.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='SGD Optimizer')


#Loading saved Models
print("Loading Saved Model")
model_new=load_model("sgd_optimizer.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))

#RMS Prop 
#-------------------------------------------------------------------
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=rmsprop,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("rmsprop.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='RMS Prop')


#Loading saved Models
print("Loading Saved Model")
model_new=load_model("rmsprop.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))




#AdaGrad
#-------------------------------------------------------------------
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
adagrad=keras.optimizers.Adagrad(learning_rate=0.01,epsilon=0.5,decay=0.0) 
model.compile(optimizer=adagrad,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("adagrad.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='AdaGrad')


#Loading saved Models
print("Loading Saved Model")
model_new=load_model("adagrad.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))




#AdaDelta
#----------------------------------------------------------------------------
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
adadelta=keras.optimizers.Adadelta(learning_rate=0.001,rho=0.95,epsilon=0.5,decay=0.0) 
model.compile(optimizer=adadelta,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=300,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("adadelta.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='AdaDelta')


#Loading saved Models
print("Loading Saved Model")
model_new=load_model("adadelta.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))





#Adam
#----------------------------------------------------------------------------
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
adam=keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=0.5,decay=0.0,amsgrad=False)
model.compile(optimizer=adam,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=75,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("adam.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='Adam')


#Loading saved Models
print("Loading Saved Model")
model_new=load_model("adam.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))



#AdaMax
#----------------------------------------------------------------------------
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
adamax=keras.optimizers.Adamax(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=0.1 ,decay=0.0)
model.compile(optimizer=adamax,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("adamax.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='AdaMax Optimizer')


#Loading saved Models
print("Loading Saved Model")
model_new=load_model("adamax.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))




#Nadam Optimizer
#--------------------------------------------------------------------
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
nadam=keras.optimizers.Nadam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=0.1,schedule_decay=0.004)
model.compile(optimizer=nadam,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=50,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("nadamax.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='NAdaMax Optimizer')


#Loading saved Models
print("Loading Saved Model")
model_new=load_model("nadamax.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))


#Evaluating Diffrent Reguralization Measures
#-------------------------------------------------------------------------------

#Weight Decay on Adam 
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,),kernel_regularizer=regularizers.l2(0.05)))
model.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.05)))
model.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.05)))
model.add(layers.Dense(3,activation='softmax'))
adam=keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=0.5,decay=0.0,amsgrad=False)
model.compile(optimizer=adam,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=200,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("wtdecay.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='Weight Decay')



#Loading saved Models
print("Loading Saved Model")
model_new=load_model("wtdecay.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))




#Dropout with RMSProp regularizer :
#-------------------------------------------------------------------------------

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(3,activation='softmax'))
rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=rmsprop,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=150,batch_size=1,validation_data=(x_val,y_val))
history_dict=history.history
model.save("dropout.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='Drop Out')


#Loading saved Models
print("Loading Saved Model")
model_new=load_model("dropout.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))





#Batch Normalization with SGD Optimizer Classifer
#--------------------------------------------------------------------------
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout

model = models.Sequential()

#Layer1
model.add(layers.Dense(16, input_dim=4, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.1))

# Layer2
model.add(layers.Dense(16, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.1))

# Output Layer
model.add(layers.Dense(3, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

sgd=keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=75,batch_size=8,validation_data=(x_val,y_val))
history_dict=history.history
model.save("batch_norm.hdf5")
print("Model Saved") 
neural_model_plot(history_dict,name='Batch Norm')

#Loading saved Models
print("Loading Saved Model")
model_new=load_model("batch_norm.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))



#Ensemble Classifier using two models : Adam Classifier and RMSProp Classifier
#----------------------------------------------------------------------------
#Loading Models
model_new1=load_model("adam.hdf5")
model_new2=load_model("rmsprop.hdf5")

#Getting Predicated Values from Two models
predicted_y1=model_new1.predict(x_test)
predicted_y2=model_new2.predict(x_test)

#Average Predicted Values
avg_predicted_y=(predicted_y1+predicted_y2)/2.0

#Predicating Labels
pred_label=[]
for i in avg_predicted_y:
        max_value = np.max(i)
        pred_label.append(list(np.where(i==max_value,1,0)))  
    
#Getting Accuracy 
count = 0
for i,j in zip(y_test,pred_label):
        if ((i[0] == j[0]) & (i[1] == j[1]) & (i[2] == j[2])):
            count +=1

acc = ((count/x_test.shape[0]))
print("Accuracy of Ensemmble Classifier {}".format(acc))