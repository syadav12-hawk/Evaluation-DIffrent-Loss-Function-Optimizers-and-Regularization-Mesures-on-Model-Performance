# -*- coding: utf-8 -*-
"""
Name: Sourav Yadav
Assignment 3
AID: A20450418
Spring 2020
Deep Learning

"""
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
from sklearn import preprocessing
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


from keras import models
from keras import layers

def build_model(optim,loss):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))#Temp
    model.add(layers.Dense(1))

    model.compile(optimizer=optim,loss=loss,metrics=['mae'])
              
    return model

def k_fold_train(optim,loss,epoch,batch_sz):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = epoch
    
    train_score=[]
    val_score=[]
    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                                [train_data[:i * num_val_samples],
                                train_data[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_targets = np.concatenate(
                                [train_targets[:i * num_val_samples],
                                train_targets[(i+1)*num_val_samples:]],
                                axis=0)
        
        
        #rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
        #loss=losses.mean_squared_error
        
        model = build_model(optim,loss)
        history=model.fit(partial_train_data,
                  partial_train_targets,
                  epochs=num_epochs,
                  batch_size=batch_sz,validation_data=(val_data,val_targets)) #,verbose=0
        train_score.append(history.history['mae'])
        val_score.append(history.history['val_mae'])
    
    
    
    #Compute AVg MAE
    avg_mea_history=[np.mean([x[i] for x in train_score]) for i in range(num_epochs)]    
    avg_val_mea_history=[np.mean([y[j] for y in val_score]) for j in range(num_epochs)] 

    return avg_mea_history,avg_val_mea_history



#Mean Square Error
#-----------------------------------------------------------------------------------
#Defining Optimizer and Loss Function 
rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
loss=losses.mean_squared_error
epoch=75
batch_sz=1

avg_mea_history,avg_val_mea_history=k_fold_train(rmsprop,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae Validation Error")
plt.show()



#Bulding Final Model
model_final=build_model(rmsprop,loss)
model_final.fit(train_data,train_targets,epochs=50,batch_size=1)

#Saving The Final Model
model_final.save("mse.hdf5")

#Loading Saved model
model_new=load_model("mse.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)

print("After evaluating on test data:")
print("Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))




#Mean Absolute Error
#-----------------------------------------------------------------------------------
#Defining Optimizer and Loss Function 
rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
loss=losses.mean_absolute_error
epoch=400
batch_sz=1


avg_mea_history,avg_val_mea_history=k_fold_train(rmsprop,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae Validation Error")
plt.title('Mean Absolute Error')
plt.show()



#Bulding Final Model
model_final=build_model(rmsprop,loss)
model_final.fit(train_data,train_targets,epochs=75,batch_size=1)

#Saving The Final Model
print("Saving the Model...")
model_final.save("mae.hdf5")

#Loading Saved model
print("Loading the Model...")
model_new=load_model("mae.hdf5")

#Evaluating Mean Square Error Loss Function
test_loss,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("Loss Value: {}".format(test_loss))
print("Mean Absolute error: {}".format(test_mae))





#Mean Absolute Percentage Error
#-----------------------------------------------------------------------------------
#Defining Optimizer and Loss Function 
rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
loss=losses.mean_absolute_percentage_error
epoch=100
batch_sz=1


avg_mea_history,avg_val_mea_history=k_fold_train(rmsprop,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae Validation Error")
plt.title('Mean Absolute Percentage Error')
plt.show()


#Bulding Final Model
model_final=build_model(rmsprop,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
print("Saving the Model...")
model_final.save("mape.hdf5")

#Loading Saved model
print("Loading the Model...")
model_new=load_model("mape.hdf5")

#Evaluating Mean Square Error Loss Function
test_loss,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("Loss Value: {}".format(test_loss))
print("Mean Absolute error: {}".format(test_mae))





#Mean Square Logarithmic Error
#-----------------------------------------------------------------------------------
#Defining Optimizer and Loss Function 
rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
loss=losses.mean_squared_logarithmic_error
epoch=75
batch_sz=1

avg_mea_history,avg_val_mea_history=k_fold_train(rmsprop,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae Validation Error")
plt.title('Mean Square Logarithmic Error')
plt.show()


#Bulding Final Model
model_final=build_model(rmsprop,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
print("Saving the Model...")
model_final.save("msle.hdf5")

#Loading Saved model
print("Loading the Model...")
model_new=load_model("msle.hdf5")

#Evaluating Mean Square Error Loss Function
test_loss,test_mae=model_new.evaluate(test_data, test_targets)

print("After evaluating on test data:")
print("Loss Value: {}".format(test_loss))
print("Mean Absolute error: {}".format(test_mae))




#Evaluating Diffrent Optimizers with MSE loss
#-----------------------------------------------------------------------------------
#SGD Optimizer
#-----------------------------------------------------------------------------------


sgd=keras.optimizers.SGD(learning_rate=0.001,momentum=0.0,nesterov=False)

loss=losses.mean_squared_error
epoch=75
batch_sz=1


avg_mea_history,avg_val_mea_history=k_fold_train(sgd,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.title('SGD Optimizer')
plt.show()



#Bulding Final Model
model_final=build_model(sgd,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("sgd.hdf5")

#Loading Saved model
model_new=load_model("sgd.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))




#RMS Prop 
#-------------------------------------------------------------------

rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

loss=losses.mean_squared_error
epoch=50
batch_sz=1


avg_mea_history,avg_val_mea_history=k_fold_train(rmsprop,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.title('RMSProp Optimizer')
plt.show()



#Bulding Final Model
model_final=build_model(rmsprop,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("rmsprop.hdf5")

#Loading Saved model
model_new=load_model("rmsprop.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))



#AdaGrad 
#-------------------------------------------------------------------

#rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
adagrad=keras.optimizers.Adagrad(learning_rate=0.1,epsilon=0.5,decay=0.0) 
loss=losses.mean_squared_error
epoch=75
batch_sz=8


avg_mea_history,avg_val_mea_history=k_fold_train(adagrad,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.title('AdaGrad Optimizer')
plt.show()



#Bulding Final Model
model_final=build_model(adagrad,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("adagrad.hdf5")

#Loading Saved model
model_new=load_model("adagrad.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))



#AdaDelta
#--------------------------------------------------------------------------

adadelta=keras.optimizers.Adadelta(learning_rate=0.01,rho=0.95,epsilon=0.5,decay=0.0)
loss=losses.mean_squared_error
epoch=150
batch_sz=8


avg_mea_history,avg_val_mea_history=k_fold_train(adadelta,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,4])
plt.title('AdaDelta Optimizer')
plt.show()


#Bulding Final Model
model_final=build_model(adadelta,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("adadelta.hdf5")

#Loading Saved model
model_new=load_model("adadelta.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))



#Adam
#------------------------------------------------------------------------------
adam=keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.5,decay=0.0,amsgrad=False)

loss=losses.mean_squared_error
epoch=100
batch_sz=8


avg_mea_history,avg_val_mea_history=k_fold_train(adam,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,4])
plt.title('Adam Optimizer')
plt.show()



#Bulding Final Model
model_final=build_model(adam,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("adam.hdf5")

#Loading Saved model
model_new=load_model("adam.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))



#AdaMax
#-----------------------------------------------------------------------------

adamax=keras.optimizers.Adamax(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.1 ,decay=0.0)
loss=losses.mean_squared_error
epoch=50
batch_sz=8

avg_mea_history,avg_val_mea_history=k_fold_train(adamax,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,3.5])
plt.title('AdaMax Optimizer')
plt.show()


#Bulding Final Model
model_final=build_model(adamax,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("adamax.hdf5")

#Loading Saved model
model_new=load_model("adamax.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))





#Nadam
#-------------------------------------------------------------------------------

nadam=keras.optimizers.Nadam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.1,schedule_decay=0.004)
loss=losses.mean_squared_error
epoch=50
batch_sz=8


avg_mea_history,avg_val_mea_history=k_fold_train(nadam,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,3])
plt.title('Nadam Optimizer')
plt.show()


#Bulding Final Model
model_final=build_model(nadam,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("nadam.hdf5")

#Loading Saved model
model_new=load_model("nadam.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))




#Evaluating Diffrent reguralization Measures
#Weight Decay on RMSProp
#------------------------------------------------------------------------------------------

def build_model_wt_decay(optim,loss):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],),kernel_regularizer=regularizers.l2(0.05)))
    model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.05)))
    model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.05)))#Temp
    model.add(layers.Dense(1))

    model.compile(optimizer=optim,loss=loss,metrics=['mae'])
              
    return model

def k_fold_train_wt_decay(optim,loss,epoch,batch_sz):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = epoch
    
    train_score=[]
    val_score=[]
    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                                [train_data[:i * num_val_samples],
                                train_data[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_targets = np.concatenate(
                                [train_targets[:i * num_val_samples],
                                train_targets[(i+1)*num_val_samples:]],
                                axis=0)
        
        
        #rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
        #loss=losses.mean_squared_error
        
        model = build_model_wt_decay(optim,loss)
        history=model.fit(partial_train_data,
                  partial_train_targets,
                  epochs=num_epochs,
                  batch_size=batch_sz,validation_data=(val_data,val_targets)) #,verbose=0
        train_score.append(history.history['mae'])
        val_score.append(history.history['val_mae'])
    
    
    
    #Compute AVg MAE
    avg_mea_history=[np.mean([x[i] for x in train_score]) for i in range(num_epochs)]    
    avg_val_mea_history=[np.mean([y[j] for y in val_score]) for j in range(num_epochs)] 

    return avg_mea_history,avg_val_mea_history




rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

loss=losses.mean_squared_error
epoch=100
batch_sz=8


avg_mea_history,avg_val_mea_history=k_fold_train_wt_decay(rmsprop,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,4])
plt.title('Weight Decay Regularization')
plt.show()



#Bulding Final Model
model_final=build_model(rmsprop,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("rmsprop_wt_decay.hdf5")

#Loading Saved model
model_new=load_model("rmsprop_wt_decay.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))



#Drop Out Regularization
#----------------------------------------------------------------------------------------


def build_model_dropout(optim,loss):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))#Temp
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))

    model.compile(optimizer=optim,loss=loss,metrics=['mae'])
              
    return model

def k_fold_train_dropout(optim,loss,epoch,batch_sz):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = epoch
    
    train_score=[]
    val_score=[]
    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                                [train_data[:i * num_val_samples],
                                train_data[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_targets = np.concatenate(
                                [train_targets[:i * num_val_samples],
                                train_targets[(i+1)*num_val_samples:]],
                                axis=0)
        
        
        #rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
        #loss=losses.mean_squared_error
        
        model = build_model_dropout(optim,loss)
        history=model.fit(partial_train_data,
                  partial_train_targets,
                  epochs=num_epochs,
                  batch_size=batch_sz,validation_data=(val_data,val_targets)) #,verbose=0
        train_score.append(history.history['mae'])
        val_score.append(history.history['val_mae'])
    
    
    
    #Compute AVg MAE
    avg_mea_history=[np.mean([x[i] for x in train_score]) for i in range(num_epochs)]    
    avg_val_mea_history=[np.mean([y[j] for y in val_score]) for j in range(num_epochs)] 

    return avg_mea_history,avg_val_mea_history




rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
loss=losses.mean_squared_error
epoch=200
batch_sz=8


avg_mea_history,avg_val_mea_history=k_fold_train_dropout(rmsprop,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,4])
plt.title('DropOut Regularization')
plt.show()



#Bulding Final Model
model_final=build_model(rmsprop,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("rmsprop_dropout.hdf5")

#Loading Saved model
model_new=load_model("rmsprop_dropout.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))





#Batch Normalization:
#-------------------------------------------------------------
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation


def build_model_batch_norm(optim,loss):    
    model = models.Sequential()
    
    #Layer1
    model.add(layers.Dense(64, input_shape=(train_data.shape[1],), init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Layer2
    model.add(layers.Dense(64, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
            
    # Layer3
    model.add(layers.Dense(64, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Output Layer
    model.add(layers.Dense(1, init='uniform'))
    
    model.compile(optimizer=optim,loss=loss,metrics=['mae'])
              
    return model

def k_fold_train_batch_norm(optim,loss,epoch,batch_sz):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = epoch
    
    train_score=[]
    val_score=[]
    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                                [train_data[:i * num_val_samples],
                                train_data[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_targets = np.concatenate(
                                [train_targets[:i * num_val_samples],
                                train_targets[(i+1)*num_val_samples:]],
                                axis=0)
        
        
        #rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
        #loss=losses.mean_squared_error
        
        model = build_model_batch_norm(optim,loss)
        history=model.fit(partial_train_data,
                  partial_train_targets,
                  epochs=num_epochs,
                  batch_size=batch_sz,validation_data=(val_data,val_targets)) #,verbose=0
        train_score.append(history.history['mae'])
        val_score.append(history.history['val_mae'])
    
    
    
    #Compute AVg MAE
    avg_mea_history=[np.mean([x[i] for x in train_score]) for i in range(num_epochs)]    
    avg_val_mea_history=[np.mean([y[j] for y in val_score]) for j in range(num_epochs)] 

    return avg_mea_history,avg_val_mea_history




rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
loss=losses.mean_squared_error
epoch=100
batch_sz=8


avg_mea_history,avg_val_mea_history=k_fold_train_batch_norm(rmsprop,loss,epoch,batch_sz)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,7])
plt.title('Batch Normalization Regularization')
plt.show()


#Bulding Final Model
model_final=build_model(rmsprop,loss)
model_final.fit(train_data,train_targets,epochs=epoch,batch_size=batch_sz)

#Saving The Final Model
model_final.save("rmsprop_bnorm.hdf5")

#Loading Saved model
model_new=load_model("rmsprop_bnorm.hdf5")

#Evaluating Mean Square Error Loss Function
test_mse,test_mae=model_new.evaluate(test_data, test_targets)

print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mse))
print("Mean Absolute error: {}".format(test_mae))



#Ensemble Classifier using two models : Adam Classifier and RMSProp Classifier
#-------------------------------------------------------------------------------
#Loading Models

model_new1=load_model("adam.hdf5")
model_new2=load_model("rmsprop.hdf5")

#Getting Predicated Values from Two models
predicted_y1=model_new1.predict(test_data)
predicted_y2=model_new2.predict(test_data)

#Average Predicted Values
avg_predicted_y=(predicted_y1+predicted_y2)/2.0

tt=test_targets.reshape((102, 1))
mae = np.sum(np.absolute(tt-avg_predicted_y))/test_targets.shape[0]

print("Mean Absolute Error {}".format(mae))
