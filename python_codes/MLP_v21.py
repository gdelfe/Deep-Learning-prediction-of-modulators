#!/usr/bin/env python

import sys
sys.modules[__name__].__dict__.clear()

#%%
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from random import shuffle
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable

import import_data_sets as data
import ANN_Models as ANN

#%%
# ## DATA 

# PATH DIRECTORY 
Sess = 15
Ch = 34
# maxCh = 20
# format image (resolution):
x_size = 46
y_size = 30

train_s = 0.75
test_s = 1 - train_s

rand_max = 100
brain_area = 'OFC'
# LOAD DATA
# tensor_hit_train, tensor_hit_test, tensor_miss_train, tensor_miss_test = data.load_data_NW(Sess,Ch,x_size,y_size,train_s,test_s)
# tensor_hit_train, tensor_hit_test, tensor_miss_train, tensor_miss_test = data.load_data_area(Sess,brain_area, x_size, y_size, train_s, test_s)
# tensor_hit_trainA, tensor_hit_testA, tensor_miss_trainA, tensor_miss_testA = data.load_data_random_NW(Sess,Ch,x_size,y_size,train_s,test_s,rand_max)
# data.load_data_homogeneous(Sess,Ch,x_size,y_size,train_s,test_s)

tensor_hit_train_tmp, tensor_hit_test_tmp, tensor_miss_train_tmp, tensor_miss_test_tmp = data.load_data_random_NW(Sess,Ch,x_size,y_size,train_s,test_s,rand_max)

#%%
print(tensor_hit_train_tmp.shape)
print(tensor_hit_test_tmp.shape)
print(tensor_miss_train_tmp.shape)
print(tensor_miss_test_tmp.shape)

#%%
# ### Balance the data set
tensor_miss_train_tmp = tensor_miss_train_tmp[1:]
tensor_miss_test_tmp = tensor_miss_test_tmp[1:]

tensor_hit_train_tmp = tensor_hit_train_tmp[1:]
tensor_hit_test_tmp = tensor_hit_test_tmp[1:]

# tensor_hit_train = tensor_hit_train[1:tensor_miss_train.shape[0]+1]
# tensor_hit_test = tensor_hit_test[1:tensor_miss_test.shape[0]+1]


print('Data set size after balancing')
print(tensor_hit_train_tmp.shape)
print(tensor_hit_test_tmp.shape)
print(tensor_miss_train_tmp.shape)
print(tensor_miss_test_tmp.shape)

#%%
plt.imshow(tensor_hit_train_tmp[10].transpose(),origin='lower')
plt.show()
plt.imshow(tensor_hit_train_tmp[0,:,5:15].transpose(),origin='lower')
plt.show()


#%%
fmin = 0
fmax = 30

#%%
tensor_miss_train = tensor_miss_train_tmp[:,:,fmin:fmax]
tensor_miss_test = tensor_miss_test_tmp[:,:,fmin:fmax]

tensor_hit_train = tensor_hit_train_tmp[:,:,fmin:fmax]
tensor_hit_test = tensor_hit_test_tmp[:,:,fmin:fmax]

print(tensor_hit_train.shape)
print(tensor_hit_test.shape)
print(tensor_miss_train.shape)
print(tensor_miss_test.shape)

#%%
# ### Generate the labels
# Generate the labels for hits and misses and stack them into a single array


labels_hit_train = np.ones(tensor_hit_train.shape[0],dtype='l')
labels_miss_train = np.zeros(tensor_miss_train.shape[0],dtype='l')

labels_hit_test = np.ones(tensor_hit_test.shape[0],dtype='l')
labels_miss_test = np.zeros(tensor_miss_test.shape[0],dtype='l')

# Total labels
labels_tot_train = np.concatenate((labels_hit_train,labels_miss_train),axis=None)
labels_tot_test = np.concatenate((labels_hit_test,labels_miss_test),axis=None)
# print(labels_tot)
print('Labels')
print(labels_tot_train.shape)
print(labels_tot_test.shape)


# ### Merge all the hits and misses matrices together

tensor_train = np.concatenate((tensor_hit_train,tensor_miss_train),axis=0)
print('Total train')
print(tensor_train.shape)
tensor_test = np.concatenate((tensor_hit_test,tensor_miss_test),axis=0)
print('Total test')
print(tensor_test.shape)

#%%

################################################
# ### Normalize inputs
################################################

for indx in range(tensor_train.shape[0]):
    mean = np.mean(tensor_train[indx,:,:])
    std = np.std(tensor_train[indx,:,:,])
    tensor_train[indx,:,:,] = (tensor_train[indx,:,:,] - mean)/std
    
for indx in range(tensor_test.shape[0]):
    mean = np.mean(tensor_test[indx,:,:])
    std = np.std(tensor_test[indx,:,:,])
    tensor_test[indx,:,:,] = (tensor_test[indx,:,:,] - mean)/std


plt.imshow(tensor_test[10,:,:].transpose(),origin='lower')

#%%
################################################
# ## Load data into a trainloader and testloader
################################################


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


TOT_data_train = []
TOT_data_test = []

# concatenate data and labels
for i in range(len(labels_tot_train)):
    TOT_data_train.append([tensor_train[i,:,:], labels_tot_train[i]])
    
for i in range(len(labels_tot_test)):
    TOT_data_test.append([tensor_test[i,:,:], labels_tot_test[i]])

# # shuffle the training data
TOT_data_train = [TOT_data_train[i] for i in range(tensor_train.shape[0])]
shuffle(TOT_data_train)

# # shuffle the test data
TOT_data_test = [TOT_data_test[i] for i in range(tensor_test.shape[0])]
shuffle(TOT_data_test)

# Create datasets
train_dataset = CustomDataset(TOT_data_train)
test_dataset = CustomDataset(TOT_data_test)

# create train and test loaders
trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=50)
testloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=50)

len(TOT_data_train)
len(TOT_data_test)
# print(TOT_data_train)

#%%

################################################
# ### Create iterable as a test 
################################################

# The batch_size decides how many images in the batch during the training or test.
# The numb of iteration in each train (test) loader are: tot length of train (test) data / batch_size. So if the tot number of data is 2000 and the batch size is 50, the iteration on the dataloader is done 40 times, each time 50 images are loaded 

batch_size = 2
images, labels = next(iter(testloader))
images.shape

for b in range(batch_size):
    plt.imshow(images[b].numpy().transpose(),origin='lower')
    plt.show()
    print('Batch: ',b)
    print('Label: ',labels[b].item())
    print()
print(labels.shape)
print(images.shape)
print(labels[0])


print(images.shape)


images.view(images.shape[0],-1).shape
images.shape[1]


#%%
# +++++++++++++++++++++++++++++++++++++++++++++
################################################
# # MODELS
################################################

#%%
# ### Linear regression

model = ANN.Linear_Regression(tensor_hit_train.shape[1],tensor_hit_train.shape[2],1)
print(model)


#%%
# ### Multilayer perceptron (with dropout)

model = ANN.MLP(tensor_hit_train.shape[1],tensor_hit_train.shape[2],0.2)
print(model)

#%%
# ### Udacity multilayer perceptron 

model = ANN.Classifier()
print(model)


#%%
# ## CONVOLUTIONAL NETWORKS

model = ANN.CNN()
print(model)


#%%
# ## CONVOLUTIONAL NETWORKS

model = ANN.CNN_small()
print(model)

#%%
################################################
# ### Set loss function and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.2)


################################################
# #### Move model on GPU if available 

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
    print('Moving model on GPU...')
next(model.parameters()).is_cuda
train_on_gpu

################################################
# ### Define accuracy

def binary_acc(logits, labels):
    y_pred_tag = torch.round(torch.sigmoid(logits)) # make probability out of logits and then round to 0 or 1

    correct_results_sum = (y_pred_tag == labels).sum().float()
    acc = correct_results_sum/labels.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


print('Train dataset length: ',len(train_dataset))
print('Test dataset length: ',len(test_dataset))
print('Trainloader length: ', len(trainloader))
print('Testloader length: ', len(testloader))


#%%
################################################
# ## Reset the weights to random numbers

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

torch.manual_seed(90)
model.apply(weights_init)


next(model.parameters()).is_cuda
print(model)


#%%
################################################
# ## Train and validate the Network 
################################################
# Training and Testing is done on the fly:
# For each epoch the model is trained (the loss is computed by using the training set) and it is tested (the loss and accuracy are computed by using the test set), in order to compare training loss with testing loss and decide when to stop the training and avoid overfitting. 


#######################
#     TRAINING  
######################

model.train() # set the network in training mode

epochs = 10
train_losses, test_losses, acc_list = [], [], []

for epoch in range(1,epochs+1):
    running_loss = 0
    for images, labels in trainloader:
        
        #         labels = labels.long() # change label type from int to long 
        images = images.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor) # convert labels to Float for BCELoss

        
        if train_on_gpu: # move data on GPU
            images, labels = images.cuda(), labels.cuda()
              
        # Clear the gradients
        optimizer.zero_grad()
        
        logits, conv_x1, conv_x2 = model(images) # forward pass
        # logits = model(images) # forward pass
        loss = criterion(torch.squeeze(logits),labels) # compute the loss
        loss.backward() # backpropagate to compute the gradients
        optimizer.step() # update the weights
        
        running_loss =+ loss.item()
    
    else:
        
        #######################
        #     VALIDATION 
        ######################
 
        test_loss = 0
        accuracy = 0
        epoch_acc = 0
        with torch.no_grad(): # set the tracing of gradients to zero
            model.eval() # set the dropout to OFF, i.e. model is in evaluation mode
            

            for images, labels in testloader:
                
#                 for b in range(batch_size):
#                     plt.imshow(np.flipud(images[b].numpy().transpose()))
#                     plt.show()
                images = images.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor) # convert labels to Float for BCELoss    
#                 labels = labels.long() # change label type from int to long 
    
                if train_on_gpu: # move data on GPU
                    images, labels = images.cuda(), labels.cuda()
                

                
#                 print('\n\n************ New batch....\n')
                logits, conv1, conv2 = model(images)
                # logits = model(images)
                logits = torch.squeeze(logits)
                print(logits.shape,labels.shape)
                valid_loss = criterion(logits,labels)
                test_loss += valid_loss.item()    

                y_pred_tag = torch.round(torch.sigmoid(logits)) # make probability out of logits and then round to 0 or 1

#                 print('y_pred : \n',y_pred_tag)
#                 print()
#                 print('labels: ',labels)

#                 print('Test loss: ',test_loss)

#                 print()
#                 print('top_p: \n',top_p)
#                 print('top_class :\n',torch.transpose(top_class,1,0))
#                 print()
#                 
                
#   
                acc = binary_acc(logits,labels) # accuracy for each batch
#                 print('accuracy',acc)
                epoch_acc += acc.item() # sum up the accuracies across batches

                
        model.train() # set the model back to train mode, i.e. dropout is ON
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        acc_list.append(epoch_acc/len(testloader))
            
        accuracy = 0
        ############################
        # PRINT ACCURACTY AND LOSSES
        ############################
#         if epoch % 10 == 0:
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.7f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.5f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(epoch_acc/len(testloader)))
        
        ep = np.arange(1,epoch+1)

        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(13,4))
        # ax = plt.subplot(111)
        ax1.plot(ep,train_losses,'r^--',linewidth=2,label = 'Train loss')
        ax1.set_xlabel('epochs',fontsize=12)
        ax1.set_ylabel('Loss',fontsize=13)
        ax1.set_title('Train loss',fontsize=16)
        ax1.legend()
        ax1.grid(True)

        ax2.plot(ep,test_losses,'go--',linewidth=2,label = 'Test loss')
        ax2.set_xlabel('epochs',fontsize=12)
        ax2.set_title('Test loss',fontsize=16)
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(ep,acc_list,'bo--',linewidth=2,label = 'Accuracy')
        ax3.set_xlabel('epochs',fontsize=12)
        ax3.set_title('Accuracy',fontsize=16)
        ax3.legend()
        ax3.grid(True)

        plt.show()


#%%
################################################
# ### Plot Train loss, Test loss, and Accuracy
################################################

filter1 = 4
filter2 = 4

ep_max = 110
ep = np.arange(1,ep_max)

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(13,4))
# ax = plt.subplot(111)
my_suptitle = fig.suptitle('MLP - Ch34, 100 random proj, size=47*30, f=[0,60]Hz', fontsize=15,y=1.05)
ax1.plot(ep[:ep_max],train_losses[1:ep_max],'r^--',linewidth=2,label = 'Train loss')
ax1.set_xlabel('epochs',fontsize=12)
ax1.set_ylabel('Loss',fontsize=13)
ax1.set_title('Train loss',fontsize=16)
ax1.legend()
ax1.grid(True)

ax2.plot(ep[:ep_max],test_losses[1:ep_max],'go--',linewidth=2,label = 'Test loss')
ax2.set_xlabel('epochs',fontsize=12)
ax2.set_title('Test loss',fontsize=16)
ax2.legend()
ax2.grid(True)

ax3.plot(ep[:ep_max],acc_list[1:ep_max],'bo--',linewidth=2,label = 'Accuracy')
ax3.set_xlabel('epochs',fontsize=12)
ax3.set_title('Accuracy',fontsize=16)
ax3.legend()
ax3.grid(True)


plt.show()

# fig.savefig('Plots/random_N_015_W_15_dn_018/LR_1Ch_100_rand_f_{}_{}_size_47_30.png'.format(fmin,fmax),bbox_inches='tight',bbox_extra_artists=[my_suptitle])
fig.savefig('Plots/random_N_015_W_15_dn_018/MLP_1Ch_100_rand_f_{}_{}_size_47_30.png'.format(fmin,fmax),bbox_inches='tight',bbox_extra_artists=[my_suptitle])
# fig.savefig('Plots/random_N_015_W_15_dn_018/CNN_1Ch_100_rand_f_{}_{}_size_47_30_{}_{}_filters.png'.format(fmin,fmax,filter1,filter2),bbox_inches='tight',bbox_extra_artists=[my_suptitle])
plt.close(fig)



model


#%%
# =============================================================================
#  FILTERS CNN
# =============================================================================
filter1 = 4
filter2 = 4
Extent = [0 , 46, fmin , fmax]
for i in range(filter1):
    fig, ax = plt.subplots()
    ax.set_title('Filter {} - conv layer 1'.format(i+1),fontsize=14)
    img1 = ax.imshow(torch.detach(conv_x1[1,i,:,:]).cpu().numpy().transpose(),origin='lower')
    fig.colorbar(img1, ax=ax,orientation="horizontal")
    ax1.set_aspect('auto')
    plt.show()
    # fig.savefig('Plots/NW_46_30/CNN_100rand_f_{}_{}_Filter_{}_conv1_{}x{}_filters_size_46_10.png'.format(fmin,fmax,i+1,filter1,filter2),bbox_inches='tight')
    # plt.close(fig)
   
    
for i in range(filter2):
    fig, ax = plt.subplots()
    ax.set_title('Filter {} - conv layer 2'.format(i+1),fontsize=14)
    img1 = ax.imshow(torch.detach(conv_x2[1,i,:,:]).cpu().numpy().transpose(),origin='lower')
    fig.colorbar(img1, ax=ax,orientation="horizontal")
    ax1.set_aspect('auto')
    plt.show()
    
    # fig.savefig('Plots/NW_46_30/CNN_100rand_f_{}_{}_Filter_{}_conv2_{}x{}_filters_size_46_10.png'.format(fmin,fmax,i+1,filter1,filter2),bbox_inches='tight')
    # plt.close(fig)



#%%
# =============================================================================
# # Weights for linear regression
# =============================================================================

w = model.fc1.weight.view(tensor_hit_train.shape[1],tensor_hit_train.shape[2])
w = w.detach().cpu()

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('time',fontsize=13)
ax.set_ylabel('frequency',fontsize=12)
Extent = [0 , 46, 10 , 30]
# ax.set_title('Weights - Linear Regression - Ch = {}, L2= 0.15'.format(Ch),fontsize=14)
ax.set_title('Weights -Linear Regression, 1Ch, 100 rand, f=[10,30]Hz'.format(Ch),fontsize=14)
plt.imshow(w.numpy().transpose(),origin='lower',extent=Extent)

# plt.gca().invert_yaxis()
# plt.grid(True)
plt.show()

# fig.savefig('Plots/NW_46_30/weights_LR_1Ch_100_rand_f_{}_{}_orig.png'.format(fmin,fmax),bbox_inches='tight',bbox_extra_artists=[my_suptitle])
# # fig.savefig('Plots/NW_46_30/weights_LR_1Ch_100rand_f_10_30_NW_46_30.png',bbox_inches='tight')
# plt.close(fig)





