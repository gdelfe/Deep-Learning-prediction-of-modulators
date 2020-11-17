#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from random import shuffle
import pylab


# ## DATA 
# ### Load all the HIT (train and test) trials in a given Session (all electrodes)



# PATH DIRECTORY 
Sess = 15
Ch = 34
# maxCh = 20
# format image (resolution):
x_size = 100
y_size = 61

train_s = 0.75
test_s = 1 - train_s

################
# CHANNEL 1 ###
###############

# name directory
pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum'.format(Sess,Ch)
# print(pathHit)
# name files
fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit.txt".format(Sess,Ch,1)) # name file hit (first one)
fnHitLabels = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_Hits_index.txt".format(Sess,Ch)) # name file with labels index
labelsHit = np.loadtxt(fnHitLabels,dtype='l',delimiter='\t') # load labels vector
# labelsHit = np.ones(43)

train_size = (np.floor(len(labelsHit)*train_s)).astype(int) # length trainig set
test_size = len(labelsHit) - train_size  # length test set

############# TRAIN ###############

# load file 1 into hit-matrix and label vector
tensor_hit_train = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
tensor_hit_train = tensor_hit_train.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches

#     #plt.imshow(np.flipud(tensor_hit.transpose()))
# load all other hit-files into a matrix, concatenate it
for indx in range(2,train_size+1):
    fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
    print(fnHit)
    newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
    newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
    tensor_hit_train = np.concatenate((tensor_hit_train,newmat_hit),axis=0) # stack matrix along the 1st dimension 
        
############# TEST ######################        

# load file 1 into hit-matrix and label vector
tensor_hit_test = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
tensor_hit_test = tensor_hit_test.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches

#     #plt.imshow(np.flipud(tensor_hit.transpose()))
# load all other hit-files into a matrix, concatenate it
for indx in range(train_size+2,len(labelsHit)+1):
    fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
    print(fnHit)
    newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
    newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
    tensor_hit_test = np.concatenate((tensor_hit_test,newmat_hit),axis=0) # stack matrix along the 1st dimension 

    

#####################
# ALL OTHER CHANNELS
#####################

channels_list = list(range(1,34)) + list(range(35,53))


for Ch in channels_list:
    
    print('Other Channels')
    
    print('Channel: ',Ch)
    # name directory
    pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)
#     print(pathHit)

    # name file
    fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,1)) # name file hit (first one)
    fnHitLabels = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_Hits_index.txt".format(Ch)) # name file hit (first one)
    labelsHit = np.loadtxt(fnHitLabels,dtype='l',delimiter='\t') # load hit matrix
    
    train_size = (np.floor(len(labelsHit)*train_s)).astype(int) # length trainig set
    test_size = len(labelsHit) - train_size  # length test set
    
    ############# TRAIN ###############
    
    # load all other hit-files into a matrix, concatenate it
    for indx in range(1,train_size+1):
        fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_train = np.concatenate((tensor_hit_train,newmat_hit),axis=0) # stack matrix along the 1st dimension 
        
        
    ############# TEST ######################        

    # load all other hit-files into a matrix, concatenate it
    for indx in range(train_size+2,len(labelsHit)+1):
        fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_test = np.concatenate((tensor_hit_test,newmat_hit),axis=0) # stack matrix along the 1st dimension 


# In[87]:


test_size + train_size


# In[3]:


print(tensor_hit_train.shape)
print(tensor_hit_test.shape)


# ### Load all the MISS trials in a given Session (all electrodes)

# In[33]:


# PATH DIRECTORIES 

################
# CHANNEL 1 ###
###############
Ch = 34
# name directory
pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum'.format(Sess,Ch)
# print(pathMiss)

# name files
fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,1)) # name file miss (first one)
fnMissLabels = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_Misses_index.txt".format(Sess,Ch)) # name file with labels index
labelsMiss = np.loadtxt(fnMissLabels,dtype='l',delimiter='\t') # load labels vector
print(fnMiss)

train_size = (np.floor(len(labelsMiss)*train_s)).astype(int) # length trainig set
test_size = len(labelsMiss) - train_size  # length test set

############# TRAIN ###############

# load miss matrix and label vector
tensor_miss_train = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
tensor_miss_train = tensor_miss_train.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches

# load all other miss-files into a matrix, concatenate it
for indx in range(2,train_size+1):
    fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
    print(fnMiss)
    newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
    newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
    tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension 

############# TEST ######################     

# load file 1 into miss-matrix and label vector
tensor_miss_test = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
tensor_miss_test = tensor_miss_test.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches

# load all other miss-files into a matrix, concatenate it
for indx in range(train_size+2,len(labelsMiss)+1):
    fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
    print(fnMiss)
    newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
    newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
    tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension 

    
#####################
# ALL OTHER CHANNELS
#####################

channels_list = list(range(1,34)) + list(range(35,53))


for Ch in channels_list:
    
    print('Other Channels')
    
    print('Channel: ',Ch)
    # name directory
    pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)
#     print(pathMiss)

    # name file
    fnMiss = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_{}_miss.txt".format(Ch,1)) # name file miss (first one)
    fnMissLabels = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_Misses_index.txt".format(Ch)) # name file miss (first one)
    labelsMiss = np.loadtxt(fnMissLabels,dtype='l',delimiter='\t') # load miss matrix
    
    train_size = (np.floor(len(labelsMiss)*train_s)).astype(int) # length trainig set
    test_size = len(labelsMiss) - train_size  # length test set
    
    ############# TRAIN ###############
    
    # load all other Miss-files into a matrix, concatenate it
    for indx in range(1,train_size+1):
        fnMiss = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_{}_miss.txt".format(Ch,indx)) # name file Miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape 
        tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension 
        
        
    ############# TEST ######################        

    # load all other Miss-files into a matrix, concatenate it
    for indx in range(train_size+2,len(labelsMiss)+1):
        fnMiss = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_{}_miss.txt".format(Ch,indx)) # name file Miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load Miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape 
        tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension 


# In[34]:


print(tensor_miss_train.shape)
print(tensor_miss_test.shape)


# In[35]:


# random inizialization 
# tensor_hit = torch.FloatTensor(858,100,61).uniform_(1,2)
# tensor_hit  = np.random.uniform(8,10,(858,100,61))
# tensor_miss  = np.random.uniform(2,2.001,(858,100,61))

# tensor_hit  = np.ones((858,100,61),dtype=float)
# tensor_miss  = np.ones((858,100,61),dtype=float)
# tensor_miss = 2*tensor_miss


# ### Balance the data set

# In[36]:


# tensor_hit = tensor_hit[0:858]

# balance the data set
tensor_hit_train = tensor_hit_train[0:tensor_miss_train.shape[0]]
tensor_hit_test = tensor_hit_test[0:tensor_miss_test.shape[0]]

print(tensor_hit_train.shape)
print(tensor_hit_test.shape)

print(tensor_miss_train.shape)
print(tensor_miss_test.shape)


# In[37]:


plt.imshow(np.flipud(tensor_hit_train[0].transpose()))


# ### Generate the labels

# Generate the labels for hits and misses and stack them into a single array

# In[38]:


labels_hit_train = np.ones(tensor_hit_train.shape[0],dtype='l')
labels_miss_train = np.zeros(tensor_miss_train.shape[0],dtype='l')

labels_hit_test = np.ones(tensor_hit_test.shape[0],dtype='l')
labels_miss_test = np.zeros(tensor_miss_test.shape[0],dtype='l')

labels_tot_train = np.concatenate((labels_hit_train,labels_miss_train),axis=None)
labels_tot_test = np.concatenate((labels_hit_test,labels_miss_test),axis=None)
# print(labels_tot)
print(labels_tot_train.shape)
print(labels_tot_test.shape)


# ### Merge all the hits and misses matrices together

# In[39]:


# tensor_trial = tensor_hit
# labels_tot = labels_hit
# print(tensor_trial.shape)


# In[47]:


tensor_train = np.concatenate((tensor_hit_train,tensor_miss_train),axis=0)
print(tensor_train.shape)
tensor_test = np.concatenate((tensor_hit_test,tensor_miss_test),axis=0)
print(tensor_test.shape)


# ### Normalize inputs

# In[48]:


for indx in range(tensor_train.shape[0]):
    mean = np.mean(tensor_train[indx,:,:])
    std = np.std(tensor_train[indx,:,:,])
    tensor_train[indx,:,:,] = (tensor_train[indx,:,:,] - mean)/std
    
for indx in range(tensor_test.shape[0]):
    mean = np.mean(tensor_test[indx,:,:])
    std = np.std(tensor_test[indx,:,:,])
    tensor_test[indx,:,:,] = (tensor_test[indx,:,:,] - mean)/std


# In[49]:


plt.imshow(np.flipud(tensor_test[10,:,:].transpose()))


# ## Load data into a trainloader and testloader

# In[50]:


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# In[51]:


TOT_data_train = []
TOT_data_test = []

# concatenate data and labels
for i in range(len(labels_tot_train)):
    TOT_data_train.append([tensor_train[i,:,:], labels_tot_train[i]])
    
for i in range(len(labels_tot_test)):
    TOT_data_test.append([tensor_test[i,:,:], labels_tot_test[i]])

# # shuffle the training data
# TOT_data_train = [TOT_data_train[i] for i in range(tensor_train.shape[0])]
# shuffle(TOT_data_train)

# # shuffle the test data
# TOT_data_test = [TOT_data_test[i] for i in range(tensor_test.shape[0])]
# shuffle(TOT_data_test)

# Create datasets
train_dataset = CustomDataset(TOT_data_train)
test_dataset = CustomDataset(TOT_data_test)

# create train and test loaders
trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=50)
testloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=50)


# In[45]:


len(TOT_data_train)
len(TOT_data_test)
TOT_data_train


# ### Create iterable as a test 

# The batch_size decides how many images in the batch during the training or test.
# The numb of iteration in each train (test) loader are: tot length of train (test) data / batch_size. So if the tot number of data is 2000 and the batch size is 50, the iteration on the dataloader is done 40 times, each time 50 images are loaded 

# In[52]:


batch_size = 5
images, labels = next(iter(testloader))
images.shape

for b in range(batch_size):
    plt.imshow(np.flipud(images[b].numpy().transpose()))
    plt.show()
    print('Batch: ',b)
    print('Label: ',labels[b].item())
    print()
print(labels.shape)
print(images.shape)
print(labels[0])


# In[17]:


print(images.shape)
labels


# In[18]:


images.view(images.shape[0],-1).shape
images.shape[1]


# # MODELS
# ### My multilayer perceptron (with dropout)

# In[115]:


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # linear layers
        self.fc1 = nn.Linear(6100,1500)
        self.fc2 = nn.Linear(1500,750)
        self.fc3 = nn.Linear(750,375)
        self.fc4 = nn.Linear(375,1)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.)    
    
    def forward(self,x):
        
        # flatten the input tensor 
        x = x.view(x.shape[0],-1)
        
        # activation functions
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = self.dropout(F.leaky_relu(self.fc3(x)))
        x = self.fc4(x)
        
#         x = F.relu(self.fc4(x))
#         x = F.sigmoid(self.fc5(x))
        
        return x

model = MLP()
print(model)


# ### Linear regression

# In[53]:


class Linear_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        
        # linear layers
        self.fc1 = nn.Linear(6100,1)
        
        # Dropout module with 0.2 drop probability
#         self.dropout = nn.Dropout(p=0.2)    
    
    def forward(self,x):
        
        # flatten the input tensor 
        x = x.view(x.shape[0],-1)
        
        # activation functions
        x = self.fc1(x)
        
        return x
    
model = Linear_Regression()
model


# ### Udacity multilayer perceptron 

# In[20]:


from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1) # careful because you have two output for a binary classification problem
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)   
   
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)

        return x
    
model = Classifier()
print(model)


# ## CNN

# In[125]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
         
        #######################    
        # convolutional layers
        # ^^^^^^^^^^^^^^^^^^^^
    
        # convolutional layer (sees 100x62x1 image tensor)
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        # convolutional layer (sees 50x32x16 tensor)
        self.conv2 = nn.Conv2d(16,32,3,padding=1,)
        # convolutional layer (sees 26x16x32 tensor)
        self.conv3 = nn.Conv2d(32,64,3,padding=1,)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2,2)
        
        # batch normalization 
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        
#         # fully connected / Linear layers
#         self.fc1 = nn.Linear(64 * 13 * 8, 512)
#         self.fc2 = nn.Linear(512,256)            
#         self.fc3 = nn.Linear(256,1)
        
        # linear layers
        self.fc1 = nn.Linear(64 * 13 * 8,1500)
        self.fc2 = nn.Linear(1500,750)
        self.fc3 = nn.Linear(750,375)
        self.fc4 = nn.Linear(375,1)
        
        # asymmetric padding
        self.pad_R  = nn.ZeroPad2d((0, 1, 0, 0)) # padding right one step
        self.pad_B = nn.ZeroPad2d((0, 0, 1, 0)) # padding bottom one step 
        
        # dropout layer 
        self.dropout = nn.Dropout(0.2)
        
            
    def forward(self,x):
        
        x = x.view(images.shape[0],1,images.shape[1],images.shape[2]) # images is expanded to host the channel (1 in this case, 3 when RGB) dimension for the CNN

        #########################
        # CONVOLUTIONAL LAYERS ##
        #########################
        
        # Convolution 1:
        x = self.pad_R(x) # padding right one step
#         print('x shape conv 1: ',x.shape)
        x = self.conv1(x) # convolutional layer
#         print('x shape conv 1: ',x.shape)
        # batch normalization 
        x = self.bn1(x) 
#         pdb.set_trace()        
        x = F.relu(x)     # activation function ReLU
#         print('x shape conv 1: ',x.shape)
#         pdb.set_trace()
        x = self.pool(x)  # max pooling
#         print('x shape conv 1: ',x.shape)
#         pdb.set_trace()
        
        # Convolution 2:
        x = self.pad_R(x) # padding right one step
#         print('x shape conv 2: ',x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # Convolution 2: in short convolution + relu + pooling  
        
        # Convolution 3:
        x = self.pad_B(x) # padding bottom one step
#         print('x shape conv 3: ',x.shape)
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # Convolution 3: in short convolution + relu + pooling 
        
#         print('x shape before MLP: ',x.shape)

        ########################
        ## LINEAR LAYERS, MLP ##
        ########################
        
        x = x.view(x.shape[0],-1) # flatten image input
        x = self.dropout(x) # dropout
        
        x = F.leaky_relu(self.fc1(x)) # Fully connected 1: linear + relu
        x = self.dropout(x) # dropout
        
        x = F.leaky_relu(self.fc2(x)) # Fully connected 2: linear + relu
        x = self.dropout(x) # dropout
        
        x = F.leaky_relu(self.fc3(x)) # Fully connected 2: linear + relu
#         x = self.dropout(x) # dropout
        
        x = self.fc4(x) # Fully connected 3: linear 
#         x = F.sigmoid(x)
        
        return x 

model = CNN()
print(model)


# ### Set loss function and optimizer

# In[54]:


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.15)


# #### Move model on GPU if available 

# In[55]:


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


# ### Define accuracy

# In[56]:


def binary_acc(logits, labels):
    y_pred_tag = torch.round(torch.sigmoid(logits)) # make probability out of logits and then round to 0 or 1

    correct_results_sum = (y_pred_tag == labels).sum().float()
    acc = correct_results_sum/labels.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


# In[57]:


print('Train dataset length: ',len(train_dataset))
print('Test dataset length: ',len(test_dataset))
print('Trainloader length: ', len(trainloader))
print('Testloader length: ', len(testloader))


# ## Reset the weights to random numbers

# In[58]:


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

torch.manual_seed(42)
model.apply(weights_init)


# In[27]:


next(model.parameters()).is_cuda


# ## Train and validate the Network 

# Training and Testing is done on the fly:
# <br>
# For each epoch the model is trained (the loss is computed by using the training set) and it is tested (the loss and accuracy are computed by using the test set), in order to compare training loss with testing loss and decide when to stop the training and avoid overfitting. 

# In[59]:


#######################
#     TRAINING  
######################

model.train() # set the network in training mode

epochs = 500
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
        
        logits = model(images) # forward pass
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
                logits = model(images)
                logits = torch.squeeze(logits)
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


# In[ ]:





# ### Plot Train loss, Test loss, and Accuracy

# In[169]:


ep_max = 130
ep = np.arange(1,ep_max)

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(13,4))
# ax = plt.subplot(111)
my_suptitle = fig.suptitle('MLP, channel 34, L2 = 0.15', fontsize=18,y=1.05)
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

fig.savefig('Plots/MLP_channel_34_wd=015.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle])
plt.close(fig)


# In[105]:


model


# In[62]:


w = model.fc1.weight.view(100,61)
w = w.detach().cpu()

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('time',fontsize=13)
ax.set_ylabel('frequency',fontsize=12)
# ax.set_title('Weights - Linear Regression - Ch = {}, L2= 0.15'.format(Ch),fontsize=14)
ax.set_title('Weights - Linear Regression - all channels, L2= 0.15'.format(Ch),fontsize=14)
plt.imshow(np.flipud(w.numpy().transpose()))
# plt.gca().invert_yaxis()
# plt.grid(True)
plt.show()

fig.savefig('Plots/weights_LR_all_channels_wd=015.png',bbox_inches='tight')
plt.close(fig)


# In[ ]:




