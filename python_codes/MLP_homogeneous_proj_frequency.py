#!/usr/bin/env python

import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from random import shuffle

#%%
# ## DATA 
# ### Load all the HIT-training trials in a given Session, one channel

# PATH DIRECTORY 
Sess = 15
Ch = 34

# format image (resolution):
x_size = 100
y_size = 61

train_s = 0.75
test_s = 1 - train_s

fmin = 0
fmax = 10

#%%
###################### 
# FIXED CHANNEL    ###
######################

######################### 
# TRAINING SET HITS    ###
#########################

#>>>>>>>>>>>>>>>>>>
# Frequency 1 : (0,60) Hz
#>>>>>>>>>>>>>>>>>>

# name directory
pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum_fmin_{}'.format(Sess,Ch,fmin)
print(pathHit)
# name file 1
fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,1)) # name file hit (first one)
print(fnHit)

fnHitLabels = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_Hits_index.txt".format(Ch)) # name file with labels index
# labelsHit = np.loadtxt(fnHitLabels,dtype='l',delimiter='\t') # load labels vector
labelsHit = np.ones(43)

train_size = (np.floor(len(labelsHit)*train_s)).astype(int) # length trainig set
test_size = len(labelsHit) - train_size  # length test set


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
    
        
    
# #>>>>>>>>>>>>>>>>>>>>>>>>>>
# # All other frequencies
# #>>>>>>>>>>>>>>>>>>>>>>>>>>    

for freq in range(fmin+1,fmax+1):
    
    pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum_fmin_{}'.format(Sess,Ch,freq) 
    
    print(pathHit)
    for indx in range(1,train_size+1):
        fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_train = np.concatenate((tensor_hit_train,newmat_hit),axis=0) # stack matrix along the 1st dimension 
    


#%%
print(tensor_hit_train.shape)


# ### Load all the HIT-test trials in a given Session, one channel

######################### 
# TEST SET HITS    ###
#########################

#>>>>>>>>>>>>>>>>>>
# Frequency 1 : (0,60) Hz
#>>>>>>>>>>>>>>>>>>

# name directory
pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum_fmin_{}'.format(Sess,Ch,fmin)
print(pathHit)
# name file 1
fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,train_size+1)) # name file hit (first one)
print(fnHit)

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
    
    
# #>>>>>>>>>>>>>>>>>>>>>>>>>>
# # All other frequencies
# #>>>>>>>>>>>>>>>>>>>>>>>>>>    

for freq in range(fmin+1,fmax+1):
    
    pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum_fmin_{}'.format(Sess,Ch,freq) 
    
    print(pathHit)
    for indx in range(train_size+1,len(labelsHit)+1):
        fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_test = np.concatenate((tensor_hit_test,newmat_hit),axis=0) # stack matrix along the 1st dimension 
    


# In[8]:


print(tensor_hit_train.shape)
print(tensor_hit_test.shape)


# ### Load all the MISS-training trials in a given Session, one channel

# In[9]:


###################### 
# FIXED CHANNEL    ###
######################

######################### 
# TRAINING SET MISSES    ###
#########################

#>>>>>>>>>>>>>>>>>>
# Frequency 1 : (0,60) Hz
#>>>>>>>>>>>>>>>>>>

# name directory
pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum_fmin_{}'.format(Sess,Ch,fmin)
print(pathMiss)
# name file 1
fnMiss = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_{}_miss.txt".format(Ch,1)) # name file miss (first one)
print(fnMiss)
fnMissLabels = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_Misses_index.txt".format(Ch)) # name file with labels index

# load file 1 into miss-matrix and label vector
tensor_miss_train = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
tensor_miss_train = tensor_miss_train.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
labelsMiss = np.loadtxt(fnMissLabels,dtype='l',delimiter='\t') # load labels vector


#     #plt.imshow(np.flipud(tensor_miss.transpose()))
# load all other miss-files into a matrix, concatenate it
for indx in range(2,train_size+1):
    fnMiss = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_{}_miss.txt".format(Ch,indx)) # name file miss (following one)
    print(fnMiss)
    newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
    newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape 
    tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension 
    
    
#>>>>>>>>>>>>>>>>>>>>>>>>>>
# All other frequencies 
#>>>>>>>>>>>>>>>>>>>>>>>>>>    

for freq in range(fmin+1,fmax+1):
    
    pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum_fmin_{}'.format(Sess,Ch,freq) 
    
    print(pathMiss)
    for indx in range(1,train_size+1):
        fnMiss = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_{}_miss.txt".format(Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape 
        tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension  
        
        
        


# ### Load all the MISS-test trials in a given Session, one channel

# In[10]:


######################### 
# TEST SET MISSES    ###
#########################

#>>>>>>>>>>>>>>>>>>
# Frequency 1 : (0,60) Hz
#>>>>>>>>>>>>>>>>>>

# name directory
pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum_fmin_{}'.format(Sess,Ch,fmin)
print(pathMiss)
# name file 1
fnMiss = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_{}_miss.txt".format(Ch,train_size+1)) # name file miss (first one)
print(fnMiss)

# load file 1 into miss-matrix and label vector
tensor_miss_test = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
tensor_miss_test = tensor_miss_test.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches


#     #plt.imshow(np.flipud(tensor_miss.transpose()))
# load all other miss-files into a matrix, concatenate it
for indx in range(train_size+2,len(labelsMiss)+1):
    fnMiss = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_{}_miss.txt".format(Ch,indx)) # name file miss (following one)
    print(fnMiss)
    newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
    newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape 
    tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension 
    
    
#>>>>>>>>>>>>>>>>>>>>>>>>>>
# All other frequencies
#>>>>>>>>>>>>>>>>>>>>>>>>>>    

for freq in range(fmin+1,fmax+1):
    
    pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/homogeneous_sum_fmin_{}'.format(Sess,Ch,freq) 
    
    print(pathMiss)
    for indx in range(train_size+1,len(labelsHit)+1):
        fnMiss = os.path.join(pathMiss,"1_Subject_15_Sess_{}_Ch_{}_miss.txt".format(Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape 
        tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension


# In[4]:


# random inizialization 
# tensor_hit = torch.FloatTensor(858,100,61).uniform_(1,2)
# tensor_hit  = np.random.uniform(8,10,(858,100,61))
# tensor_miss  = np.random.uniform(2,2.001,(858,100,61))

# tensor_hit  = np.ones((858,100,61),dtype=float)
# tensor_miss  = np.ones((858,100,61),dtype=float)
# tensor_miss = 2*tensor_miss


# ### Balance the data set

# In[11]:


# tensor_hit = tensor_hit[0:858]

# balance the data set
# tensor_hit = tensor_hit[0:4344]
print(tensor_hit_train.shape)
print(tensor_hit_test.shape)

print(tensor_miss_train.shape)
print(tensor_miss_test.shape)


# In[12]:


plt.imshow(np.flipud(tensor_hit_train[100].transpose()))


# ### Generate the labels

# Generate the labels for hits and misses and stack them into a single array

# In[13]:


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

# In[179]:


# tensor_trial = tensor_hit
# labels_tot = labels_hit
# print(tensor_trial.shape)


# In[14]:


tensor_train = np.concatenate((tensor_hit_train,tensor_miss_train),axis=0)
print(tensor_train.shape)
tensor_test = np.concatenate((tensor_hit_test,tensor_miss_test),axis=0)
print(tensor_test.shape)


# ### Normalize inputs

# In[15]:


for indx in range(tensor_train.shape[0]):
    mean = np.mean(tensor_train[indx,:,:])
    std = np.std(tensor_train[indx,:,:,])
    tensor_train[indx,:,:,] = (tensor_train[indx,:,:,] - mean)/std
    
for indx in range(tensor_test.shape[0]):
    mean = np.mean(tensor_test[indx,:,:])
    std = np.std(tensor_test[indx,:,:,])
    tensor_test[indx,:,:,] = (tensor_test[indx,:,:,] - mean)/std


# In[16]:


plt.imshow(np.flipud(tensor_test[66,:,:].transpose()))


# In[17]:


tensor_train.shape


# ## Load data into a trainloader and testloader

# In[18]:


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# In[19]:


TOT_data_train = []
TOT_data_test = []

# concatenate data and labels
for i in range(len(labels_tot_train)):
    TOT_data_train.append([tensor_train[i,:,:], labels_tot_train[i]])
    
for i in range(len(labels_tot_test)):
    TOT_data_test.append([tensor_test[i,:,:], labels_tot_test[i]])

# shuffle the training data
TOT_data_train = [TOT_data_train[i] for i in range(tensor_train.shape[0])]
shuffle(TOT_data_train)

# shuffle the test data
TOT_data_test = [TOT_data_test[i] for i in range(tensor_test.shape[0])]
shuffle(TOT_data_test)

# Create datasets
train_dataset = CustomDataset(TOT_data_train)
test_dataset = CustomDataset(TOT_data_test)

# create train and test loaders
trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=20)
testloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=20)


# In[123]:


len(TOT_data_train)
len(TOT_data_test)
TOT_data_train


# ### Create iterable as a test 

# The batch_size decides how many images in the batch during the training or test.
# The numb of iteration in each train (test) loader are: tot length of train (test) data / batch_size. So if the tot number of data is 2000 and the batch size is 50, the iteration on the dataloader is done 40 times, each time 50 images are loaded 

# In[20]:


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


# In[21]:


print(images.shape)
labels


# In[22]:


images.view(images.shape[0],-1).shape
images.shape[1]


# # MODELS
# ### My multilayer perceptron (with dropout)

# In[196]:


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # linear layers
        self.fc1 = nn.Linear(6100,1500)
        self.fc2 = nn.Linear(1500,750)
        self.fc3 = nn.Linear(750,375)
        self.fc4 = nn.Linear(375,1)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.3)    
    
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
    
model = FFN()
model


# ### My CNN

# In[187]:


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
        self.dropout = nn.Dropout(0.3)
        
            
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
        x = self.pool(F.relu(self.conv2(x))) # Convolution 2: in short convolution + relu + pooling  
        
        # Convolution 3:
        x = self.pad_B(x) # padding bottom one step
#         print('x shape conv 3: ',x.shape)
        x = self.pool(F.relu(self.conv3(x))) # Convolution 3: in short convolution + relu + pooling 
        
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


# In[70]:


class small_CNN(nn.Module):
    def __init__(self):
        super(small_CNN,self).__init__()
         
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
        
#         # fully connected / Linear layers
#         self.fc1 = nn.Linear(64 * 13 * 8, 512)
#         self.fc2 = nn.Linear(512,256)            
#         self.fc3 = nn.Linear(256,1)
        
        # linear layers
        self.fc1 = nn.Linear(64 * 13 * 8,256)
        self.fc2 = nn.Linear(256,1)
#         self.fc3 = nn.Linear(750,375)
#         self.fc4 = nn.Linear(375,1)
        
        # asymmetric padding
        self.pad_R  = nn.ZeroPad2d((0, 1, 0, 0)) # padding right one step
        self.pad_B = nn.ZeroPad2d((0, 0, 1, 0)) # padding bottom one step 
        
        # dropout layer 
        self.dropout = nn.Dropout(0.3)
        
            
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
#         pdb.set_trace()        
        x = F.leaky_relu(x)     # activation function ReLU
#         print('x shape conv 1: ',x.shape)
#         pdb.set_trace()
        x = self.pool(x)  # max pooling
#         print('x shape conv 1: ',x.shape)
#         pdb.set_trace()
        
        # Convolution 2:
        x = self.pad_R(x) # padding right one step
#         print('x shape conv 2: ',x.shape)
        x = self.pool(F.leaky_relu(self.conv2(x))) # Convolution 2: in short convolution + relu + pooling  
        
        # Convolution 3:
        x = self.pad_B(x) # padding bottom one step
#         print('x shape conv 3: ',x.shape)
        x = self.pool(F.leaky_relu(self.conv3(x))) # Convolution 3: in short convolution + relu + pooling 
        
#         print('x shape before MLP: ',x.shape)

        ########################
        ## LINEAR LAYERS, MLP ##
        ########################
        
        x = x.view(x.shape[0],-1) # flatten image input
        x = self.dropout(x) # dropout
        
        x = F.leaky_relu(self.fc1(x)) # Fully connected 1: linear + relu
#         x = self.dropout(x) # dropout
        
#         x = F.leaky_relu(self.fc2(x)) # Fully connected 2: linear + relu
#         x = self.dropout(x) # dropout
        
#         x = F.leaky_relu(self.fc3(x)) # Fully connected 2: linear + relu
# #         x = self.dropout(x) # dropout
        
        x = self.fc2(x) # Fully connected 3: linear 
#         x = F.sigmoid(x)
        
        return x 

model = small_CNN()
print(model)


# ### Smaller multilayer perceptron 

# In[69]:


from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
#         self.fc4 = nn.Linear(64, 1) # careful because you have two output for a binary classification problem
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.25)  
    
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
#         x = self.dropout(F.leaky_relu(self.fc3(x)))
        
        x = self.fc3(x)

        return x
    
model = Classifier()
model


# ### Set loss function and optimizer

# In[71]:


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


# #### Move model on GPU if available 

# In[72]:


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

# In[73]:


def binary_acc(logits, labels):
    y_pred_tag = torch.round(torch.sigmoid(logits)) # make probability out of logits and then round to 0 or 1

    correct_results_sum = (y_pred_tag == labels).sum().float()
    acc = correct_results_sum/labels.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


# In[74]:


print('Train dataset length: ',len(train_dataset))
print('Test dataset length: ',len(test_dataset))
print('Trainloader length: ', len(trainloader))
print('Testloader length: ', len(testloader))


# ## Reset the weights to random numbers

# In[75]:


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

torch.manual_seed(42)
model.apply(weights_init)


# In[67]:


next(model.parameters()).is_cuda


# ## Train and validate the Network 

# Training and Testing is done on the fly:
# <br>
# For each epoch the model is trained (the loss is computed by using the training set) and it is tested (the loss and accuracy are computed by using the test set), in order to compare training loss with testing loss and decide when to stop the training and avoid overfitting. 

# In[76]:


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


# In[205]:


model


# ### Plot Train loss, Test loss, and Accuracy

# In[206]:


ep = np.arange(1,501)

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(13,4))
# ax = plt.subplot(111)
my_suptitle = fig.suptitle('MLP - homogeneous projections, fmin = [20-30] - batch size = 20', fontsize=18,y=1.05)

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
fig.savefig('Plots/MLP_homogeneous_f_20_30_bs_20.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle])
plt.close(fig)


# In[89]:


pwd


# In[ ]:




