#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:51:31 2020

@author: bijanadmin
"""

from torch import nn
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt



#%%
################################################
# ### Multilayer perceptron (with dropout)


class MLP(nn.Module):
    def __init__(self,x,y,prob):
        super().__init__()
        
        # linear layers
        # self.fc1 = nn.Linear(6100,1500)
        # self.fc2 = nn.Linear(1500,750)
        # self.fc3 = nn.Linear(750,375)
        # self.fc4 = nn.Linear(375,1)
        
        self.fc1 = nn.Linear(x*y,200)
        self.fc2 = nn.Linear(200,1)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=prob)    
    
    def forward(self,x):
        
        # flatten the input tensor 
        x = x.view(x.shape[0],-1)
        
        # activation functions
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        # x = self.dropout(F.leaky_relu(self.fc2(x)))
        # x = self.dropout(F.leaky_relu(self.fc3(x)))
        x = self.fc2(x)
        
#         x = F.relu(self.fc4(x))
#         x = F.sigmoid(self.fc5(x))
        
        return x




#%%
################################################
# ### Multilayer perceptron (with dropout)


class MLP_deep(nn.Module):
    def __init__(self,x,y,prob):
        super().__init__()
        
        # linear layers
        # self.fc1 = nn.Linear(6100,1500)
        # self.fc2 = nn.Linear(1500,750)
        # self.fc3 = nn.Linear(750,375)
        # self.fc4 = nn.Linear(375,1)
        
        self.fc1 = nn.Linear(x*y,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,25)
        self.fc4 = nn.Linear(25,1)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=prob)    
    
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




#%%
################################################
# ### Linear regression


class Linear_Regression(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        
        # linear layers
        self.fc1 = nn.Linear(x*y,1)
        
        # Dropout module with 0.2 drop probability
#         self.dropout = nn.Dropout(p=0.2)    
    
    def forward(self,x):
        
        # flatten the input tensor 
        x = x.view(x.shape[0],-1)
        
        # activation functions
        x = self.fc1(x)
        
        return x
    


#%%
################################################
# ### Udacity multilayer perceptron 

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
    


#%%
################################################
# ## CONVOLUTIONAL NETWORKS

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
        # sees a layer 13*8 which is from pooling 26*16
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
        
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2]) # images is expanded to host the channel (1 in this case, 3 when RGB) dimension for the CNN

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





#%%
################################################
# ## CONVOLUTIONAL NETWORKS SMALL


    
class CNN_small(nn.Module):
    def __init__(self):
        super(CNN_small,self).__init__()
         
        #######################    
        # convolutional layers
        # ^^^^^^^^^^^^^^^^^^^^
    
        # convolutional layer (sees 62x30x1 image tensor)
        self.conv1 = nn.Conv2d(1,4,3,padding=1)
        # convolutional layer (sees 32x16x4 tensor)
        self.conv2 = nn.Conv2d(4,8,3,padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2,2)
        
         # batch normalization 
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        
#         # fully connected / Linear layers
#         self.fc1 = nn.Linear(64 * 13 * 8, 512)
#         self.fc2 = nn.Linear(512,256)            
#         self.fc3 = nn.Linear(256,1)
        
        # linear layers
        # sees a layer 12x8x8
        self.fc1 = nn.Linear(16*8*8,1)
        # self.fc2 = nn.Linear(384,1)
        
        # asymmetric padding
        self.pad_L  = nn.ZeroPad2d((1, 0, 0, 0)) # padding right one step
        self.pad_R  = nn.ZeroPad2d((0, 1, 0, 0)) # padding right one step
        self.pad_T = nn.ZeroPad2d((0, 0, 1, 0)) # padding bottom one step 
        self.pad_B = nn.ZeroPad2d((0, 0, 0, 1)) # padding bottom one step 
        
        # dropout layer 
        self.dropout = nn.Dropout(0.2)
        
            
    def forward(self,x):
        
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2]) # images is expanded to host the channel (1 in this case, 3 when RGB) dimension for the CNN

        #########################
        # CONVOLUTIONAL LAYERS ##
        #########################
        
        # Convolution 1:
        # print('x shape before conv 1: ',x.shape)
        # pdb.set_trace()
        x = self.pad_B(x) # padding right one step  -- comment out for 46x20 images
        # x = x [:,:,1:,:] 
        
        # print('x shape conv 1,, after padding: ',x.shape)
        # pdb.set_trace()
        # plt.imshow(x[1,0,:,:].detach().cpu().numpy().transpose(),origin='lower')
        # plt.show()

        conv_x1 = self.conv1(x) # convolutional layer
        # print('x shape after conv 1: ',x.shape)
        # pdb.set_trace() 
        # batch normalization 
        x = self.bn1(conv_x1) 
        x = F.relu(x)     # activation function ReLU
        # print('x shape after bn relu 1: ',x.shape)
        # pdb.set_trace()
        x = self.pool(x)  # max pooling
        # x = x [:,:,1:,1:] 
        # print('x shape after pool 1: ',x.shape)
        # pdb.set_trace()
        
        # Convolution 2:
        x = self.pad_R(x) # padding right one step  -- comment out for 46x20 images
        x = self.pad_B(x) # padding bottom one step
        
        # plt.imshow(x[1,0,:,:].detach().cpu().numpy().transpose(),origin='lower')
        # plt.show()

        # print('x shape before conv 2: ',x.shape)
        # pdb.set_trace()
        conv_x2 = self.conv2(x)
        x = self.pool(F.relu(self.bn2(conv_x2))) # Convolution 2: in short convolution + relu + pooling  
        
        
        # print('x shape before MLP: ',x.shape)
        # pdb.set_trace()
        ########################
        ## LINEAR LAYERS, MLP ##
        ########################
        
        x = x.view(x.shape[0],-1) # flatten image input
        # print('x shape in MLP: ',x.shape)
        # pdb.set_trace() 
        x = self.dropout(x) # dropout
        
        # print('x shape before fc1: ',x.shape)
        # pdb.set_trace() 
        # x = F.leaky_relu(self.fc1(x)) # Fully connected 1: linear + relu
        # x = self.dropout(x) # dropout
        
        x = self.fc1(x) # Fully connected 3: linear 
#         x = F.sigmoid(x)
        
        # print('x shape before fc2: ',x.shape)
        # pdb.set_trace() 

        return x, conv_x1, conv_x2 

# model = CNN()
model = CNN_small()
print(model)



#%%
################################################
# ## CONVOLUTIONAL NETWORKS DEEP


    
class CNN_deep(nn.Module):
    def __init__(self):
        super(CNN_deep,self).__init__()
         
        #######################    
        # convolutional layers
        # ^^^^^^^^^^^^^^^^^^^^
    
        # convolutional layer (sees 46x30x1 image tensor)
        self.conv1 = nn.Conv2d(1,2,3,padding=1)
        # convolutional layer (sees 24x16x1 image tensor)
        self.conv2 = nn.Conv2d(2,4,3,padding=1)
        # convolutional layer (sees 12x8x1 image tensor)
        self.conv3 = nn.Conv2d(4,8,3,padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2,2)
        
         # batch normalization 
        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(8)
        
#         # fully connected / Linear layers
#         self.fc1 = nn.Linear(64 * 13 * 8, 512)
#         self.fc2 = nn.Linear(512,256)            
#         self.fc3 = nn.Linear(256,1)
        
        # linear layers
        # sees a layer 6x4x8
        self.fc1 = nn.Linear(6*4*8,1)
        # self.fc2 = nn.Linear(384,1)
        
        # asymmetric padding
        self.pad_L  = nn.ZeroPad2d((1, 0, 0, 0)) # padding right one step
        self.pad_R  = nn.ZeroPad2d((0, 1, 0, 0)) # padding right one step
        self.pad_T = nn.ZeroPad2d((0, 0, 1, 0)) # padding bottom one step 
        self.pad_B = nn.ZeroPad2d((0, 0, 0, 1)) # padding bottom one step 
        
        # dropout layer 
        self.dropout = nn.Dropout(0.2)
        
            
    def forward(self,x):
        
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2]) # images is expanded to host the channel (1 in this case, 3 when RGB) dimension for the CNN

        #########################
        # CONVOLUTIONAL LAYERS ##
        #########################
        
        # Convolution 1:
        # print('x shape before conv 1: ',x.shape)
        # pdb.set_trace()
        # x = self.pad_B(x) # padding right one step  -- comment out for 46x20 images
        
        # print('x shape conv 1,, after padding: ',x.shape)
        # pdb.set_trace()
        # plt.imshow(x[1,0,:,:].detach().cpu().numpy().transpose(),origin='lower')
        # plt.show()

        conv_x1 = self.conv1(x) # convolutional layer
        # print('x shape after conv 1: ',x.shape)
        # pdb.set_trace() 
        # batch normalization 
        x = self.bn1(conv_x1) 
        x = F.relu(x)     # activation function ReLU
        # print('x shape after bn relu 1: ',x.shape)
        # pdb.set_trace()
        x = self.pool(x)  # max pooling
        # x = x [:,:,1:,1:] 
        # print('x shape after pool 1: ',x.shape)
        # pdb.set_trace()
        
        # Convolution 2:
        x = self.pad_R(x) # padding right one step  -- comment out for 46x20 images
        x = self.pad_B(x) # padding bottom one step
        
        # plt.imshow(x[1,0,:,:].detach().cpu().numpy().transpose(),origin='lower')
        # plt.show()

        # print('x shape before conv 2: ',x.shape)
        # pdb.set_trace()
        conv_x2 = self.conv2(x)
        x = self.pool(F.relu(self.bn2(conv_x2))) # Convolution 2: in short convolution + relu + pooling  
        
        # print('x shape before conv 2: ',x.shape)
        # pdb.set_trace()
        
        conv_x3 = self.conv3(x)
        x = self.pool(F.relu(self.bn3(conv_x3))) # Convolution 2: in short convolution + relu + pooling  
        
        # print('x shape before MLP: ',x.shape)
        # pdb.set_trace()
        ########################
        ## LINEAR LAYERS, MLP ##
        ########################
        
        x = x.view(x.shape[0],-1) # flatten image input
        # print('x shape in MLP: ',x.shape)
        # pdb.set_trace() 
        x = self.dropout(x) # dropout
        
        # print('x shape before fc1: ',x.shape)
        # pdb.set_trace() 
        # x = F.leaky_relu(self.fc1(x)) # Fully connected 1: linear + relu
        # x = self.dropout(x) # dropout
        
        x = self.fc1(x) # Fully connected 3: linear 
#         x = F.sigmoid(x)
        
        # print('x shape before fc2: ',x.shape)
        # pdb.set_trace() 

        return x, conv_x1, conv_x2, conv_x3

# model = CNN()
model = CNN_deep()
print(model)

