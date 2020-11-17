#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import pdb


def load_data_homogeneous_complex(Sess,Ch,x_size,y_size):
        
    # name directory
    pathHit = 'Shaoyu_data/Data/Hits/1_Subject/{}_Sess/{}_Ch/complex/N_015_W_25_dn_018'.format(Sess,Ch)
    pathMiss = 'Shaoyu_data/Data/Misses/1_Subject/{}_Sess/{}_Ch/complex/N_015_W_25_dn_018'.format(Sess,Ch)
    # pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)
    # pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)
    print(pathHit)
    print(pathMiss)

    tensor_hit = np.empty((1,x_size,y_size,2), dtype='f')
    tensor_miss = np.empty((1,x_size,y_size,2), dtype='f')
    
# =============================================================================
# #### HITS #########
# =============================================================================
 
    # name files
    fnHitLabels = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_Hits_index.txt".format(Sess,Ch)) # name file with labels index
    labelsHit = np.loadtxt(fnHitLabels,dtype='l',delimiter='\t') # load labels vector
    # labelsHit = np.ones(43)
    
            # load all other hit-files into a matrix, concatenate it
    for indx in range(1,len(labelsHit)+1):
        ## load real part
        fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit_real.txt".format(Sess,Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit_Re = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit_Re = newmat_hit_Re.reshape(1,x_size,y_size,1) # reshape 
        
        ## load imaginary part 
        fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit_imaginary.txt".format(Sess,Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit_Im = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit_Im = newmat_hit_Im.reshape(1,x_size,y_size,1) # reshape 
        # merge real and imaginary
        newmat_hit = np.concatenate((newmat_hit_Re,newmat_hit_Im),axis=3)    
        # merge along the trial direction
        tensor_hit = np.concatenate((tensor_hit,newmat_hit),axis=0) # stack matrix along the 1st dimension 
            

# =============================================================================
# #### MISSES #######
# =============================================================================
    
    # name files
    fnMissLabels = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_Misses_index.txt".format(Sess,Ch)) # name file with labels index
    labelsMiss = np.loadtxt(fnMissLabels,dtype='l',delimiter='\t') # load labels vector
      

    # load all miss-files into a matrix, concatenate it
    for indx in range(1,len(labelsMiss)+1):
        ## load real part       
        fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss_real.txt".format(Sess,Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss_Re = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss_Re = newmat_miss_Re.reshape(1,x_size,y_size,1) # reshape the tensor leaving dim = 0 for batches
        
        ## load imaginary part       
        fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss_imaginary.txt".format(Sess,Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss_Im = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss_Im = newmat_miss_Im.reshape(1,x_size,y_size,1) # reshape the tensor leaving dim = 0 for batches
        # merge real and imaginary
        newmat_miss = np.concatenate((newmat_miss_Re,newmat_miss_Im),axis=3)    
        # merge along the trial direction
        tensor_miss = np.concatenate((tensor_miss,newmat_miss),axis=0) # stack matrix along the 1st dimension 
    

    return(tensor_hit, tensor_miss)



# =============================================================================
# ############################################################################
# =============================================================================




def load_data_random_complex(Sess,Ch,x_size,y_size,train_s,test_s,R):
   

    
    
    tensor_hit_train = np.empty((1,x_size,y_size,2), dtype='f')
    tensor_hit_test = np.empty((1,x_size,y_size,2), dtype='f')
    tensor_miss_train = np.empty((1,x_size,y_size,2), dtype='f')
    tensor_miss_test = np.empty((1,x_size,y_size,2), dtype='f')
   
 
    
    for R in range(0,R+1):
        
        
    
        print('Random: ',R) # print current channel
        
        # name directory
        # pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/random_different_weights/random_{}'.format(Sess,Ch,R)
        pathHit = 'Shaoyu_data/Data/Hits/1_Subject/{}_Sess/{}_Ch/complex/random_N_015_W_25_dn_018/random_{}'.format(Sess,Ch,R)
        # pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/random_NW/random_{}'.format(Sess,Ch,R)
        print(pathHit)
        # pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/random_different_weights/random_{}'.format(Sess,Ch,R)
        pathMiss = 'Shaoyu_data/Data/Misses/1_Subject/{}_Sess/{}_Ch/complex/random_N_015_W_25_dn_018/random_{}'.format(Sess,Ch,R)
        print(pathMiss)

    # =============================================================================
    # #### HITS #########
    # =============================================================================
        
        print('LOADING HIT TRIAL...')

        # name files
        fnHitLabels = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_Hits_index.txt".format(Sess,Ch)) # name file with labels index
        labelsHit = np.loadtxt(fnHitLabels,dtype='l',delimiter='\t') # load labels vector
        # labelsHit = np.ones(43)
        
            
        train_size = (np.floor(len(labelsHit)*train_s)).astype(int) # length trainig set
    
    ############# TRAIN ###############  
            
        for indx in range(1,train_size+1):
              ## load real part
                fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit_real.txt".format(Sess,Ch,indx)) # name file hit (following one)
                print(fnHit)
                newmat_hit_Re = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
                newmat_hit_Re = newmat_hit_Re.reshape(1,x_size,y_size,1) # reshape 
    
                ## load imaginary part 
                fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit_imaginary.txt".format(Sess,Ch,indx)) # name file hit (following one)
                print(fnHit)
                newmat_hit_Im = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
                newmat_hit_Im = newmat_hit_Im.reshape(1,x_size,y_size,1) # reshape 
    
                # merge real and imaginary
                newmat_hit = np.concatenate((newmat_hit_Re,newmat_hit_Im),axis=3)    
                # merge along the trial direction
                tensor_hit_train = np.concatenate((tensor_hit_train,newmat_hit),axis=0) # stack matrix along the 1st dimension   
                
            
    ############# TEST ######################        
    
        for indx in range(train_size+2,len(labelsHit)+1):
          ## load real part
                fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit_real.txt".format(Sess,Ch,indx)) # name file hit (following one)
                print(fnHit)
                newmat_hit_Re = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
                newmat_hit_Re = newmat_hit_Re.reshape(1,x_size,y_size,1) # reshape 
    
                ## load imaginary part 
                fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit_imaginary.txt".format(Sess,Ch,indx)) # name file hit (following one)
                print(fnHit)
                newmat_hit_Im = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
                newmat_hit_Im = newmat_hit_Im.reshape(1,x_size,y_size,1) # reshape 
    
                # merge real and imaginary
                newmat_hit = np.concatenate((newmat_hit_Re,newmat_hit_Im),axis=3)    
                # merge along the trial direction
                tensor_hit_test = np.concatenate((tensor_hit_test,newmat_hit),axis=0) # stack matrix along the 1st dimension  


    # =============================================================================
    # #### MISSES #######
    # =============================================================================

        print('LOADING MISS TRIAL...')

        # name files
        fnMissLabels = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_Misses_index.txt".format(Sess,Ch)) # name file with labels index
        labelsMiss = np.loadtxt(fnMissLabels,dtype='l',delimiter='\t') # load labels vector
               
        train_size = (np.floor(len(labelsMiss)*train_s)).astype(int) # length trainig set
        
        ############# TRAIN ###############
        
        print('Loading miss TRAIN...')
        # load all miss-files into a matrix, concatenate it
        for indx in range(1,train_size+1):
            
            ## load real part       
            fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss_real.txt".format(Sess,Ch,indx)) # name file miss (following one)
            print(fnMiss)
            newmat_miss_Re = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
            newmat_miss_Re = newmat_miss_Re.reshape(1,x_size,y_size,1) # reshape the tensor leaving dim = 0 for batches
            
            ## load imaginary part       
            fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss_imaginary.txt".format(Sess,Ch,indx)) # name file miss (following one)
            print(fnMiss)
            newmat_miss_Im = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
            newmat_miss_Im = newmat_miss_Im.reshape(1,x_size,y_size,1) # reshape the tensor leaving dim = 0 for batches
            
            # merge real and imaginary
            newmat_miss = np.concatenate((newmat_miss_Re,newmat_miss_Im),axis=3)    
            # merge along the trial direction
            tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension 
        
        
        ############# TEST ######################  
        
        print('Loading miss TEST...')
        # load all  miss-files into a matrix, concatenate it
        for indx in range(train_size+1,len(labelsMiss)+1):
            
        ## load real part       
            fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss_real.txt".format(Sess,Ch,indx)) # name file miss (following one)
            print(fnMiss)
            newmat_miss_Re = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
            newmat_miss_Re = newmat_miss_Re.reshape(1,x_size,y_size,1) # reshape the tensor leaving dim = 0 for batches
            
            ## load imaginary part       
            fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss_imaginary.txt".format(Sess,Ch,indx)) # name file miss (following one)
            print(fnMiss)
            newmat_miss_Im = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
            newmat_miss_Im = newmat_miss_Im.reshape(1,x_size,y_size,1) # reshape the tensor leaving dim = 0 for batches
            # merge real and imaginary
            newmat_miss = np.concatenate((newmat_miss_Re,newmat_miss_Im),axis=3)    
            # merge along the trial direction
            tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension 
        
        
    return(tensor_hit_train, tensor_hit_test, tensor_miss_train, tensor_miss_test)






def load_data_homogeneous(Sess,Ch,x_size,y_size,train_s,test_s):
        

    tensor_hit_train = np.empty((1,x_size,y_size), dtype='f')
    tensor_hit_test = np.empty((1,x_size,y_size), dtype='f')
    tensor_miss_train = np.empty((1,x_size,y_size), dtype='f')
    tensor_miss_test = np.empty((1,x_size,y_size), dtype='f')
    
   
###################
#### HITS #########
###################
 
    # pathHit = 'Ryan_data/Data/Hits/1_Subject/{}_Sess/{}_Ch/all_pulses_N_15_W_25_dn_018'.format(Sess,Ch)
    pathHit = 'Shaoyu_data/Data/Hits/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)
    print(pathHit)
    
    
    # name files
    fnHitLabels = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_Hits_index.txt".format(Sess,Ch)) # name file with labels index


    labelsHit = np.loadtxt(fnHitLabels,dtype='l',delimiter='\t') # load labels vector
    # labelsHit = np.ones(43)
    
    train_size = (np.floor(len(labelsHit)*train_s)).astype(int) # length trainig set
    test_size = len(labelsHit) - train_size  # length test set
    
    ############# TRAIN ###############  
        
    #     #plt.imshow(np.flipud(tensor_hit.transpose()))
    # load all other hit-files into a matrix, concatenate it
    for indx in range(1,train_size+1):
        fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit.txt".format(Sess,Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_train = np.concatenate((tensor_hit_train,newmat_hit),axis=0) # stack matrix along the 1st dimension 
            
    ############# TEST ######################        
    
    #     #plt.imshow(np.flipud(tensor_hit.transpose()))
    # load all other hit-files into a matrix, concatenate it
    for indx in range(train_size+2,len(labelsHit)+1):
        fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit.txt".format(Sess,Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_test = np.concatenate((tensor_hit_test,newmat_hit),axis=0) # stack matrix along the 1st dimension 
    
        
###################
#### MISSES #######
###################
    
        # name directory
    # pathMiss = 'Ryan_data/Data/Misses/1_Subject/{}_Sess/{}_Ch/all_pulses_N_15_W_25_dn_018'.format(Sess,Ch)
    pathMiss = 'Shaoyu_data/Data/Misses/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)

    # print(pathMiss)
    
    # name files
    fnMissLabels = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_Misses_index.txt".format(Sess,Ch)) # name file with labels index
    labelsMiss = np.loadtxt(fnMissLabels,dtype='l',delimiter='\t') # load labels vector
    
    train_size = (np.floor(len(labelsMiss)*train_s)).astype(int) # length trainig set
    test_size = len(labelsMiss) - train_size  # length test set
    
    ############# TRAIN ###############
    
    
    # load all miss-files into a matrix, concatenate it
    for indx in range(1,train_size+1):
        fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
        tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension 
    
    ############# TEST ######################     
     
    # load all  miss-files into a matrix, concatenate it
    for indx in range(train_size+2,len(labelsMiss)+1):
        fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
        tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension 


    return(tensor_hit_train, tensor_hit_test, tensor_miss_train, tensor_miss_test)





#########################################################
### IMPORT different NW, different image size        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#########################################################




def load_data_NW(Sess,Ch,x_size,y_size,train_s,test_s):
        
    
    tensor_hit_train = np.empty((1,x_size,y_size), dtype='f')
    tensor_hit_test = np.empty((1,x_size,y_size), dtype='f')
    tensor_miss_train = np.empty((1,x_size,y_size), dtype='f')
    tensor_miss_test = np.empty((1,x_size,y_size), dtype='f')
    
###################
#### HITS #########
###################
    
    ######################
    # 1st CHANNEL LOADED #
    ######################
    # name directory
    pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/N_025_W_6_k_2'.format(Sess,Ch)
    # pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)

    # print(pathHit)
    # name files
    fnHitLabels = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_Hits_index.txt".format(Sess,Ch)) # name file with labels index
    labelsHit = np.loadtxt(fnHitLabels,dtype='l',delimiter='\t') # load labels vector
    # labelsHit = np.ones(43)
    
    train_size = (np.floor(len(labelsHit)*train_s)).astype(int) # length trainig set
    test_size = len(labelsHit) - train_size  # length test set
    
    ############# TRAIN ###############  
        
    #     #plt.imshow(np.flipud(tensor_hit.transpose()))
    # load all other hit-files into a matrix, concatenate it
    for indx in range(1,train_size+1):
        
        pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/N_025_W_6_k_2'.format(Sess,Ch)
        fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_train = np.concatenate((tensor_hit_train,newmat_hit),axis=0) # stack matrix along the 1st dimension 
            
        pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/N_021_W_7_k_2_dn_002'.format(Sess,Ch)
        fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_train = np.concatenate((tensor_hit_train,newmat_hit),axis=0) # stack matrix along the 1st dimension 
            
        
    ############# TEST ######################        
    
    #     #plt.imshow(np.flipud(tensor_hit.transpose()))
    # load all other hit-files into a matrix, concatenate it
    for indx in range(train_size+2,len(labelsHit)+1):
        
        pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/N_025_W_6_k_2'.format(Sess,Ch)
        fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_test = np.concatenate((tensor_hit_test,newmat_hit),axis=0) # stack matrix along the 1st dimension 
        
        pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/homogeneous/N_021_W_7_k_2_dn_002'.format(Sess,Ch)
        fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
        print(fnHit)
        newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
        newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
        tensor_hit_test = np.concatenate((tensor_hit_test,newmat_hit),axis=0) # stack matrix along the 1st dimension 
    
    
        
###################
#### MISSES #######
###################
    
    ######################
    # 1st CHANNEL LOADED #
    ######################
        # name directory
    pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/N_025_W_6_k_2'.format(Sess,Ch)
    # pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)

    # print(pathMiss)
    
    # name files
    fnMissLabels = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_Misses_index.txt".format(Sess,Ch)) # name file with labels index
    labelsMiss = np.loadtxt(fnMissLabels,dtype='l',delimiter='\t') # load labels vector
    
    train_size = (np.floor(len(labelsMiss)*train_s)).astype(int) # length trainig set
    test_size = len(labelsMiss) - train_size  # length test set
    
    ############# TRAIN ###############
    
    
    # load all miss-files into a matrix, concatenate it
    for indx in range(1,train_size+1):
        
        pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/N_025_W_6_k_2'.format(Sess,Ch)
        fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
        tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension 
        
        pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/N_021_W_7_k_2_dn_002'.format(Sess,Ch)
        fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
        tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension 
    
    ############# TEST ######################     
     
    # load all  miss-files into a matrix, concatenate it
    for indx in range(train_size+2,len(labelsMiss)+1):
        
        pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/N_025_W_6_k_2'.format(Sess,Ch)
        fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
        tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension 

        pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/homogeneous/N_021_W_7_k_2_dn_002'.format(Sess,Ch)
        fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
        print(fnMiss)
        newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
        newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
        tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension 


    return(tensor_hit_train, tensor_hit_test, tensor_miss_train, tensor_miss_test)











#########################################################
### IMPORT AREA-BY-AREA                          +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#########################################################

def load_data_area(Sess,area,x_size,y_size,train_s,test_s):
    
    print('\n\n\n')
    path_areas = 'Data/Hits/1_Subject/{}_Sess/'.format(Sess)
    fn = os.path.join(path_areas,"{}_ch_index.txt".format(area)) # name file miss (first one)
    area_ch = np.loadtxt(fn,dtype='i',delimiter='\n') 
    print(area_ch)
    print('number of channel in {} = {}'.format(area,len(area_ch)))
    
    tensor_hit_train = np.empty((1,x_size,y_size), dtype='f')
    tensor_hit_test = np.empty((1,x_size,y_size), dtype='f')
    tensor_miss_train = np.empty((1,x_size,y_size), dtype='f')
    tensor_miss_test = np.empty((1,x_size,y_size), dtype='f')
    
    
###################   
#### HITS #########
###################   
    
    for Ch in area_ch:
    
        print('Channel: ',Ch) # print current channel
        
        print('LOADING HIT TRAIL...')
        
        # name directory
        pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)
        print(pathHit)
    
        # trial name file
        fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,1)) # name file hit (first one)
        fnHitLabels = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_Hits_index.txt".format(Ch)) # name file hit (first one)
        labelsHit = np.loadtxt(fnHitLabels,dtype='l',delimiter='\t') # load hit labels
        
        train_size = (np.floor(len(labelsHit)*train_s)).astype(int) # length trainig set
        test_size = len(labelsHit) - train_size  # length test set
        
        print('hits train and test and size: ',train_size,test_size)
        
        ############# TRAIN ###############
        
        print('loading hit training test...')
        # load all other hit-files into a matrix, concatenate it
        for indx in range(1,train_size+1):
            fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
            print(fnHit)
            newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
            newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
            tensor_hit_train = np.concatenate((tensor_hit_train,newmat_hit),axis=0) # stack matrix along the 1st dimension 
            
            
        ############# TEST ######################        
    
        print('loading hit testing test...')
        # load all other hit-files into a matrix, concatenate it
        for indx in range(train_size+2,len(labelsHit)+1):
            fnHit = os.path.join(pathHit,"1_Subject_15_Sess_{}_Ch_{}_hit.txt".format(Ch,indx)) # name file hit (following one)
            print(fnHit)
            newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
            newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
            tensor_hit_test = np.concatenate((tensor_hit_test,newmat_hit),axis=0) # stack matrix along the 1st dimension 


###################
#### MISSES #######
###################

        print('LOADING MISS TRAIL...')
        
        # name directory
        pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)
        print(pathMiss)
        
        # name trial files
        fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,1)) # name file miss (first one)
        fnMissLabels = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_Misses_index.txt".format(Sess,Ch)) # name file with labels index
        labelsMiss = np.loadtxt(fnMissLabels,dtype='l',delimiter='\t') # load labels vector
        
        train_size = (np.floor(len(labelsMiss)*train_s)).astype(int) # length trainig set
        test_size = len(labelsMiss) - train_size  # length test set
        
        print('misses train and test and size: ',train_size,test_size)
    
        ############# TRAIN ###############
        
        print('loading miss training test...')
     
        # load all other miss-files into a matrix, concatenate it
        for indx in range(1,train_size+1):
            fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
            print(fnMiss)
            newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
            newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
            tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension 
        ############# TEST ######################     
        
        print('loading miss testing test...')
        # load file 1 into miss-matrix and label vector
    
        # load all other miss-files into a matrix, concatenate it
        for indx in range(train_size+2,len(labelsMiss)+1):
            fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
            print(fnMiss)
            newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
            newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
            tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension 
    

    print(tensor_hit_train.shape)
    print(tensor_hit_test.shape)
    print(tensor_miss_train.shape)
    print(tensor_miss_test.shape)

    return(tensor_hit_train, tensor_hit_test, tensor_miss_train, tensor_miss_test)







##########################################
####   RANDOM PROJECTIONS                +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##########################################


def load_data_random_NW(Sess,Ch,x_size,y_size,train_s,test_s,rand_max):
   
    tensor_hit_train = np.empty((1,x_size,y_size), dtype='f')
    tensor_hit_test = np.empty((1,x_size,y_size), dtype='f')
    tensor_miss_train = np.empty((1,x_size,y_size), dtype='f')
    tensor_miss_test = np.empty((1,x_size,y_size), dtype='f')
   
    
###################   
#### HITS #########
###################   
    
    for R in range(0,rand_max+1):
    
        print('Random: ',R) # print current channel
        
        
###################
#### HITS #########
###################
        
        print('LOADING HIT TRIAL...')
        
        # name directory
        # pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/random_different_weights/random_{}'.format(Sess,Ch,R)
        pathHit = 'Ryan_data/Data/Hits/1_Subject/{}_Sess/{}_Ch/random_N_015_W_25_dn_018/random_{}'.format(Sess,Ch,R)
        # pathHit = 'Data/Hits/1_Subject/{}_Sess/{}_Ch/random_NW/random_{}'.format(Sess,Ch,R)
        print(pathHit)
    
        # name files labels
        fnHitLabels = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_Hits_index.txt".format(Sess,Ch)) # name file with labels index
        labelsHit = np.loadtxt(fnHitLabels,dtype='l',delimiter='\t') # load labels vector
        # labelsHit = np.ones(43)
        
        train_size = (np.floor(len(labelsHit)*train_s)).astype(int) # length trainig set
        test_size = len(labelsHit) - train_size  # length test set
        
        ############# TRAIN ###############  
            
        #     #plt.imshow(np.flipud(tensor_hit.transpose()))
        # load all other hit-files into a matrix, concatenate it
        print('Loading hit TRAIN...')
        for indx in range(1,train_size+1):
            
            fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit.txt".format(Sess,Ch,indx)) # name file hit (following one)
            print(fnHit)
            newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
            newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
            tensor_hit_train = np.concatenate((tensor_hit_train,newmat_hit),axis=0) # stack matrix along the 1st dimension 
                 
        
        ############# TEST ######################        
        print('Loading hit TEST...')
        #     #plt.imshow(np.flipud(tensor_hit.transpose()))
        # load all other hit-files into a matrix, concatenate it
        for indx in range(train_size+1,len(labelsHit)+1):
            
            fnHit = os.path.join(pathHit,"1_Subject_{}_Sess_{}_Ch_{}_hit.txt".format(Sess,Ch,indx)) # name file hit (following one)
            print(fnHit)
            newmat_hit = np.loadtxt(fnHit,dtype='f',delimiter='\t') # load hit matrix
            newmat_hit = newmat_hit.reshape(1,x_size,y_size) # reshape 
            tensor_hit_test = np.concatenate((tensor_hit_test,newmat_hit),axis=0) # stack matrix along the 1st dimension 
            
            
        
###################
#### MISSES #######
###################
    
        print('LOADING MISS TRIAL...')
    
    
        # name directory
        # pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/random_different_weights/random_{}'.format(Sess,Ch,R)
        pathMiss = 'Ryan_data/Data/Misses/1_Subject/{}_Sess/{}_Ch/random_N_015_W_25_dn_018/random_{}'.format(Sess,Ch,R)

        # pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/random_NW/random_{}'.format(Sess,Ch,R)

        # pathMiss = 'Data/Misses/1_Subject/{}_Sess/{}_Ch/1_Subject_{}_Sess_{}_Ch'.format(Sess,Ch,Sess,Ch)
        
        # name files labels
        fnMissLabels = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_Misses_index.txt".format(Sess,Ch)) # name file with labels index
        labelsMiss = np.loadtxt(fnMissLabels,dtype='l',delimiter='\t') # load labels vector
        
        train_size = (np.floor(len(labelsMiss)*train_s)).astype(int) # length trainig set
        test_size = len(labelsMiss) - train_size  # length test set
        
        ############# TRAIN ###############
        
        print('Loading miss TRAIN...')
        # load all miss-files into a matrix, concatenate it
        for indx in range(1,train_size+1):
            
            fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
            print(fnMiss)
            newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
            newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
            tensor_miss_train = np.concatenate((tensor_miss_train,newmat_miss),axis=0) # stack matrix along the 1st dimension 
            
    
        ############# TEST ######################     
        print('Loading miss TEST...')
        # load all  miss-files into a matrix, concatenate it
        for indx in range(train_size+1,len(labelsMiss)+1):
            
            fnMiss = os.path.join(pathMiss,"1_Subject_{}_Sess_{}_Ch_{}_miss.txt".format(Sess,Ch,indx)) # name file miss (following one)
            print(fnMiss)
            newmat_miss = np.loadtxt(fnMiss,dtype='f',delimiter='\t') # load miss matrix
            newmat_miss = newmat_miss.reshape(1,x_size,y_size) # reshape the tensor leaving dim = 0 for batches
            tensor_miss_test = np.concatenate((tensor_miss_test,newmat_miss),axis=0) # stack matrix along the 1st dimension 
    
            

   
    print(tensor_hit_train.shape)
    print(tensor_hit_test.shape)
    print(tensor_miss_train.shape)
    print(tensor_miss_test.shape)
    
    return(tensor_hit_train, tensor_hit_test, tensor_miss_train, tensor_miss_test)   


        
        

    