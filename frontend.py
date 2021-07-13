import cv2
import os
import numpy as np
import tensorflow as tf
import pickle
import datetime

from network import JoshuaNetwork, Deep_homography, PaoNetwork, HomographyWarp, AffineWarp, NewNetwork
from dataset import DataLoader
from tqdm import trange

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.models import Model
import keras.backend as K
from sklearn.metrics import mean_absolute_error
from skimage.measure import compare_psnr as psnr



class HomoNet():
    def __init__(self, network_name, image_w, image_h):

        self.network_name = network_name
        self.image_w = image_w
        self.image_h = image_h
        self.channel = 3
        
        if network_name == 'JanoNet':
            self.network = JoshuaNetwork(self.image_w,self.image_h,self.channel)
            self.losses = self.pointAffineHomo_loss
            self.network.model.summary()
        elif network_name == 'DeepNet':
            self.network = Deep_homography(self.image_w, self.image_h, 6)
            self.losses = self.euclidean_distance_loss
            self.network.model.summary()
        elif network_name == 'PaoNet':
            self.network = PaoNetwork(self.image_w, self.image_h,self.channel)
            self.losses = self.pointHomo_loss
            self.network.model.summary()
        elif network_name == 'NewNet':
            self.network = NewNetwork(self.image_w, self.image_h, 6)
            self.losses = self.euclidean_distance_loss
            self.network.model.summary()
        elif network_name == 'PaoNet_l2':
            self.network = PaoNetwork(self.image_w, self.image_h,self.channel)
            self.losses = self.euclidean_distance_loss
            self.network.model.summary()
        elif network_name == 'DLT':
            self.network = None
        elif network_name == 'RANSAC':
            self.network = None
        else:
            raise Exception('Please implement the ' + network_name +' class first!')
        
        

    # This function is for loading the pretrained weights
    def load_weights(self, weight_path):
        self.network.model.load_weights(weight_path)  
                
    def save_weights(self, weight_path):
        self.network.model.save_weights(weight_path) 
        
    def pointAffineHomo_loss(self, y_true, y_pred):

        f_pred =  AffineWarp((10, 10))(y_pred)
        f_true =  HomographyWarp((10, 10))(y_true)

        return K.sqrt(K.sum(K.square(f_pred - f_true)))
    
    def pointHomo_loss(self, y_true, y_pred):

        f_pred =  HomographyWarp((10, 10))(y_pred)
        f_true =  HomographyWarp((10, 10))(y_true)

        return K.sqrt(K.sum(K.square(f_pred - f_true)))

    def euclidean_distance_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    # This function works for performing training using the given training data
    def train(self, train_images,
                    valid_images,
                    batch_size,
                    lr_rate,
                    weight_path,
                    epochs,
                    print_frequency = 1,
                    weight_frequency = 1):

        self.batch_size = batch_size
        
        # This config variable consists of several parameters for Keras sequence
        generator_config = {
            'image_h'         : self.image_h, 
            'image_w'         : self.image_w,
            'channel'         : self.channel,
            'batch_size'      : self.batch_size,
            'network'         : self.network_name
        }    

        # This is the dataset generator for training images
        train_loader = DataLoader(train_images, 
                                        generator_config, 
                                        norm=self.network.normImage,
                                        train=True)
        # This is the dataset generator for validation images
        valid_loader = DataLoader(valid_images, 
                                        generator_config, 
                                        norm=self.network.normImage,
                                        train=False)   
        
        # This is the optimizer. You can choose Adam, SGD, etc. We refer to https://keras.io/optimizers/
        optimizer = Adam(lr=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-06)
             
        # We compile the network model, using the MSE loss. We refer to https://keras.io/losses/
        self.network.model.compile(loss= self.losses, optimizer=optimizer)
        
        start_epoch = datetime.datetime.now()
        for epoch in range(epochs):

            length = train_loader.totalIteration()
            training_losses = []

            with trange(length) as t:
                for idx in t:
                    left, right, output = train_loader.load(idx)
                    network_loss = self.network.model.train_on_batch([left,right],output)    
                    # Save losses
                    training_losses.append(network_loss)
                    t.set_postfix(loss=network_loss)

            train_loader.on_epoch_end()

            length = valid_loader.totalIteration()
            validation_losses = []

            with trange(length) as t:
                for idx in t:
                    left, right, output = valid_loader.load(idx)
                    network_loss = self.network.model.train_on_batch([left,right],output)   
                    # Save losses
                    validation_losses.append(network_loss)
                    t.set_postfix(loss=network_loss)

            valid_loader.on_epoch_end()

            # Plot the progress
            if epoch % print_frequency == 0:
                print("Epoch {}/{} | Time: {}s\n>> Training loss: {}\n>> Validation loss: {}\n".format(
                    epoch+1, epochs,
                    (datetime.datetime.now() - start_epoch).seconds,
                    sum(training_losses) / float(len(training_losses)),
                    sum(validation_losses) / float(len(validation_losses))
                ))

            # Check if we should save the network weights
            if weight_frequency and epoch % weight_frequency == 0:
                # Save the network weights
                self.save_weights(weight_path)
                
    def evaluatePositionError(self, test_folder):
        
        image_full_w = self.image_w + int(self.image_w / 2)
        image_full_h = self.image_h
        gt_folder = test_folder + '/gt_point/'
                
        path = os.listdir(gt_folder)
        error = []
        for filename in path: 
            with open(test_folder+'/' + self.network_name + '/'+filename,"r") as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            H = np.array(content[0].split()).astype(np.float)
            H = np.reshape(H,(3,3))
            H = np.linalg.inv(H)
                        
            top_left            =   (160, 0, 1)
            bottom_left         =   (160, 240, 1)
            bottom_right        =   (480, 240, 1)
            top_right           =   (480, 0, 1)
            initial_point       =   [top_left, bottom_left, bottom_right, top_right]
            outer_point = np.matmul(H,np.transpose(initial_point))
                        
            outer_point[0] = outer_point[0] / outer_point[2]
            outer_point[1] = outer_point[1] / outer_point[2]
            point_diff = np.zeros((8,1))
            point_diff[0] = outer_point[0][0] - 0
            point_diff[1] = outer_point[1][0] - 0
            point_diff[2] = outer_point[0][1] - 0
            point_diff[3] = outer_point[1][1] - 240
            point_diff[4] = outer_point[0][2] - 320
            point_diff[5] = outer_point[1][2] - 240
            point_diff[6] = outer_point[0][3] - 320
            point_diff[7] = outer_point[1][3] - 0
            
            with open(gt_folder + '/'+filename,"r") as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            gt_point = np.array(content[0].split()).astype(np.float)

            mae = mean_absolute_error(point_diff,gt_point)
            error.append(mae)
        
        print("The position MAE of " + self.network_name + " is")
        print(np.mean(error))
            
    def evaluatePixelError(self, test_folder):
        
        image_full_w = self.image_w + int(self.image_w / 2)
        image_full_h = self.image_h
        gt_folder = test_folder + '/image_resize/'
                
        path = os.listdir(gt_folder)
        error = []
        for filename in path: 
            with open(test_folder+'/' + self.network_name + '/'+filename[:-4]+'.txt',"r") as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            H = np.array(content[0].split()).astype(np.float)
            H = np.reshape(H,(3,3))

            left = cv2.imread(test_folder+'/image_left/'+filename[:-4]+'.png')
            right = cv2.imread(test_folder+'/image_right/'+filename[:-4]+'.png')
            warped_image = cv2.warpPerspective(right.copy(), H, (image_full_w,image_full_h))
            warped_image[0:self.image_h,0:self.image_w,:] = left
            cv2.imwrite(test_folder+'/' + self.network_name + '/'+filename[:-4]+'.png',warped_image)
            
            
            with open(test_folder+'/gt/'+filename[:-4]+'.txt',"r") as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            H_GT = np.array(content[0].split()).astype(np.float)
            H_GT = np.append(H_GT,[1])
            H_GT = np.reshape(H_GT,(3,3))
            
            left_mask = np.ones((self.image_h,self.image_w))
            right_mask = np.ones((self.image_h,self.image_w))
            warped_mask = np.zeros((image_full_h,image_full_w))
            warped_mask = cv2.warpPerspective(right_mask.copy(), H_GT, (image_full_w,image_full_h))
            warped_mask[0:self.image_h,0:self.image_w] = left_mask
            cv2.imwrite(test_folder+'/' + self.network_name + '/'+filename[:-4]+'_mask.png',warped_mask * 255)

            gt_image = cv2.imread(gt_folder + '/'+filename)
            gt_image[warped_mask == 0] = [0,0,0]
            mse = psnr(gt_image,warped_image)

            error.append(mse)
            
        print("The pixel PSNR of " + self.network_name + " is")
        print(np.mean(error))

            
    def predict(self, test_folder):
        if os.path.isdir(test_folder+'/' + self.network_name) == False:
            os.mkdir(test_folder+'/' + self.network_name)

        image_full_w = self.image_w + int(self.image_w / 2)
        image_full_h = self.image_h

        count = 0
        path = os.listdir(test_folder+'/image_left/')
        for filename in path:            
            left = []
            left.append(cv2.imread(test_folder+'/image_left/'+filename).astype(np.float))
            left = np.array(left) / 127.5 - 1      
            right = []
            right.append(cv2.imread(test_folder+'/image_right/'+filename).astype(np.float))
            right = np.array(right) / 127.5 - 1

            if self.network_name == 'DeepNet' or self.network_name == 'NewNet':
            
                homography          =   np.zeros((1, 9))  
                homography[0,0:8]   =   self.network.model.predict([left,right])
                homography[0,0:8]   =   homography[0,0:8]*24
                
                top_left            =   (0 + homography[0, 0], 0+ homography[0, 1])
                bottom_left         =   (0 + homography[0, 2], 240 + homography[0, 3])
                bottom_right        =   (320 + homography[0, 4], 240 + homography[0, 5])
                top_right           =   (320 + homography[0, 6], 0 + homography[0, 7])
                initial_point       =   [top_left, bottom_left, bottom_right, top_right]

                top_left_pert       =   (160, 0)
                bottom_left_pert    =   (160, 240)
                bottom_right_pert   =   (480, 240)
                top_right_pert      =   (480, 0)
                pertubated_point    =   [top_left_pert, bottom_left_pert, bottom_right_pert, top_right_pert]
                
                H = cv2.getPerspectiveTransform(np.float32(initial_point), np.float32(pertubated_point))
            elif self.network_name == 'JanoNet':
                H    = np.zeros((1, 9)) 
                H[0,0:6] = self.network.model.predict([left,right]) 
                H[0,8] = 1
                H = np.reshape(H,(3,3))
            elif self.network_name == 'PaoNet':
                H    = np.zeros((1, 9)) 
                H = self.network.model.predict([left,right]) 
                H = np.reshape(H,(3,3))
            elif self.network_name == 'DLT':
                left_image  =   cv2.imread(test_folder + '/image_left/' + filename)
                right_image =   cv2.imread(test_folder + '/image_right/' + filename)
            
                orb = cv2.ORB_create()
                kp1 = orb.detect(left_image,None)
                kp2 = orb.detect(right_image, None)

                kp1, des1 = orb.compute(left_image, kp1)
                kp2, des2 = orb.compute(right_image, kp2)
            
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
                H = None
                if des1 is not None and des2 is not None:
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key = lambda x:x.distance)

                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                    H, mask = cv2.findHomography(dst_pts, src_pts, method = 0)
                if H is None:                            
                    H    = np.zeros((9, 1)) 
                    H[0] = 1
                    H[2] = 160
                    H[4] = 1
                    H[8] = 1
            elif self.network_name == 'RANSAC':
                left_image  =   cv2.imread(test_folder + '/image_left/' + filename)
                right_image =   cv2.imread(test_folder + '/image_right/' + filename)
            
                orb = cv2.ORB_create()
                kp1 = orb.detect(left_image,None)
                kp2 = orb.detect(right_image, None)

                kp1, des1 = orb.compute(left_image, kp1)
                kp2, des2 = orb.compute(right_image, kp2)
            
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
                H = None
                if des1 is not None and des2 is not None:
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key = lambda x:x.distance)

                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
                if H is None:               
                    H    = np.zeros((9, 1)) 
                    H[0] = 1
                    H[2] = 160
                    H[4] = 1
                    H[8] = 1


            H = np.reshape(H,(9,1))
            with open(test_folder+'/' + self.network_name + '/'+ filename[:-4] + '.txt', 'w') as f:
                for item in H:
                    f.write("%s " % float(item))            
            count+=1
            print(count)
