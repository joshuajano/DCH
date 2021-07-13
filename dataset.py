import cv2
import os
import numpy as np
# We refer to https://keras.io/utils/#sequence for Keras sequence
from keras.utils import Sequence
# We refer to https://github.com/aleju/imgaug for data augmentation
import imgaug as ia
from imgaug import augmenters as iaa

# This function is to prepare the path of the training data
def prepareData(input_folder, output_folder):
    train_images = []

    for filename in sorted(os.listdir(input_folder+'image_left/')):
        # filename is the filename of each data in the input_folder
        image = {}

        # With the assumption that the input and output have similar name:
        image['left'] = input_folder +'image_left/'+ filename
        image['right'] = input_folder +'image_right/'+ filename
        image['output'] = output_folder + filename[:-4] + '.txt'
        
        # Each image filename is added to train_images array
        train_images += [image]
                        
    return train_images



# This class is the dataset generator for fit_generator function in keras
class DataLoader():
    def __init__(self, images, 
                       config,
                       shuffle  =   True,
                       train    =   True,
                       norm     =   None):

        self.images         = images    # This is the array of training images
        self.shuffle        = shuffle   # It shuffles the train images if True
        self.train          = train     # It augment the train images if true
        self.norm           = norm      # It performs the normalization for each train image
        self.batch_size     = config['batch_size']    
        self.image_w        = config['image_w']
        self.image_h        = config['image_h']
        self.channel        = config['channel']
        self.network_name   = config['network']
        
        self.seq            = iaa.Sequential([iaa.GaussianBlur((0, 3.0))])

        # Perform the shuffling for the train images
        #if self.shuffle:
        #    np.random.shuffle(self.images)

    def normDeepHomo(self, homography):
        homography = homography / 24
        return homography

    def normGeometric(self, homography):
        homography[2] = (homography[2])
        homography[5] = (homography[5])
        return homography

    def totalIteration(self):
        return int(np.ceil(float(len(self.images))/self.batch_size))

    def load(self, idx):
        
        # The left and right boundary index in the array for each batch_size
        l_idx = idx     * self.batch_size
        r_idx = (idx+1) * self.batch_size

        
        # If the right boundary index is larger than the total array, then it shift the array a little bit
        if r_idx > len(self.images):
            r_idx = len(self.images)
            l_idx = r_idx - self.batch_size
            
        # The array for each batch_size
        left_batch     = np.zeros((self.batch_size, self.image_h, self.image_w, self.channel))    
        right_batch     = np.zeros((self.batch_size, self.image_h, self.image_w, self.channel))
        if self.network_name == 'JanoNet' or self.network_name == 'PaoNet':
            output_batch    = np.zeros((self.batch_size, 9))  
        else:
            output_batch    = np.zeros((self.batch_size, 8))  
        
        index = 0
        for image in self.images[l_idx:r_idx]:            
            left, right, output = self.loadData(image, train=self.train)
            # Assign input and output to batch array and perform normalization if needed
            if self.norm != None: 
                left_batch[index] = self.norm(left)
                right_batch[index] = self.norm(right)
                if self.network_name == 'DeepNet' or self.network_name == 'NewNet':
                    output_batch[index] = self.normDeepHomo(output)
                else:
                    output_batch[index] = self.normGeometric(output)

            else:
                left_batch[index] = left
                right_batch[index] = right
                output_batch[index] = output
            # Increase the index in batch array
            index += 1

        return left_batch, right_batch, output_batch

    # Every sequence might have on_epoch_end function to perform any operation after each epoch ends
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    # Function loadData performs loading operation for each input and output image
    def loadData(self, image, train):

        # Load the image using OpenCV function
        left_name = image['left']
        left = cv2.imread(left_name)
        right_name = image['right']
        right = cv2.imread(right_name)
        output_name = image['output']
        with open(output_name,"r") as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        output = np.array(content[0].split()).astype(np.float)
        
        if self.network_name == 'JanoNet' or self.network_name == 'PaoNet':
            output = np.append(output,1)
        

        # Perform data augmentation for training process
        #if train:
        #    seq_det = self.seq.to_deterministic()
        #    left = seq_det.augment_images(left)
        #    right = seq_det.augment_images(right)
            
        # Resize the image input desired image size for training and validation
        left = cv2.resize(left, (self.image_w, self.image_h))
        right = cv2.resize(right, (self.image_w, self.image_h))
        
        return left, right, output
