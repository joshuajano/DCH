import cv2
import os
import numpy as np
from dataset import prepareData
from frontend import HomoNet
from glob import glob
import random

def runTrain(network_name):    
    # Training parameter setup
    epochs              = 50
    batch_size          = 16
    lr_rate             = 1e-3
    image_w             = 320
    image_h             = 240
    weight_path         = network_name + '_model.h5'

    # Training data folder
    
    input_folder        = '../train/'
    if network_name == 'JanoNet' or network_name == 'PaoNet' or network_name == 'PaoNet_l2':
        output_folder       = '../train/gt/'
    else:
        output_folder       = '../train/gt_point/'
    

    train_images = prepareData(input_folder,output_folder)
    np.random.shuffle(train_images)

    # We split the training data into 80% train data and 20% validation data
    split = int(0.8*len(train_images))
    valid_images = train_images[split:]
    train_images = train_images[:split]

    # We call the network class
    object = HomoNet(network_name        = network_name,
                            image_h          = image_h,
                            image_w          = image_w)

    # If it exists, we call the pretrained weights
    if(os.path.exists(weight_path)):
        object.load_weights(weight_path)

    # We call train function to train
    object.train(train_images   = train_images,
                 valid_images   = valid_images,
                 batch_size     = batch_size,
                 lr_rate        = lr_rate,
                 weight_path    = weight_path,
                 epochs         = epochs)

def runPredict(network_name):    
    # Training parameter setup
    image_w             = 320
    image_h             = 240
    weight_path         = network_name + '_model.h5'

    # Training data folder
    input_folder        = '../test'

    # We call the network class
    object = HomoNet(network_name        = network_name,
                            image_h          = image_h,
                            image_w          = image_w)

    # If it exists, we call the pretrained weights
    if(os.path.exists(weight_path)):
        object.load_weights(weight_path)
    
    
    # We call train function to train
    object.predict(input_folder)


def runEvaluation(network_name):    
    # Training parameter setup
    image_w             = 320
    image_h             = 240
    weight_path         = network_name + '_model.h5'

    # Training data folder
    input_folder        = '../test/'

    # We call the network class
    object = HomoNet(network_name        = network_name,
                            image_h          = image_h,
                            image_w          = image_w)

    # If it exists, we call the pretrained weights
    if(os.path.exists(weight_path)):
        object.load_weights(weight_path)

    # We call train function to train
    object.evaluatePositionError(input_folder)
    object.evaluatePixelError(input_folder)


def generateDataset(path):
    
    count = 0
    for img in glob(path):
        rho = 24
        image_w = 320
        image_h = 240

        image_full_w = image_w + int(image_w / 2)
        image_full_h = image_h

        image = cv2.imread(img)
        image = cv2.resize(image,(image_full_w+rho*2,image_full_h+rho*2))

        start_x = rho;
        start_y = rho;
                
        image_full = image[start_y:image_full_h+rho*2-start_y,start_x:image_full_w+rho*2-start_x].copy()
        cv2.imwrite('train/image_resize/'+str(count) +'.png',image_full)

        # four points for cropping left image
        top_left        = (start_x,start_y)
        bottom_left     = (start_x,start_y + image_h)
        bottom_right    = (start_x + image_w,start_y + image_h)
        top_right       = (start_x + image_w,start_y)

        four_left_points = [top_left,bottom_left,bottom_right,top_right]
    
        image_left = image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]].copy()
        cv2.imwrite('train/image_left/'+str(count) +'.png',image_left)

        # four points for initial right image
        top_left_r1        = (start_x + 160,start_y)
        bottom_left_r1     = (start_x + 160,start_y + image_h)
        bottom_right_r1    = (start_x + 160 + image_w,start_y + image_h)
        top_right_r1       = (start_x + 160 + image_w,start_y)
    
        four_r1_points = [top_left_r1,bottom_left_r1,bottom_right_r1,top_right_r1]
                
        rand_rho_tl = (random.randint(-rho,rho),random.randint(-rho,rho))
        rand_rho_bl = (random.randint(-rho,rho),random.randint(-rho,rho))
        rand_rho_br = (random.randint(-rho,rho),random.randint(-rho,rho))
        rand_rho_tr = (random.randint(-rho,rho),random.randint(-rho,rho))

        # four points for warped right image
        top_left_r2        = (start_x + 160 + rand_rho_tl[0],start_y + rand_rho_tl[1])
        bottom_left_r2     = (start_x + 160 + rand_rho_bl[0],start_y + image_h + rand_rho_bl[1])
        bottom_right_r2    = (start_x + 160 + image_w + rand_rho_br[0],start_y + image_h + rand_rho_br[1])
        top_right_r2       = (start_x + 160 + image_w + rand_rho_tr[0],start_y + rand_rho_tr[1])

        four_r2_points = [top_left_r2,bottom_left_r2,bottom_right_r2,top_right_r2]

        
        homography_right = cv2.getPerspectiveTransform(np.float32(four_r1_points), np.float32(four_r2_points) )
        warped_image = cv2.warpPerspective(image.copy(), homography_right, (image_full_w+rho*2,image_full_h+rho*2))
        image_right = warped_image[top_left_r1[1]:bottom_right_r1[1],top_left_r1[0]:bottom_right_r1[0]].copy()
        cv2.imwrite('train/image_right/'+str(count) +'.png',image_right)

        # four points to find homography matrix from cropped right image to original right image
        top_left_r3        = (start_x  + rand_rho_tl[0],start_y + rand_rho_tl[1])
        bottom_left_r3     = (start_x  + rand_rho_bl[0],start_y + image_h + rand_rho_bl[1])
        bottom_right_r3    = (start_x  + image_w + rand_rho_br[0],start_y + image_h + rand_rho_br[1])
        top_right_r3       = (start_x  + image_w + rand_rho_tr[0],start_y + rand_rho_tr[1])

        four_r3_points = [top_left_r3,bottom_left_r3,bottom_right_r3,top_right_r3]
        
        homography_left2right = cv2.getPerspectiveTransform(np.float32(four_r3_points), np.float32(four_r1_points))
        file = open('train/gt/' + str(count) + '.txt',"w")
        file.write(str(homography_left2right[0][0]) +" ")
        file.write(str(homography_left2right[0][1]) +" ")
        file.write(str(homography_left2right[0][2] )+" ")
        file.write(str(homography_left2right[1][0]) +" ")
        file.write(str(homography_left2right[1][1]) +" ")
        file.write(str(homography_left2right[1][2]) +" ")
        file.write(str(homography_left2right[2][0]) +" ")
        file.write(str(homography_left2right[2][1]))
        file.close()
        
        file = open('train/gt_point/' + str(count) + '.txt',"w")
        file.write(str(rand_rho_tl[0]) +" ")
        file.write(str(rand_rho_tl[1]) +" ")
        file.write(str(rand_rho_bl[0] )+" ")
        file.write(str(rand_rho_bl[1]) +" ")
        file.write(str(rand_rho_br[0]) +" ")
        file.write(str(rand_rho_br[1]) +" ")
        file.write(str(rand_rho_tr[0]) +" ")
        file.write(str(rand_rho_tr[1]))
        file.close()
        count+=1
        print(count)


if __name__ == '__main__':
    #runTrain('NewNet')
    #runTrain('PaoNet_l2')
    runPredict('DLT')
    runEvaluation('DLT')
    #runPredict('DeepNet')  
    #runEvaluation('DeepNet')
    #runPredict('PaoNet')
    #runEvaluation('PaoNet')
    
    #runPredict('NewNet')
    #runEvaluation('NewNet')
    