"""
Created on Mon May 28 04:16:47 2018

@author: dmdm02

The loading model codes is modified from Andrew Ng's deep learning course.
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from inception_blocks import *
from PIL import Image, ImageDraw, ImageFont

from keras.models import load_model
import tensorflow as tf
from numpy import genfromtxt
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from align import AlignDlib


################## CONSTANT ########################
frontal_face_model = 'Libraries/haarcascade_frontalface_alt.xml'
face_recog_architecture = 'Libraries/Facenet_architecture.json'
face_recog_weights = 'Libraries/Facenet_weights.h5'
font = cv2.FONT_HERSHEY_SIMPLEX
alignment = AlignDlib('Libraries/landmarks.dat')


################# LOADING THE MODEL #######################

WEIGHTS = [
  'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
  'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
  'inception_3a_pool_conv', 'inception_3a_pool_bn',
  'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
  'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
  'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
  'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
  'inception_3b_pool_conv', 'inception_3b_pool_bn',
  'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
  'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
  'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
  'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
  'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
  'inception_4a_pool_conv', 'inception_4a_pool_bn',
  'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
  'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
  'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
  'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
  'inception_5a_pool_conv', 'inception_5a_pool_bn',
  'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
  'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
  'inception_5b_pool_conv', 'inception_5b_pool_bn',
  'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
  'dense_layer'
]

conv_shape = {
  'conv1': [64, 3, 7, 7],
  'conv2': [64, 64, 1, 1],
  'conv3': [192, 64, 3, 3],
  'inception_3a_1x1_conv': [64, 192, 1, 1],
  'inception_3a_pool_conv': [32, 192, 1, 1],
  'inception_3a_5x5_conv1': [16, 192, 1, 1],
  'inception_3a_5x5_conv2': [32, 16, 5, 5],
  'inception_3a_3x3_conv1': [96, 192, 1, 1],
  'inception_3a_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_3x3_conv1': [96, 256, 1, 1],
  'inception_3b_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_5x5_conv1': [32, 256, 1, 1],
  'inception_3b_5x5_conv2': [64, 32, 5, 5],
  'inception_3b_pool_conv': [64, 256, 1, 1],
  'inception_3b_1x1_conv': [64, 256, 1, 1],
  'inception_3c_3x3_conv1': [128, 320, 1, 1],
  'inception_3c_3x3_conv2': [256, 128, 3, 3],
  'inception_3c_5x5_conv1': [32, 320, 1, 1],
  'inception_3c_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_3x3_conv1': [96, 640, 1, 1],
  'inception_4a_3x3_conv2': [192, 96, 3, 3],
  'inception_4a_5x5_conv1': [32, 640, 1, 1,],
  'inception_4a_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_pool_conv': [128, 640, 1, 1],
  'inception_4a_1x1_conv': [256, 640, 1, 1],
  'inception_4e_3x3_conv1': [160, 640, 1, 1],
  'inception_4e_3x3_conv2': [256, 160, 3, 3],
  'inception_4e_5x5_conv1': [64, 640, 1, 1],
  'inception_4e_5x5_conv2': [128, 64, 5, 5],
  'inception_5a_3x3_conv1': [96, 1024, 1, 1],
  'inception_5a_3x3_conv2': [384, 96, 3, 3],
  'inception_5a_pool_conv': [96, 1024, 1, 1],
  'inception_5a_1x1_conv': [256, 1024, 1, 1],
  'inception_5b_3x3_conv1': [96, 736, 1, 1],
  'inception_5b_3x3_conv2': [384, 96, 3, 3],
  'inception_5b_pool_conv': [96, 736, 1, 1],
  'inception_5b_1x1_conv': [256, 736, 1, 1],
}

def load_weights_from_FaceNet(FRmodel, dirPath):
    '''
    Load weights from csv files (which was exported from Openface torch model)
    
    Arguments:
    ---------
        FRmodel:
            The desired model
        dirPath:
            Path to the weights
    
    '''
    weights = WEIGHTS
    weights_dict = load_weights(dirPath)

    # Set layer weights of the model
    for name in weights:
        if FRmodel.get_layer(name) != None:
            FRmodel.get_layer(name).set_weights(weights_dict[name])
        elif model.get_layer(name) != None:
            model.get_layer(name).set_weights(weights_dict[name])

def load_weights(dirPath):
    # Set weights path
    fileNames = filter(lambda f: not f.startswith('.'), os.listdir(dirPath))
    paths = {}
    weights_dict = {}

    for n in fileNames:
        paths[n.replace('.csv', '')] = dirPath + '/' + n

    for name in WEIGHTS:
        if 'conv' in name:
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            conv_w = np.reshape(conv_w, conv_shape[name])
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            weights_dict[name] = [conv_w, conv_b]     
        elif 'bn' in name:
            bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dense_w = genfromtxt(dirPath+'/dense_w.csv', delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt(dirPath+'/dense_b.csv', delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]

    return weights_dict

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.square(anchor-positive)
    neg_dist = tf.square(anchor-negative)
    basic_loss = tf.reduce_sum(pos_dist-neg_dist) + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss,0.))
    return loss
    

############### IMPORTING IMAGES AND EXTRACT FACES ##################

def import_image(image_path, plot=False):
    '''
    Import the image.
    
    Arguments:
    ---------
        image_path:
            Path to the image
        plot:
            Plot or not
            
    Returns:
    --------
        img:
            Array of type uint8 contains the RGB values of the image
    '''
    img_orig = cv2.imread(image_path,1)
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    if plot==True:
        plt.figure()
        plt.imshow(img)
    return img

def detect_face_from_image_path(image_path, face_cascade, window_ratio=1.2):
    if not os.path.exists(frontal_face_model_file_path):
        print('failed to find face detection opencv model: ', frontal_face_model_file_path)

    face_cascade = cv2.CascadeClassifier(frontal_face_model_file_path)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    print('faces detected: ', len(faces))
    for (x, y, w, h) in faces:
        
        center_x = x+w/2.
        center_y = y+h/2.
        x = int(center_x - w/2*window_ratio)
        y = int(center_y - h/2*window_ratio)
        w = int(w*window_ratio)
        h = int(h*window_ratio)
        
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def get_faces_from_image(image, window_ratio=1.0, face_num=False, plot=False):
    '''
    Get all faces in an image.
    
    Arguments:
    --------
        image:
            input image, type uint8
        aligment:
            Opencv face detection model
        window_ratio:
            ratio of output window and face detection window
        face_num:
            return number of face or not
        plot:
            plot the image or not
        
    Returns:
    -------
        output_faces:
            list containing all faces in the image
    '''
    img = copy.deepcopy(image)
    boxes = alignment.getAllFaceBoundingBoxes(image)
    if face_num == True:
        print('faces detected: ', len(boxes))
    
    output_faces = []
    face_pos = []
    for box in boxes:
        x, y, w, h = box.left(), box.top(), box.width(), box.height()
        this_face = image[y:y+h, x:x+w]
        output_faces.append(this_face)
        face_pos.append(box)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    if plot==True:
        '''
        fig,ax = plt.subplots(1)
        plt.imshow(img)
        for (x,y,w,h) in faces:
            rect = patches.Rectangle((x,y), w, h, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
        '''
        plt.imshow(img)
                
    return output_faces, face_pos, img

def image_to_encoding(image, FRmodel, bb=None, aligned=True):
    '''
    Embedding the image
    Arguments:
    --------
        image: Image numpy array of type uint8
        FRmodel:
            Predicting model
        
    Returns:
    --------
        embedding:
            Embedding of the image
    '''
    #img_resize = cv2.resize(image,(96,96))
    if aligned == True:
        img_resize = face_aligned(image, bb=bb)
    else:
        img_resize = cv2.resize(image, (96,96))
    img = np.around(np.transpose(img_resize, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = FRmodel.predict_on_batch(x_train)
    return embedding

def show_pair(image1, image2, FRmodel, aligned=False):
    embedding1 = image_to_encoding(image1, FRmodel, aligned=aligned)
    embedding2 = image_to_encoding(image2, FRmodel, aligned=aligned)
    distance = np.linalg.norm(embedding1-embedding2)
    if aligned==True:
        face_list, face_pos, image1 = get_faces_from_image(image1, plot=False)
        face_list, face_pos, image2 = get_faces_from_image(image2, plot=False)
    plt.figure(figsize=(10,4))
    plt.suptitle('Distance = %.2f' %distance)
    plt.subplot(121)
    plt.imshow(image1)
    plt.subplot(122)
    plt.imshow(image2)

def get_person_name(image, FRmodel, database, threshold=None, bb=None):
    '''
    Return name of the person in the image
    
    Arguments:
    ---------
        image:
            Image numpy array of type uint8
        FRmodel:
            face recognition model
        database:
            Database that stores all the label
        
    Returns:
    --------
        label:
            Label of the person in the image
    '''
    if threshold==None:
        threshold = 0.7
    embedding = image_to_encoding(image, FRmodel, bb=bb)
    min_dist = 100.
    identity = None
    for (name, encodes) in database.items():
        dist = np.linalg.norm(encodes-embedding)
        if dist < min_dist:
            identity = name
            min_dist = dist
    
    if min_dist > threshold:
        label = 'Not in database'
    else:
        label = identity
        label = ''.join([i for i in identity if not i.isdigit()])
        label = label.replace('_',' ')
        label = label.replace('-','')
    
    return label, min_dist
            
def face_recognition(image, FRmodel, database, threshold=None, plot=True, faces_out=False):
    '''
    Identify all faces presented in given image
    
    Arguments:
    ---------
        image:
            Image numpy array of type uint8
        FRmodel:
            face recognition model
        database:
            Database that stores all the label
        threshold:
            Value above which return 'Not in database'
        plot:
            plot or not
        faces_out:
            return face dictionary or not
        
    Returns:
    --------
        faces_out:
            Dictionary contained all identified face
    '''
    faces, face_pos, image_face = get_faces_from_image(image)    
    image_out = Image.fromarray(image_face)
    draw = ImageDraw.Draw(image_out)
    font = ImageFont.truetype(font='Font/FiraMono-Medium.otf',size=np.floor(2e-2 * image_out.size[1] + 0.5).astype('int32'))
    
    output_faces = {}
    for i in range(len(face_pos)):
        bb = face_pos[i]
        x,y,w,h = bb.left(), bb.top(), bb.width(), bb.height()
        label, min_dist = get_person_name(image, FRmodel, database, threshold=threshold, bb=bb)
        output_faces[label] = faces[i]
        '''
        plt.text(x, y, label, size=10, color='white',
                 bbox=dict(facecolor='blue', edgecolor='blue'))
        '''
        #image_out = cv2.putText(image_out, label, (x,y+h), font, 1, (255,255,255),1)
 
        label_size = draw.textsize(label, font)
        draw.rectangle([(x,y-label_size[1]/2),(x+label_size[0],y+label_size[1]/2)],fill='blue')
        draw.text((x,y-label_size[1]/2), label, fill='white', font=font)
        
    del draw
    image_out = np.array(image_out)
    if plot==True:
        plt.imshow(image_out)
        
    if faces_out == True:
        return output_faces
    
    return image_out                   

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

def face_aligned(image,bb=None):
    if bb == None:
        bb = alignment.getLargestFaceBoundingBox(image)
    return alignment.align(96, image, bb=bb,
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    



'''
FRModel = faceRecoModel(input_shape=(3,96,96))


frontal_face_model_file_path = 'Libraries/haarcascade_frontalface_alt.xml'
img = import_image('1.jpg', plot=False)
output = get_faces_from_image(frontal_face_model_file_path, img)
plt.imshow(output[0])

    
img1 = cv2.imread('1.jpg',1)
image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (50, 50)) 
plt.imshow(resized_image)
plt.imshow(image)

img = np.around(np.transpose(img1, (2,0,1))/255.0, decimals=12)
x_train = np.array([img])
embedding = model.predict_on_batch(x_train)
a = np.transpose(img1, (0,1,2))/255.0
plt.imshow(img)
'''
