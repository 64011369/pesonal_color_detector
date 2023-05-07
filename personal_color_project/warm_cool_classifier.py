import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

image_list = []
scores = []
means = []
red = 0
green = 0
blue = 0

# load filenames of datasets
for filename in glob.glob('./dataset/*.jpg'):
    image_list.append(filename)

print(image_list)

for img_dir in image_list:
    # read filename for naming convention
    filename = os.path.splitext(img_dir)

    # read image
    img = cv2.imread(img_dir)
    print(img_dir)

    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # face detection model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')

    # detecting face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20)
    nose = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=100)

    for (x, y, w, h) in faces:
        # determine forehead region
        forehead_y = int(y + h/8)
        forehead_h = int(h/8)
        forehead_x = int(x + w/2)
        forehead_w = int(w/15)

        # draw a rectangle around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0,), 2)

        # crop forehead region
        forehead = img[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_w]

        # standardize RGB values of the forehead
        r_f, g_f, b_f = cv2.split(forehead)
        r_std = np.std(r_f)
        g_std = np.std(g_f)
        b_std = np.std(b_f)
        r_mean = np.mean(r_f)
        g_mean = np.mean(g_f)
        b_mean = np.mean(b_f)
        r = (r_f - r_mean) / r_std
        g = (g_f - g_mean) / g_std
        b = (b_f - b_mean) / b_std

        # calculate the Warm/Cool score
        score = (np.mean(r) - np.mean(g)) * (np.mean(r) - np.mean(b)) / (np.mean(r) + np.mean(g) + np.mean(b))
        scores.append(score)
        print('R Mean:', r_mean)
        print('G Mean:', g_mean)
        print('B Mean:', b_mean)
        print('R Std:', r_std)
        print('G Std:', g_std)
        print('B Std:', b_std)
        print('Warm/Cool score (face):', score)

    for (x, y, w, h) in nose:
        # determine forehead region
        cheek_y = int(y + h/8)
        cheek_h = int(h/5)
        cheek_x = int(x + w/10)
        cheek_w = int(w/5)

        # draw a rectangle around eye  
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0,), 2)

        # crop eye region
        cheek = img[cheek_y:cheek_y+cheek_h, cheek_x:cheek_x+cheek_w]

        # standardize RGB values of the forehead
        r_n, g_n, b_n = cv2.split(cheek)
        r_std = np.std(r_n)
        g_std = np.std(g_n)
        b_std = np.std(b_n)
        r_mean = np.mean(r_n)
        g_mean = np.mean(g_n)
        b_mean = np.mean(b_n)
        r = (r_n - r_mean) / r_std
        g = (g_n - g_mean) / g_std
        b = (b_n - b_mean) / b_std

        # calculate the Warm/Cool score
        score = (np.mean(r) - np.mean(g)) * (np.mean(r) - np.mean(b)) / (np.mean(r) + np.mean(g) + np.mean(b))
        scores.append(score)

        print('R Mean:', r_mean)
        print('G Mean:', g_mean)
        print('B Mean:', b_mean)
        print('R Std:', r_std)
        print('G Std:', g_std)
        print('B Std:', b_std)
        print('Warm/Cool score (cheek):', score)
    
    #plot
    fig, ax = plt.subplots()
    ax.hist(r_f.ravel(), bins=256, range=(0, 256), color='b', alpha=0.5)
    ax.hist(g_f.ravel(), bins=256, range=(0, 256), color='g', alpha=0.5)
    ax.hist(b_f.ravel(), bins=256, range=(0, 256), color='r', alpha=0.5)
    ax.set_xlim([0, 256])
    ax.set_ylim([0, forehead.shape[0]*forehead.shape[1]/2])
    ax.set_title('{}\'s forehead RGB Histogram'.format(filename[0]))
    ax.set_xlabel('Value')
    ax.set_ylabel('Pixel Count')

    #save plot
    plt.savefig('{}_forehead.png'.format(filename[0]))
    plt.show()
    plt.clf()
    
    #plot
    fig, ax = plt.subplots()
    ax.hist(r_n.ravel(), bins=256, range=(0, 256), color='b', alpha=0.5)
    ax.hist(g_n.ravel(), bins=256, range=(0, 256), color='g', alpha=0.5)
    ax.hist(b_n.ravel(), bins=256, range=(0, 256), color='r', alpha=0.5)
    ax.set_xlim([0, 256])
    ax.set_ylim([0, cheek.shape[0]*cheek.shape[1]/2])
    ax.set_title('{}\'s cheek RGB Histogram'.format(filename[0]))
    ax.set_xlabel('Value')
    ax.set_ylabel('Pixel Count')
 
    #save plot
    plt.savefig('{}_cheek.png'.format(filename[0]))
    plt.show()
    plt.clf()

    # show images
    cv2.imshow('original img', img)
    cv2.imshow('forehead', forehead)
    cv2.imshow('cheek', cheek)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    score = np.mean(np.sum(scores))

    print('The Undertone is Likely to be')
    if score > 0:
        print('==========\nWARM\n==========\n')
    else:
        print('**********\nCOOL\n**********\n')
    print('\n')

    scores = []
