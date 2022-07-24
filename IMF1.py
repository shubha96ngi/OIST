#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 00:37:15 2022

@author: macintoshhd
"""

import numpy as np
import pandas as pd
import cv2,os,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage import measure
from skimage.filters.rank import entropy
from skimage.morphology import disk
from PIL import Image
from scipy import stats
from scipy import *
from PIL import Image
#read image from directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".tif"):
            img = cv2.imread(os.path.join(folder, filename))
            #if (0<abs(img.shape[1]-256)<100 or 0<abs(img.shape[0]-256)<100):
             #   img = cv2.resize(img, (256,256))
            #elif (0<abs(img.shape[1]-512)<100 or 0<abs(img.shape[0]-512)<100):
             #   img = cv2.resize(img, (512,512))
            #print('1=', np.shape(img))
            images.append(img)
    return images
root_folder = ''

folders = [os.path.join(root_folder, x) for x in ('o1', 'o2')]
imgs = [img for folder in folders for img in load_images_from_folder(folder)]

#convert images to grayscale imagesm
def load_images(gray_folder):
    gray_images = []
    for filename in os.listdir(gray_folder):
        if filename.endswith(".tif"):
            img = cv2.imread(os.path.join(gray_folder, filename))
            #if (0<abs(img.shape[1]-256)<100 or 0<abs(img.shape[0]-256)<100):
             #   img = cv2.resize(img, (256,256))
            #elif (0<abs(img.shape[1]-512)<100 or 0<abs(img.shape[0]-512)<100):
             #   img = cv2.resize(img, (512,512))
           # print('1=', np.shape(img))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_images.append(gray_img)
    return gray_images

gray_folders = [os.path.join(root_folder, x) for x in ('o1', 'o2')]
gray_imgs = [gray_img for gray_folder in gray_folders for gray_img in load_images(gray_folder)]
for j in range(len(gray_imgs)):
    cv2.imwrite('/flash/TerenzioU/program/shubhangi/gray/gray_{}.png'.format(j), gray_imgs[j])

def sp_idx(s, index=True):
    u = np.unique(s)
    return [np.where(s == i) for i in u]

def numberofsegments():
    b = np.empty((0, 100))
    for j in range(len(imgs)):
        segments_slic = slic(imgs[j], n_segments=1800, compactness=10, sigma=1, start_label=1)
        a = len(np.unique(segments_slic))
        b = np.append([b], [a])
    return b
#Calculating the SNR ratio
def signaltonoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m/sd)

for j in range(len(gray_imgs)):
    print('j=', j)
    #entropy_img = entropy(gray_imgs[j], disk(10))
    #cv2.imwrite('/flash/TerenzioU/program/shubhangi/entropy/e1_'+str(j)+'.png',entropy_img)

    #First SLIC segmentation of imgs[j]
    segments_slic = slic(imgs[j], n_segments=1800, compactness=10, sigma=1, start_label=1)
    segments_ids = np.unique(segments_slic)
    #print('segments_ids=', segments_ids)
    id_max = np.max(segments_ids)
    print('id= ', id_max)
    superpixel_list = sp_idx(segments_slic)
    superpixel = [idx for idx in superpixel_list]
    reimg = mark_boundaries(imgs[j], segments_slic)
    #for segVal in np.unique(segments_slic):
        
      #reimg = cv2.putText(mark_boundaries(imgs[j], segments_slic), "########{}".format(segVal + 1), (175, 150 - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    reim = mark_boundaries(reimg, segments_slic)*255
       # cv2.imwrite('/flash/TerenzioU/program/shubhangi/resegment/remask2_'+str(j)+'_'+str(segVal-1)+'.png', reim)
    #fig, ax = plt.subplots()
    #ax.imshow(mark_boundaries(reimg, segments_slic)*255)
    #plt.axis('off')
    #plt.savefig('/flash/TerenzioU/program/shubhangi/shubhangi/resegment/remask_'+str(j)+'.png',pad_inches = 0)
    #cv2.imwrite('/flash/TerenzioU/program/shubhangi/res1/remask2_'+str(j)+'.png', reim)
    w = []; h = []; rad=[];rad2=[];centx = []; centy = []; intensity=[];  rows = []; cols = []
    sp_x = []; sp_y = []; sp_xmin = []; sp_xmax = []; sp_ymin = []; sp_ymax = [];sp_T1 = []
    sp_SNR =[]; sp_entropy = []
    im_no = []; im_sp_area = []; im_sp_intensity = []; im_sp_eccentricity = []; im_sp_gray_avg = []
    sp_mask2 = []; sp_cent_px = []
    p = []; q = []
    p1=[];p2=[];p3=[];lp3=[];sv1=[];sv2=[];sv3=[]

    # regionprops properties although not useful
    regions = measure.regionprops(segments_slic, intensity_image=gray_imgs[j])
    for r in regions:
        sp_area = r.area
        sp_eccentricity = r.eccentricity
        sp_mean_intensity = r.mean_intensity
        
        im_sp_area.append(sp_area)
        im_sp_eccentricity.append(sp_eccentricity)
        im_sp_gray_avg.append(sp_mean_intensity)

    x=[0 for i in range(len(superpixel))]
    y=[0 for i in range(len(superpixel))] 
    X_dataset = []; ee1 =[]; ee2=[]; es1 =[]; es2=[];ex1=[];ey1=[]
    for segVal in np.unique(segments_slic):           #segval is im_sp_centroid=[] 1,2,3,4,5,6,7,8,9 i.e. superpixels
#
        #creating a mask and saving each segment in a folder mask2
        mask = np.ones(gray_imgs[j].shape[:2], dtype='uint8') #   self.height, self.width = img.shape[:2]
        mask[segments_slic == segVal] = 255
        pos = np.where(mask == 255)
    #print('1=', np.shape(imgs[j]))
        #properties of each superpixel
        #print('superpixel =', superpixel[segVal])
       
        x = pos[:][0]  #  XY = np.array([superpixel[i][0], superpixel[i][1]]).T
        y = pos[:][1]
        
        # calculating entropy of each superpixel
        c1 = np.array(imgs[j][x,y])
        values, counts = np.unique(c1, return_counts=True)
        #counts = c1.value_counts()
        entropy_sp = stats.entropy(counts)
        if entropy_sp >2.5:
            ee1.append(entropy_sp)
            es1.append(segVal-1)
            ex1.append(x)
            ey1.append(y)
            
        else:
            ee2.append(entropy_sp)
            es2.append(segVal-1)
        sp_entropy.append(entropy_sp)
        len_x = len(x); len_y = len(y)
        ymin = np.min(pos[:][1]); ymax = np.max(pos[:][1])
        xmin = np.min(pos[:][0]); xmax = np.max(pos[:][0])
        sp_xmin.append(xmin); sp_xmax.append(xmax)
        sp_ymin.append(ymin); sp_ymax.append(ymax)
        cx = np.mean(x); cy = np.mean(y)
        #reimg = cv2.putText(mark_boundaries(imgs[j], segments_slic), "{}".format(segVal + 1), (int(cx), int(cy)),cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
        #reim = mark_boundaries(reimg, segments_slic)*255
        #cv2.imwrite('/flash/TerenzioU/program/shubhangi/resegment/remask2_'+str(j)+'_'+str(segVal-1)+'.png', reim)

        width = xmax - xmin + 1; w.append(width)
        height = ymax - ymin + 1; h.append(height)
        radius = width/2; rad.append(radius)
        radius2 =height/2; rad2.append(radius2)
        sp_x.append(x); sp_y.append(y)
        centx.append(cx); centy.append(cy)
        im_no.append(j)  
        l1=[];l2=[]
        for list1 in x:
            l1.append(list1)
        for list2 in y:
            l2.append(list2)
        img1 = imgs[j][x,y]
        img2 = np.array(img1)/255
        X_dataset.append(img2)
        img = np.array(img1)
        img3= Image.fromarray(img1)
        data = np.zeros((932,932,3), dtype=np.uint8)       #    32x32 patch 
        data[0:931, 0:931] = [255,128,0]
        data[l1,l2] = np.array(img1)
        img4 = Image.fromarray(data)
        img4.save('/flash/TerenzioU/program/shubhangi/patch2/patch_'+str(j)+'_'+str(segVal-1)+'.png')
        im2 = cv2.imread('/flash/TerenzioU/program/shubhangi/patch2/patch_'+str(j)+'_'+str(segVal-1)+'.png')    
        im2= im2[xmin:xmax, ymin:ymax]
        #print('1=', np.shape(im2))   
        #im3 = cv2.resize(im2, (8, 8))
        cv2.imwrite('/flash/TerenzioU/program/shubhangi/m2/mask_'+str(j)+'_'+str(segVal-1)+'.png', im2)
        #cv2.imwrite('/flash/TerenzioU/program/shubhangi/m4/mask_'+str(j)+'_'+str(segVal-1)+'.png', im3)
        #calculating SNR ratio of each superpixel
        SNR = signaltonoise(imgs[j][x,y],0,0)     #print('SNR=', SNR)
        sp_SNR.append(SNR)
        # #centerx,centery = [np.mean(np.nonzero(segments_slic == i), axis=1) for i in segments_ids]
        row = np.size(superpixel[segVal-1][0])
        col =  np.size(superpixel[segVal-1][1])
        rows.append(row);cols.append(col)
    
    sx  = pd.DataFrame(sp_x).to_csv('/flash/TerenzioU/program/shubhangi/y2/x_'+str(j)+'.csv', sep=',', index=True, header=True)
    sy  = pd.DataFrame(sp_y).to_csv('/flash/TerenzioU/program/shubhangi/x2/y_'+str(j)+'.csv', sep=',', index=True, header=True)
    bx = pd.read_csv('/flash/TerenzioU/program/shubhangi/y2/x_'+str(j)+'.csv')                #.dropna()          
    by = pd.read_csv('/flash/TerenzioU/program/shubhangi/x2/y_'+str(j)+'.csv')                #.dropna()
    f2=[]
    for segVal in np.unique(segments_slic):
# making a  sliding window to calculate IMF and entropy
#don't touch this
        f1=[]      
        for a1, b1, c1, d1, e1, a2, b2, c2, d2, e2 in zip(bx.iloc[segVal-1 ,1:rows[segVal-1]].astype(int), bx.iloc[segVal-1, 2:rows[segVal-1]-1].astype(int), bx.iloc[segVal-1 ,3:rows[segVal-1]-2].astype(int), bx.iloc[segVal-1, 4:rows[segVal-1]-3].astype(int), bx.iloc[segVal-1 ,5:rows[segVal-1]-4].astype(int), by.iloc[segVal-1, 1:rows[segVal-1]].astype(int), by.iloc[segVal-1 ,2:rows[segVal-1]-1].astype(int), by.iloc[segVal-1, 3:rows[segVal-1]-2].astype(int), by.iloc[segVal-1 ,4:rows[segVal-1]-3].astype(int), by.iloc[segVal-1, 5:rows[segVal-1]-4].astype(int)):
            mask2 = np.array([gray_imgs[j][a1, a2], gray_imgs[j][b1, a2], gray_imgs[j][b1, b2], gray_imgs[j][c1, a2], gray_imgs[j][c1, b2], gray_imgs[j][c1, c2], gray_imgs[j][c1, d2], gray_imgs[j][d1, b2], gray_imgs[j][d1, c2], gray_imgs[j][d1, d2], gray_imgs[j][d1, e2], gray_imgs[j][e2, d2], gray_imgs[j][e1, e2]])
            cent_px = [gray_imgs[j][c1, c2]]
            #print('mask2=', mask2); print('center_pixel=', cent_px)
            f =[]
            for k in range(len(mask2)):
                #print('k=',k) 
                p_i = cent_px; pj = mask2; p_j = pj[k]
                if abs(p_i-p_j) <= 3:
                    e1 = 0
                    #f = np.append()
                    f.append(e1)
                if 3<abs(p_i-p_j) <= 12:
                    e2 = math.exp(abs(p_i-p_j)/3)
                    f.append(e2)
                if abs(p_i-p_j) > 12:
                    e3 = math.exp(4)
                    f.append(e3)
            #print('f=', f)
            #print('shape of f =', np.shape(f))
            f1.append(f)
            p.append(mask2); q.append(cent_px)
        #f1.append(f[segVal])
        #print('f =', f)
        #print('f is equal to=',np.shape(f1))
        #print('SegVal=',segVal)
        IMF = np.sum(f1)/(13*len(superpixel[segVal-1][0]))
        #print('IMF=', IMF)
        f2.append(IMF)
        if IMF >=29:
            P_ST = superpixel[segVal-1][0]
            p1.append(P_ST)
            sv1.append(segVal-1)
        #elif IMF <25:
         #   P_UD = superpixel[segVal-1][0]
          #  p2.append(P_UD)
           # sv2.append(segVal-1)
        else:
            P_OB = superpixel[segVal-1][0]
            lP_OB = len(P_OB)
            lp3.append(lP_OB)
            p3.append(P_OB)
            sv3.append(segVal-1)

    print('length of lp3=', len(lp3))
    p_up = np.max(lp3); p_fl = np.min(lp3)
    d_p = (p_up-p_fl)/len(lp3)
    print('p_up=', p_up); print('p_fl=', p_fl)
    print('delta p=', d_p)
# if we devide ith superpixel in P_OB into 10 parts. and calculate
#number of pixel in each part and then find one with max number of pixel
# and the second max and then define function A and theta and thus calculate u1 and u2
    #read image from directory
    
    #v = np.column_stack([im_no,segments_ids,centx,centy,w,h,im_sp_area,sp_SNR,sp_entropy,im_sp_eccentricity,f2,im_sp_gray_avg])
    v = np.column_stack([im_no,segments_ids,centx,centy,w,h,im_sp_area,sp_SNR,sp_entropy,im_sp_eccentricity,im_sp_gray_avg])
    dfv = pd.DataFrame(v).to_csv('/flash/TerenzioU/program/shubhangi/d2/im_'+str(j)+'_data.csv', sep=',', index=True, header=True)
    #v1 = pd.DataFrame(f2).to_csv('/flash/TerenzioU/program/shubhangi/IMF1/im_'+str(j)+'_IMF.csv', sep=',', index=True, header=True)
    #v2 = pd.DataFrame(sv1).to_csv('/flash/TerenzioU/program/shubhangi/sv3/sp_'+str(j)+'_data.csv' , sep=',', index=True, header=True)
    #v3 = pd.DataFrame(sv2).to_csv('/flash/TerenzioU/program/shubhangi/sv2/sp_'+str(j)+'_data.csv' , sep=',', index=True, header=True)
    #v4 = pd.DataFrame(sv3).to_csv('/flash/TerenzioU/program/shubhangi/sv4/sp_'+str(j)+'_data.csv' , sep=',', index=True, header=True)
    #v5 = np.column_stack([im_sp_eccentricity,sp_entropy])
    #v6 = pd.DataFrame(v5).to_csv('/flash/TerenzioU/program/shubhangi/ecc/sp_'+str(j)+'_e.csv' , sep=',', index=True, header=True)
    '''
    v6 = pd.DataFrame(im_sp_eccentricity).to_csv('/flash/TerenzioU/program/shubhangi/ecc1/sp_'+str(j)+'_e.csv' , sep=',', index=True, header=True)
    v7 = np.column_stack([es1,ee1])
    v8 = pd.DataFrame(v7, columns=['sp_ID', 'eccentricity']).to_csv('/flash/TerenzioU/program/shubhangi/en1/sp_'+str(j)+'_e.csv' , sep=',', index=True, header=True)
    '''
    
    v7 = np.column_stack([es1,ee1])
    v8 = pd.DataFrame(v7, columns=['sp_ID', 'eccentricity']).to_csv('/flash/TerenzioU/program/shubhangi/en2/sp_'+str(j)+'_e.csv' , sep=',', index=True, header=True)
    '''
    v9 = np.column_stack([es2,ee2])
    v10 = pd.DataFrame(v9, columns=['sp_ID', 'eccentricity']).to_csv('/flash/TerenzioU/program/shubhangi/en2/sp_'+str(j)+'_e.csv' , sep=',', index=True, header=True)
    '''
    #v6 = pd.DataFrame(p_fl).to_csv('/flash/TerenzioU/program/shubhangi/pmin/p_fl.csv' , sep=',', index=True, header=True)
    #v7 = pd.DataFrame(d_p).to_csv('/flash/TerenzioU/program/shubhangi/delta/d_p.csv' , sep=',', index=True, header=True)
    
    #   ,columns=['Img_no','sp_ID','cent_X','cent_Y','width','height','Area','SNR','Entropy','eccentricity','IMF','Intensity_avg']).to_csv('/flash/TerenzioU/program/shubhangi/data/im_'+str(j)+'_data.csv', sep=',', index=False, header=True