#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 05:08:37 2022

@author: macintoshhd
"""



#creating a dataset for soma detection
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import color,filters
from skimage.future.graph import rag
from skimage.io import imsave, imread
from skimage.segmentation import slic, mark_boundaries
from skimage.feature import blob_doh
from skimage import measure
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, KBinsDiscretizer, OneHotEncoder
from scipy import *
from scipy.spatial import distance # Gaurav added the following line
import os
from PIL import Image, ImageDraw, ImageOps
plt.rcParams.update({'figure.max_open_warning': 0})
from datetime import datetime
from itertools import combinations
from collections import namedtuple
from pyvis.network import Network
import networkx as nx

#import numpy_indexed as npi
start_time = datetime.now()
# do your work here

pi = math.pi

#read image from directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".tif"):
            img = cv2.imread(os.path.join(folder, filename))
          
            images.append(img)
    return images
root_folder = ''

folders = [os.path.join(root_folder, x) for x in ('o1', 'o2')]
imgs = [img for folder in folders for img in load_images_from_folder(folder)]

#convert images to grayscale images
def load_images(gray_folder):
    gray_images = []
    for filename in os.listdir(gray_folder):
        if filename.endswith(".tif"):
            img = cv2.imread(os.path.join(gray_folder, filename))
           
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_images.append(gray_img)
    return gray_images

gray_folders = [os.path.join(root_folder, x) for x in ('o1', 'o2')]
gray_imgs = [gray_img for gray_folder in gray_folders for gray_img in load_images(gray_folder)]
for j in range(len(gray_imgs)):
    cv2.imwrite('/flash/TerenzioU/program/shubhangi/gray/gray_{}.png'.format(j), gray_imgs[j])
# apply SLIC and extract (approximately) the supplied number of segments
def sp_idx(s, index=True):
    u = np.unique(s)
    return [np.where(s == i) for i in u]

def numberofsegments():
    b = np.empty((0, 100))
    for j in range(len(gray_imgs)):
        segments_slic = slic(imgs[j], n_segments=12000, compactness=10, sigma=1, start_label=1)
        a = len(np.unique(segments_slic))
        b = np.append([b], [a])
    return b

def numberofsegments2():
    b = np.empty((0, 100))
    for j in range(len(gray_imgs)):
        segments_slic2 = slic(imgs[j], n_segments=1800, compactness=10, sigma=1, start_label=1)
        a = len(np.unique(segments_slic2))
        b = np.append([b], [a])
    return b

# def distance(p1, p2):
#     return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

a = numberofsegments()
im_segments_slic = []; a4 =[]; a7=[]
vdf = []; v1df=[]; v2df=[]; v3df=[]; z6df=[]; z11df=[]
aa = numberofsegments2()
im_segments_slic2 = []; a42 =[]; a72=[]
vdf2 = []; v1df2=[]; v2df2=[]; v3df2=[]; z6df2=[]; z11df2=[];vdf3=[] 

for j in range(5):
    
    print('j=', j)
    blobs_doh = blob_doh(gray_imgs[j], min_sigma=10,  max_sigma=20, threshold=.001, overlap=0.8)
    fig, ax = plt.subplots()
    circx=[];circy=[];circr=[];n1c=[];circa=[]
    sim = imgs[j].copy()
    sim1 = Image.open('/flash/TerenzioU/program/shubhangi/shubh/gray/gray_0.png')
    bl = 0
    for blob in blobs_doh:
        #print('blob=', blob)
        nc = j
        bl +=1
        yc, xc, rc = blob
        area = pi*math.pow(rc,2)
        circx.append(xc); circy.append(yc); circr.append(rc); circa.append(area)
        n1c.append(nc)
        draw = ImageDraw.Draw(sim1)
        draw.ellipse((xc-rc,yc-rc,xc+rc,yc+rc),fill= 'red')
        sim1.save('/flash/TerenzioU/program/shubhangi/shubh/circle/DoH_{}.png'.format(j))
    z11 = np.column_stack([circx, circy, circr, circr, circa])
    dfz11 = pd.DataFrame(z11, columns=['circ_x', 'circ_y', 'Radius', 'Radius2', 'Area'])                           #.to_csv('/flash/TerenzioU/program/shubhangi/center/c_'+str(j)+'.csv')

    #First SLIC segmentation of imgs[j]
    segments_slic = slic(sim, n_segments=12000, compactness=10, sigma=1, start_label=1)
    segments_ids = np.unique(segments_slic)
    
    id_max = np.max(segments_ids)
    #print('id= ', id_max)

    superpixel_list = sp_idx(segments_slic)
    superpixel = [idx for idx in superpixel_list]
   

    reimg = mark_boundaries(imgs[j], segments_slic)
    #cv2.imwrite('/flash/TerenzioU/program/re/remask_'+str(j)+'.png', mark_boundaries(reimg, segments_slic))
    #Calculating properties of each superpixel
    x=[0 for i in range(len(superpixel))]
    #print('length=', superpixel[15] )
    y=[0 for i in range(len(superpixel))] 
    #centers = np.array([np.mean(np.nonzero(segments_slic == i), axis=1) for i in segments_ids])
    w = []; h = []; rad=[];rad2=[];centx = []; centy = [];im_no=[];X_dataset=[]; minr =[]
    sp_x = []; sp_y = []; sp_xmin = []; sp_xmax = []; sp_ymin = []; sp_ymax = []
    sp_xl=[];sp_yl=[];cex=[];cey=[]

    #First SLIC segmentation of imgs[j]
    segments_slic2 = slic(imgs[j], n_segments=1800, compactness=10, sigma=1, start_label=1)
    segments_ids2 = np.unique(segments_slic2)
    id_max2 = np.max(segments_ids2)
    #print('id2= ', id_max2)
    superpixel_list2 = sp_idx(segments_slic2)
    superpixel2 = [idx for idx in superpixel_list2]
    reimg2 = mark_boundaries(imgs[j], segments_slic2)
    #cv2.imwrite('/flash/TerenzioU/program/re/remask2_'+str(j)+'.png', mark_boundaries(reimg, segments_slic2)*255)
    #Calculating properties of each superpixel
    x2=[0 for i in range(len(superpixel2))]
    y2=[0 for i in range(len(superpixel2))] 

    #centers = np.array([np.mean(np.nonzero(segments_slic == i), axis=1) for i in segments_ids])
    w2 = []; h2 = []; rad3=[];rad4=[]; centx2 = []; centy2 = []; im_no2=[];  X_dataset2 = []; minr2 =[]; maxr2=[]; fiii =[]; fii1=[]; sp_lx=[];sp_ly=[]; sp_dist=[]
    sp_x2 = []; sp_y2 = []; sp_ux2 = []; sp_uy2 = []; sp_xmin2 = []; sp_xmax2 = []; sp_ymin2 = []; sp_ymax2 = []; fcentx=[]; fcenty=[]; fi =[]; fi1 =[];fi2=[];fi3=[]
    sp_area2=[]
    for segVal in np.unique(segments_slic):           #segval is im_sp_centroid=[] 1,2,3,4,5,6,7,8,9 i.e. superpixels
#
        #creating a mask and saving each segment in a folder mask2
        mask = np.ones(imgs[j].shape[:2], dtype='uint8') #   self.height, self.width = img.shape[:2]
        mask[segments_slic == segVal] = 255
        pos = np.where(mask == 255)
        #properties of each superpixel
        x = pos[:][0]  #  XY = np.array([superpixel[i][0], superpixel[i][1]]).T
        y = pos[:][1]
        #print('x=', x)
        lx1 = list(x);ly1=list(y)
        lisxy = [(x,y) for x,y in zip(lx1,ly1)]
        sp_xl.append(lisxy)
        len_x = len(x); len_y = len(y)
        ymin = np.min(pos[:][1]); ymax = np.max(pos[:][1])
        xmin = np.min(pos[:][0]); xmax = np.max(pos[:][0])
        sp_xmin.append(xmin); sp_xmax.append(xmax)
        sp_ymin.append(ymin); sp_ymax.append(ymax)
        cx = np.mean(x); cy = np.mean(y)
        liscxy =(cx,cy)
        sp_yl.append(liscxy)
        width = xmax - xmin + 1; w.append(width)
        height = ymax - ymin + 1; h.append(height)
        radius = width/2; rad.append(radius)
        radius2 =height/2; rad2.append(radius2)
        minrad = min(int(radius), int(radius2))
        minr.append(minrad)
        sp_x.append(x); sp_y.append(y)
        centx.append(int(cx)); centy.append(int(cy))
        cex.append(cx);cey.append(cy)
        im_no.append(j)
        
        
    regions2 = measure.regionprops(segments_slic2, intensity_image=gray_imgs[j])
    for r in regions2:
        area2 = r.area
        sp_area2.append(area2)

    for segVal2 in np.unique(segments_slic2):           #segval is im_sp_centroid=[] 1,2,3,4,5,6,7,8,9 i.e. superpixels
#
        #creating a mask and saving each segment in a folder mask2
        mask2 = np.ones(imgs[j].shape[:2], dtype='uint8') #   self.height, self.width = img.shape[:2]
        mask2[segments_slic2 == segVal2] = 255
        pos2 = np.where(mask2 == 255)
        #properties of each superpixel
        x2 = pos2[:][0]  #  XY = np.array([superpixel[i][0], superpixel[i][1]]).T
        y2 = pos2[:][1]
        lx2 = list(x2); ly2= list(y2)
        #  i want to calculate center for nearly circular superpixel by finding out
        # maximum distance between two pairs and then getting mid point
        lisxy = [(x,y) for x,y in zip(lx2,ly2)]
       
        fiii.append(lisxy)
        sp_lx.append(lx2),sp_ly.append(ly2)
        dist = distance.cdist(np.array(lisxy), np.array(lisxy), 'euclidean')        
        fi = np.max(dist)
        fi1 = np.where(dist ==fi)
        sp_dist.append(fi/2)
       
        fi3.append(fi1)                 
        len_x2 = len(x2); len_y2 = len(y2)
        ymin2 = np.min(pos2[:][1]); ymax2 = np.max(pos2[:][1])
        xmin2 = np.min(pos2[:][0]); xmax2 = np.max(pos2[:][0])
        sp_xmin2.append(xmin2); sp_xmax2.append(xmax2)
        sp_ymin2.append(ymin2); sp_ymax2.append(ymax2)
        cx2 = np.mean(x2); cy2 = np.mean(y2)
        width2 = xmax2 - xmin2 + 1; w2.append(width2)
        height2 = ymax2 - ymin2 + 1; h2.append(height2)
        radius3 = width2/2; rad3.append(radius3)
        radius4 = height2/2; rad4.append(radius4)
        minrad2 = min(int(radius3), int(radius4))
        minr2.append(minrad2)
        maxrad2 = max(int(radius3), int(radius4))
        maxr2.append(maxrad2)
        sp_ux2.append(list(np.unique(x2))); sp_uy2.append(list(np.unique(y2)))
        sp_x2.append(x2); sp_y2.append(y2)
        centx2.append(cx2); centy2.append(cy2)
        im_no2.append(j)
    dfi = pd.DataFrame(fi3).to_csv('/flash/TerenzioU/program/shubhangi/shubh/ind/i_'+str(j)+'.csv', sep=',', index=True, header=True)
    dfi1 = np.column_stack([sp_lx, sp_ly,sp_dist])
    dfi2 = pd.DataFrame(dfi1, columns=['LX','LY','Radius']).to_csv('/flash/TerenzioU/program/shubhangi/shubh/lisxy/i_'+str(j)+'.csv', sep=',', index=True, header=True)
    

    v = np.column_stack([segments_ids, sp_y, sp_x, centy, centx,cey,cex, minr, w, h])
    dfv = pd.DataFrame(v, columns=['sp_ID', 'X', 'Y', 'cent_X', 'cent_Y','CX','CY', 'minr', 'width', 'height']).to_csv('/flash/TerenzioU/program/shubhangi/shubh/data/im_'+str(j)+'_data.csv', sep=',', index=False, header=True)

    v2 = np.column_stack([segments_ids2, sp_y2, sp_x2, centy2, centx2, rad3, rad4, minr2, maxr2, w2, h2, sp_ux2, sp_uy2])
    dfv2 = pd.DataFrame(v2, columns=['sp_ID', 'X', 'Y','cent_X', 'cent_Y', 'rad3', 'rad4', 'minr2', 'maxr2', 'width', 'height', 'UX', 'UY']).to_csv('/flash/TerenzioU/program/shubhangi/shubh/data1/im_'+str(j)+'_data.csv', sep=',', index=False, header=True)
    
    v3 = np.column_stack([centy2, centx2, minr2, maxr2, sp_area2])
    dfz12 = pd.DataFrame(v3, columns=['circ_x', 'circ_y', 'Radius','Radius2','Area'])                                  #.to_csv('/flash/TerenzioU/program/shubhangi/center1/im_'+str(j)+'_data.csv', sep=',', index=True, header=True)

    
    b1 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/data/im_'+str(j)+'_data.csv')
    b2 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/data1/im_'+str(j)+'_data.csv')
    
    def Union(lst1, lst2):
        final_list = lst1 + lst2
        return final_list
        
    def PointsInCircum(ep,i,n):
        centerx = [(b1.cent_X.values[i]+math.cos(pi/n)*ep,b1.cent_Y.values[i]+math.sin(pi/n)*ep)]                     # +math.cos(pi/n)*ep
        return centerx
    
    def circle(x1, y1, x2, y2, r1, r2):
  
        distSq = (((x1 - x2)* (x1 - x2))+ ((y1 - y2)* (y1 - y2)))**(.5)
        if (distSq<=r2):
            #print('lies inside =', distSq)
            return r2

        
    ap1=[]; bp1=[];ap2=[]; bp2=[];out=[]; cent =[];circp=[]
    eps = b1.minr.values; eps2 = b2.minr2.values
    k1=0;ii = []; d3=[]; d4=[];d5=[];d6=[]; indx=[];indx1=[];ii1=[];k=0;j1=[];b10=[];b11=[];ex=[];ey=[]
# circle filter se subtract kiya 1000 wale data ko to get center and exact superpixel coordinates then from there 10 wale k x and y coordinates mn dekha ki center h ya nahi 
    for i, jj in zip(dfz11.itertuples(index = False, name ='Pandas'), range(len(dfz11))):
        #print('fiest time j_jj=', str(j)+'_'+str(jj))
        #print('i=', i)
        a3 = dfz12.subtract(i, axis=1)
        ax  = pd.DataFrame(a3).to_csv('/flash/TerenzioU/program/shubhangi/shubh/comparison/t_'+str(j)+'_'+str(jj)+'.csv', sep=',', index = True, header=True)
        a1 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/comparison/t_'+str(j)+'_'+str(jj)+'.csv')        

        for index1, i2 in zip(range(len(a1)), a1.iterrows()):
            k1 =  k1+1
         
            if((0<=abs(a1.circ_x[index1])<=2.5 and 0<=abs(a1.circ_y[index1])<=2.5)  and (abs(a1.Radius[index1])<=10 and abs(a1.Radius2[index1])<=10)  ): 
                print('r1=',  abs(a1.Radius[index1]), 'r2=',  abs(a1.Radius2[index1]))
                
          
                if os.path.isfile('/flash/TerenzioU/program/shubhangi/shubh/im1/mask_'+str(j)+'_'+str(index1)+'.png'):
                    #print('^^^^^^')
                    print('t_'+str(j)+'_'+str(jj)+'')
                    print('index1=', index1)
                    read_image = Image.open('/flash/TerenzioU/program/shubhangi/shubh/im1/mask_'+str(j)+'_'+str(index1)+'.png')
                    read_image.save('/flash/TerenzioU/program/shubhangi/shubh/read1/mask_'+str(j)+'_'+str(index1)+'.png')
              
                k = k+1
                indx.append(index1)
                #print('index=', len(indx))
                ii.append(i2)
                b3 = b2.X[index1]; c3 = b2.Y[index1]
                b6 = b2.cent_X[index1]; c6 = b2.cent_Y[index1]
                b4 = b2['sp_ID']
                #print('b=', len(indx))
                b5 = b1.cent_X.values; c5 = b1.cent_Y.values
                bc5 = b1.CX.values; cb5 = b1.CY.values
                #print('pair = ',(b6,c6))
                element =  [l for l in range(len(b5)) if str(b5[l]) in b3]
                element2 = [l for l in range(len(b5)) if str(c5[l]) in c3] 
                union_list = np.unique(Union(element,element2))
                #eps1 = eps[index1]
                for po in union_list:
                   
                    #ax4 = b1.LX[po]; ay4 = b1.LY[po]
                    ax4 = sp_xl[po]
                 #   eps1 = eps[po]
                  #  cir = PointsInCircum(eps1,po,4)
                    #print('cir=', cir)
               #     poo = 0
               #     for pai in range(len(cir)):
                    x1, y1 = b5[po], c5[po]                          #cir[pai]
                            #print('x1=', x1, 'y1=', y1)
                    
                    x2 ,y2 = b6, c6
                    x3, y3 = bc5[po],cb5[po]
                    #print('x2=', x2, 'y2=', y2)
                    r1 ,r2 = eps[po],eps2[index1]
                    #print('r1=', r1); print('r2=', r2)
                    output = circle(x1, y1, x2, y2, r1, r2)
                    if output == r2:
                    
                        ex.append(ax4)              #;ey.append(ay4)
                        ap1.append(x3)
                        bp1.append(y3)
                        ap2.append(x1)
                        bp2.append(y1)
                        out.append(po)
                        #poo +=1
                        if os.path.isfile('/flash/TerenzioU/program/shubhangi/shubh/im/mask_'+str(j)+'_'+str(po)+'.png'):
                           
                            out.append(po)
                            read_image2 = Image.open('/flash/TerenzioU/program/shubhangi/shubh/im/mask_'+str(j)+'_'+str(po)+'.png')
                            read_image2.save('/flash/TerenzioU/program/shubhangi/shubh/read/mask_'+str(j)+'_'+str(po)+'.png')
                            
                out1 = pd.DataFrame(ex).to_csv('/flash/TerenzioU/program/shubhangi/shubh/ex/mask_'+str(j)+'.csv')
                out2 = pd.DataFrame(out).to_csv('/flash/TerenzioU/program/shubhangi/shubh/out/mask_'+str(j)+'.csv')

        ad1 = pd.DataFrame(ap1).to_csv('/flash/TerenzioU/program/shubhangi/shubh/ap/ap_'+str(j)+'.csv')
        bd1 = pd.DataFrame(bp1).to_csv('/flash/TerenzioU/program/shubhangi/shubh/bp/bp_'+str(j)+'.csv')
        
    
    df16 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/ap/ap_'+str(j)+'.csv')
    df17 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/bp/bp_'+str(j)+'.csv')
    df18 = df16['0'].values
    df19 = df17['0'].values
    
    dg = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/data/im_'+str(j)+'_data.csv')
    del dg['X']
    del dg['Y']
    #df = df.drop(['13'], axis =1).values
    scaler = StandardScaler()
    
    min_max_scaler = MinMaxScaler()
    dg1  = min_max_scaler.fit_transform(dg)
    
    kbins = KBinsDiscretizer(n_bins=2, encode='onehot', strategy='uniform')
    dg2 = kbins.fit_transform(dg1)
    
    de1 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/data/im_'+str(j)+'_data.csv')
    de1['Label'] = ''
    de2 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/out/mask_'+str(j)+'.csv')
    de3 = de2['0'].values
    de4 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/en/sp_'+str(j)+'_e.csv')
    de5 = de4['sp_ID'].values
    de4['Label'] = ''
    de6 = de4['Label'].values
    
   
    for b in range(len(de4)):
        i = de5[b]
        for a in range(len(de2)):
            p = de3[a]
            #print('p=', p, 'i=,', i)
            if p==int(i):
                de6[b] = 1
                #print('p=', p, 'i=,', i)
                if os.path.isfile('/flash/TerenzioU/program/shubhangi/shubh/im/mask_'+str(j)+'_'+str(p)+'.png'):
                    #print('p=', p, 'i=,', i)
                    #print('************yes***************')
                    blob = Image.open('/flash/TerenzioU/program/shubhangi/shubh/im/mask_'+str(j)+'_'+str(p)+'.png')
                    blob.save('/flash/TerenzioU/program/shubhangi/shubh/train/Positive/mask_'+str(j)+'_'+str(p)+'.png')
                #else:
                 #   break
        if de6[b] =='':
            #print('i=', i)
            if os.path.isfile('/flash/TerenzioU/program/shubhangi/shubh/im/mask_'+str(j)+'_'+str(int(i))+'.png'):
            #print('$$$$$$$$$$$$$$')
                seg = Image.open('/flash/TerenzioU/program/shubhangi/shubh/im/mask_'+str(j)+'_'+str(int(i))+'.png')
                #print('i=', i)
                seg.save('/flash/TerenzioU/program/shubhangi/shubh/train/Positive/mask_'+str(j)+'_'+str(int(i))+'.png')
                de6[b] = 2
            #else:
             #   break
    de12 = pd.DataFrame(de6).to_csv('/flash/TerenzioU/program/shubhangi/shubh/data2/im_'+str(j)+'.csv', sep=',', index=True, header=True)
    de7 = de1['sp_ID'].values
    de9 = de1['Label'].values
    for a in range(len(de7)):
        for b in range(len(de5)):
            if int(de5[b])==de7[a]:
                #print('222222')
                de9[a] = de6[b]
            if de9[a]=='':
                #print('333333333')
                if os.path.isfile('/flash/TerenzioU/program/shubhangi/shubh/m/mask_'+str(j)+'_'+str(a)+'.png'):
                 #   print('11111111')
                    bg = Image.open('/flash/TerenzioU/program/shubhangi/shubh/m/mask_'+str(j)+'_'+str(a)+'.png')
                    bg.save('/flash/TerenzioU/program/shubhangi/shubh/train/Negative/mask_'+str(j)+'_'+str(a)+'.png')
                de9[a] = 0
               
               
    de10 = pd.DataFrame(de9).to_csv('/flash/TerenzioU/program/shubhangi/shubh/data3/im_'+str(j)+'_data.csv', sep=',', index=True, header=True)
    de8 = pd.DataFrame(de1).to_csv('/flash/TerenzioU/program/shubhangi/shubh/data4/im_'+str(j)+'_data.csv', sep=',', index=False, header=True)
    
    dg3 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/data4/im_'+str(j)+'_data.csv')
    dg5 = pd.DataFrame(dg2).to_csv('/flash/TerenzioU/program/shubhangi/shubh/data5/im_'+str(j)+'_data.csv', sep=',', index=True, header=True)
    dg6 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/data5/im_'+str(j)+'_data.csv',)
    
    
    da = [(x,y) for x,y in zip(df18,df19)]
    #print('data=',data)

    
    
    da = [(x,y) for x,y in zip(df18,df19)]
    #print('data=',data)
    
    Graph = namedtuple("Graph", ["nodes", "edges", "is_directed"])
    n = []
    #nodes = df3.astype(float)
    node = da
    
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(node)
    distances, indices = nbrs.kneighbors(node)
    wi=[]
    #print('nbrs=',nbrs)
    #print('distances=',distances)
    for i in distances:
        wi.append(int(i[1]))
    print('weight=',wi )
    list1=[];list2=[]
    for i in indices:
        
        list1.append(list(i)[0])
        list2.append(list(i)[1])
    union_list = np.unique(Union(list1,list2))
    nodes = list(union_list)
    edges = [(list1[a],list2[a]) for a in range(len(list1))]
    
    g = Graph(nodes,edges,is_directed=True)
   # print('edges=', G.edges)
    #for g in G.nodes:
     #   print('g=', g)
    def adjacency_dict(graph):
        adj = {node: [] for node in graph.nodes}
        for edge in graph.edges:
            node1, node2 = edge[0], edge[1]
            #if abs(int(node2)-int(node1))<=15:
            adj[node1].append(node2)
                 #print('adjacent nodes are', adj[node1])
                 
            if not graph.is_directed:
                adj[node2].append(node1)
        return adj
    #print('dictionary =', adjacency_dict(G))
     
    def adjacency_matrix(graph):
        k = 0        
        adj = [[0 for node in graph.nodes] for node in graph.nodes]
        #print('shape of adj=', np.shape(adj))
        for edge in graph.edges:
            node1, node2 = edge[0], edge[1]
            #node1  = int(node1)
            #node2 = int(node2)
            adj[node1][node2] += 1
            if not graph.is_directed:
                adj[node2][node1] += 1
        return adj
    #print('Matrix =', adjacency_matrix(G))
    
    def show(graph, output_filename, notebook=False):
        G = Network(directed=graph.is_directed, notebook=notebook)
        G.add_nodes(graph.nodes)
        G.add_edges(graph.edges)
        G.show(output_filename)
        return G
   # show(g,'/flash/TerenzioU/program/shubhangi/shubh/html/basic_'+str(j)+'.html')
    
    
    def _degrees(graph):
        """Return a dictionary of degrees for each node in the graph"""
        adj_list = adjacency_dict(graph)
        degrees = {node: len(neighbors) for node, neighbors in adj_list.items()}
        return degrees
    print('_degrees=', _degrees(g))
    
   
        

                    
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))