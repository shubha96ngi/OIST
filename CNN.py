import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
from scipy.ndimage import rotate
from PIL import Image
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils.np_utils import normalize, to_categorical
from keras.layers import Dropout
#from keras.optimizers import gradient_descent_v2 
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
#import splitfolders
import os, cv2, random
import pathlib
'''
initial_count = 0
for path in pathlib.Path("/flash/TerenzioU/program/gray2").iterdir():
    if path.is_file():
        initial_count += 1

print(initial_count)
list =[]
'''

for j in range(5):
    ds1 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/en/sp_'+str(j)+'_e.csv')
    ds2 = ds1['sp_ID'].values
    ds3 = pd.read_csv('/flash/TerenzioU/program/shubhangi/shubh/en1/sp_'+str(j)+'_e.csv')
    ds4 = ds3['sp_ID'].values
    '''
    for i in range(len(ds1)):
        ds2[i] = int(ds2[i])
        print('********')
        #print('ds2=', str(int(ds2[i])))
        files = cv2.imread('/flash/TerenzioU/program/shubhangi/shubh/m/mask_'+str(j)+'_'+str(int(ds2[i]))+'.png')
        cv2.imwrite('/flash/TerenzioU/program/shubhangi/shubh/im/mask_'+str(j)+'_'+str(int(ds2[i]))+'.png', files)
    '''
    for i1 in range(len(ds3)):
        #print('ds2=', ds4[i1])
        ds4[i1] = int(ds4[i1])
        print(int(ds4[i1]))
        files1 = cv2.imread('/flash/TerenzioU/program/shubhangi/shubh/m1/mask_'+str(j)+'_'+str(int(ds4[i1]))+'.png')
        cv2.imwrite('/flash/TerenzioU/program/shubhangi/shubh/im1/mask_'+str(j)+'_'+str(int(ds4[i1]))+'.png', files1)

'''
images_to_generate=1000
seed_for_random = 42

#Define functions for each operation
#Define seed for random to keep the transformation same for image and mask

# Make sure the order of the spline interpolation is 0, default is 3. 
#With interpolation, the pixel values get messed up.
def rotation(image, seed):
    random.seed(seed)
    angle= random.randint(-180,180)
    r_img = rotate(image, angle, mode='constant', reshape=False, order=0)
    return r_img

def h_flip(image, seed):
    hflipped_img= np.fliplr(image)
    return  hflipped_img

def v_flip(image, seed):
    vflipped_img= np.flipud(image)
    return vflipped_img

def v_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    vtranslated_img = np.roll(image, n_pixels, axis=0)
    return vtranslated_img

def h_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    htranslated_img = np.roll(image, n_pixels, axis=1)
    return htranslated_img



transformations = {'rotate': rotation,
                'horizontal flip': h_flip, 
                'vertical flip': v_flip,
                'vertical shift': v_transl,
                'horizontal shift': h_transl
}                #use dictionary to store names of functions 

images_path="/flash/TerenzioU/program/images/" #path to original images
img_augmented_path="/flash/TerenzioU/program/augmented/" # path to store aumented images
images=[] # to store paths of images from folder

for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array 
    #print('im=', im)    
    images.append(os.path.join(images_path,im))

i=1   # variable to iterate till images_to_generate

for i in range(images_to_generate): 
    number = random.randint(0, len(images))  #PIck a number to select an image & mask
    image = images[number]
    #print(image, mask)
    #image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    transformed_image = None
#     print(i)
    n = 0       #variable to iterate till number of transformation to apply
    transformation_count = random.randint(1, len(transformations)) #choose random number of transformation to apply on the image
    
    print(list(transformations))
    print(random.choice(list(transformations)))
    print('transform=', transformation_count)
    
    for n in range(transformation_count):
        key = random.choice(list(transformations)) #randomly choosing method to call
        seed = random.randint(1,100)  #Generate seed to supply transformation functions. 
        transformed_image = transformations[key](original_image, seed)
        n = n + 1
        
    new_image_path= "%s/augmented_image_%s.png" %(img_augmented_path, i)
    io.imsave(new_image_path, transformed_image)
    i =i+1
  
datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')    #Also try nearest, constant, reflect, wrap

image_directory = '/flash/TerenzioU/program/images/'

aug_dataset = []
my_images =  os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if(image_name.endswith(".png")):
        print('i=', i)
        a_image = cv2.imread(os.path.join(image_directory, image_name))      #io.imread(image_directory + image_name)
        #print(np.shape(a_image))
        #a_image = cv2.resize(a_image, (16,16))
        #a_image = cv2.imread(os.path.join(image_directory, image_name)) 
        a_image = Image.fromarray(a_image, 'RGB')
        aug_dataset.append(np.array(a_image))
#Array with shape (256, 256, 3)
#print(np.array(a_image))
#print(np.shape(my_images))
print('&&&&&&&&&', np.size(aug_dataset))
dg = np.array(aug_dataset)

i = 0
for batch in datagen.flow(dg, batch_size=16,  save_to_dir='augmented', save_prefix='aug', save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely  

df = pd.read_csv('/flash/TerenzioU/program/im_sp_data1.csv')
#X = np.array(X_dataset) # independant features
X = df.drop(['Entropy'], axis =1).values
y = df['Entropy'].values					# dependant variable
#print('X =',  X)

min_max_scaler = MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
print("*********************")
print('X_scale=', X_scale)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(9,)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(4, activation='softmax'))

#opt = gradient_descent_v2.SGD(lr=0.001, momentum=0.9) #Use stochastic gradient descent. Also try others. 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_train, y_train))
model.evaluate(X_test, y_test)[1]


df = pd.read_csv('/flash/TerenzioU/program/dta2/im_'+str(j)+'_data.csv')
df1 = pd.read_csv('/flash/TerenzioU/program/dta1/im_'+str(j)+'_data.csv')


def check_core_point(eps,minPts,df,index,index1):
    #get points from given index
    x, y = df1.iloc[index]['X']  ,  df1.iloc[index]['Y']
    
    #check available points within radius
    temp =  (np.abs(df1[x] - df.iloc[index1]['X']) <= eps) or (np.abs(df1[y] - df.iloc[index1]['Y']) <= eps)
    
    #check how many points are present within radius
    if len(temp) >= minPts:
        #return format (dataframe, is_core, is_border, is_noise)
        return (temp.index , True, False, False)
    
    elif (len(temp) < minPts) and len(temp) > 0:
        #return format (dataframe, is_core, is_border, is_noise)
        return (temp.index , False, True, False)
    
    elif len(temp) == 0:
        #return format (dataframe, is_core, is_border, is_noise)
        return (temp.index , False, False, True)

'''
