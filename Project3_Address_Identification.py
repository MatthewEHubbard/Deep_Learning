# -*- coding: utf-8 -*-


#adds google drive
from google.colab import drive
drive.mount('/content/drive')

# Set the working directory
import os
os.chdir("/content/drive/MyDrive/BZAN 554/SVHN")

"""# **Organize digitStruct.json Data**"""

import json
import pandas as pd

#opens JSON file
f = open("digitStruct.json")
#returns JSON object as a dictionary
json_data = json.load(f)
#flattens the JSON data variables and saves as a dataframe
json_df=pd.json_normalize(data=json_data)

#length of longest digit
longest_digit = max([len(x['boxes']) for x in json_data])

json_df_boxes = pd.DataFrame(json_df["boxes"].to_list())

digitStruct_df = json_df.join(json_df_boxes)

digitStruct_df.head()

#expands the list of information for each digit
for order in range(0,longest_digit,1):
  digitStruct_df[['width.'+str(order),'top.'+str(order),'label.'+str(order),'left.'+str(order),'height.'+str(order)]] = digitStruct_df[order].apply(pd.Series)

#drops unnecessary columns
digitStruct_df = digitStruct_df.drop(columns = [*range(0,longest_digit,1)])
digitStruct_df = digitStruct_df.drop(columns = ['boxes'])

"""# **Number Verticality/Horizontality**"""

#creates list of all the measurements in order
width_list = []
top_list = []
left_list = []
height_list = []

for order in range(0,longest_digit,1):
  width_list.append('width.'+ str(order))
  top_list.append('top.'+ str(order))
  left_list.append('left.'+ str(order))
  height_list.append('height.'+ str(order))

#combines the labels to the final full number
digitStruct_df['width_list'] = digitStruct_df[width_list].astype(str).agg(', '.join, axis=1) 
digitStruct_df['top_list'] = digitStruct_df[top_list].astype(str).agg(', '.join, axis=1) 
digitStruct_df['left_list'] = digitStruct_df[left_list].astype(str).agg(', '.join, axis=1) 
digitStruct_df['height_list'] = digitStruct_df[height_list].astype(str).agg(', '.join, axis=1) 

def metrics_cleaner(string):
  #removes the nans
  list = string.split(",")
  no_nan = [x for x in list if x != ' nan']
  #replaces all the 10s with 0s
  no_ten = map(lambda x: x.replace('10.0', '0'), no_nan)
  #turns the list of string numbers into float numbers
  metric_float = [float(x) for x in no_ten]
  return metric_float

digitStruct_df['width_list'] = digitStruct_df['width_list'].apply(metrics_cleaner)
digitStruct_df['top_list'] = digitStruct_df['top_list'].apply(metrics_cleaner)
digitStruct_df['left_list'] = digitStruct_df['left_list'].apply(metrics_cleaner)
digitStruct_df['height_list'] = digitStruct_df['height_list'].apply(metrics_cleaner)

def range_finder(list):
  range = max(list)-min(list)
  return range
  
digitStruct_df['width_list_range'] = digitStruct_df['width_list'].apply(range_finder)
digitStruct_df['top_list_range'] = digitStruct_df['top_list'].apply(range_finder)
digitStruct_df['left_list_range'] = digitStruct_df['left_list'].apply(range_finder)
digitStruct_df['height_list_range'] = digitStruct_df['height_list'].apply(range_finder)

### GREATEST HEIGHT RANGE
import tensorflow as tf
height_path = digitStruct_df.sort_values(by='height_list_range', 
                            ascending=False).reset_index(drop=True)['filename'][0]
tf.keras.utils.load_img("test/"+ height_path)

### GREATEST WIDTH RANGE
import tensorflow as tf
width_path = digitStruct_df.sort_values(by='width_list_range', 
                            ascending=False).reset_index(drop=True)['filename'][0]
tf.keras.utils.load_img("test/"+ width_path)

### GREATEST TOP RANGE
import tensorflow as tf
top_path = digitStruct_df.sort_values(by='top_list_range', 
                            ascending=False).reset_index(drop=True)['filename'][0]
tf.keras.utils.load_img("test/"+ top_path)

### GREATEST LEFT RANGE
import tensorflow as tf
left_path = digitStruct_df.sort_values(by='left_list_range', 
                            ascending=False).reset_index(drop=True)['filename'][0]
tf.keras.utils.load_img("test/" + left_path)

label_list = []

for order in range(0,longest_digit,1):
  label_list.append('label.'+ str(order))

digitStruct_df['label_list'] = digitStruct_df[label_list].astype(str).agg(', '.join, axis=1) 

def label_cleaner(string):
  #removes the nans
  list = string.split(",")
  #replaces all the 10s with 0s
  no_ten = map(lambda x: x.replace('10.0', '0'), list)
  #replaces all nans with 10
  no_nan = map(lambda x:x.replace(' nan','10.0'),no_ten)
  #turns the list of string numbers into list of integers
  label_int = [round(float(x)) for x in no_nan]
  #turns the list of label integers into array
  #label_array = np.array(label_int).reshape(-1,1)
  return label_int

digitStruct_df['final_label'] = digitStruct_df['label_list'].apply(label_cleaner)

#digitStruct_df = digitStruct_df[['filename','final_label']]

def array_maker(x):
  array_x= np.array(x).reshape(-1,1)
  return array_x
digitStruct_df['array_label']=digitStruct_df['final_label'].apply(array_maker)
digit_list = [[1,2,3,4,5],[6,7,8,9,0],[10,1,2,3,4]]
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder().fit(np.array(digit_list).reshape(-1,1))


def encoding(x):
  encoded_x=encoder.transform(x).toarray()
  return encoded_x

digitStruct_df['encoded_label'] = digitStruct_df['array_label'].apply(encoding)
Y_train = np.stack(digitStruct_df['encoded_label'].values)

Y_train

Y_train[0]

"""**Separate into Train and Test Data**"""

os.listdir("test/")

#lists the file names in the training dataset
train_list = os.listdir("train/")
#subsets the digitStruct into just training data
train_digitStruct = digitStruct_df[digitStruct_df['filename'].isin(train_list)]

#lists the file names in the testing dataset
test_list = os.listdir("test/")
#subsets the digitStruct into just testing data
test_digitStruct = digitStruct_df[digitStruct_df['filename'].isin(test_list)]

"""**asdfadsf**"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

"""**Load in the Data**"""

from keras.preprocessing.image import ImageDataGenerator

# define data generator
train_datagen = ImageDataGenerator(rescale=1./255)

# define dataset
mnist = train_datagen.flow_from_directory(
        'SVHN/train_mini/',
        target_size=(28, 28),
        batch_size=32,
        color_mode="grayscale",
        class_mode='sparse')

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_labels

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential([
  Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D(pool_size=(2, 2)),
  Dropout(0.25),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)