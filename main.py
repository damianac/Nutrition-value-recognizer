import os
import json
import cv2
from keras.utils.np_utils import to_categorical
from colorama import Fore, Back, Style
from colorama import init
from dotenv import load_dotenv, find_dotenv
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import Dense
from keras.layers import Input
from keras.applications.inception_v3 import InceptionV3
from scipy.misc import imresize
import collections
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.regularizers import l2
import keras.backend as K
import image_gen_extended as T
import multiprocessing as mp



load_dotenv(find_dotenv())
init()
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
EPOCHS = float(os.environ.get("epochs"))
BATCH_SIZE = int(os.environ.get("batch_size"))
TRAIN_IMAGES_LOCATION = os.environ.get("train_images_url")
TEST_IMAGES_LOCATION = os.environ.get("test_images_url")
DROPOUT = float(os.environ.get("dropout"))
shape = (224, 224, 3)


def load_image(img_path, dimensions, rescale=1. / 255):
    img = image.load_img(img_path, target_size=dimensions)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x *= rescale # rescale the same as when trained

    return x


# Load dataset images and resize to meet minimum width and height pixel size
def load_images(location, min_side=224):
	data = json.load(open(location))
	all_imgs = []
	all_classes = []
	resize_count = 0
	invalid_count = 0
	i = 0
	j = 0
	for food_obj in data.values():
		j = j + 1
		if j > 30:
			break
		i = 0
		for food_url in food_obj:
			# if "pizza" not in food_url:
 		# 		break
			i = i + 1
			if i > 1:
				break
			img_arr = cv2.imread('images/' + food_url + '.jpg')
			img_arr_rs = img_arr
			try:
				w, h, _ = img_arr.shape
				if w < min_side:
					wpercent = (min_side/float(w))
					hsize = int((float(h)*float(wpercent)))
                    #print('new dims:', min_side, hsize)
					img_arr_rs = imresize(img_arr, (min_side, hsize))
					resize_count += 1
				elif h < min_side:
					hpercent = (min_side/float(h))
					wsize = int((float(w)*float(hpercent)))
					#print('new dims:', wsize, min_side)
					img_arr_rs = imresize(img_arr, (wsize, min_side))
					resize_count += 1
				food_class = food_url.split('/')[0]
				all_classes.append(class_to_num[food_class])
				all_imgs.append(img_arr_rs)
			except:
				print('Skipping bad image: ', food_url)
				invalid_count += 1
	print(len(all_imgs), 'images loaded')
	print(resize_count, 'images resized')
	print(invalid_count, 'images skipped')
	return np.array(all_imgs), np.array(all_classes)


def create_model(num_classes, dropout, shape):
	base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(
            shape=shape))

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(dropout)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	predictions = Dense(num_classes, activation='softmax')(x)

	model_final = Model(inputs=base_model.input, outputs=predictions)

	return model_final


def load_model(model_final, weights_path, shape):
   model_final = create_model(101, 0, shape)
   model_final.load_weights(weights_path)

   return model_final

def train_model(model_final, train_generator, validation_generator, callbacks, samples, val_samples):
	model_final.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

	model_final.fit_generator(train_generator, validation_data=validation_generator,
                              epochs=EPOCHS, callbacks=callbacks,
                              steps_per_epoch=samples,
                              validation_steps=val_samples)


if __name__ == '__main__': 
	num_processes = 3
	pool = mp.Pool(processes=num_processes)  
	class_to_num = {}
	num_to_class = {}
	with open('meta/classes.txt', 'r') as txt:
	    classes = [l.strip() for l in txt.readlines()]
	    class_to_num = dict(zip(classes, range(len(classes))))
	    num_to_class = dict(zip(range(len(classes)), classes))
	    class_to_num = {v: k for k, v in num_to_class.items()}
	sorted_class_to_num = collections.OrderedDict(sorted(class_to_num.items()))

	#X_test, y_test = load_images2(TRAIN_IMAGES_LOCATION, min_side=299)

	test_data, test_classes = load_images(TEST_IMAGES_LOCATION)
	train_data, train_classes = load_images(TRAIN_IMAGES_LOCATION)


	train_classes_cat = to_categorical(train_classes, 101)
	test_classes_cat = to_categorical(test_classes, 101)


	train_datagen = T.ImageDataGenerator(
	    featurewise_center=False,  # set input mean to 0 over the dataset
	    samplewise_center=False,  # set each sample mean to 0
	    featurewise_std_normalization=False,  # divide inputs by std of the dataset
	    samplewise_std_normalization=False,  # divide each input by its std
	    zca_whitening=False,  # apply ZCA whitening
	    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
	    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
	    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
	    horizontal_flip=True,  # randomly flip images
	    vertical_flip=False, # randomly flip images
	    zoom_range=[.8, 1],
	    channel_shift_range=30,
	    fill_mode='reflect')
	train_datagen.config['random_crop_size'] = (224, 224)
	train_datagen.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
	train_generator = train_datagen.flow(train_data, train_classes_cat, batch_size=BATCH_SIZE, seed=11, pool=pool)

	# #datagen.fit(train_data)
	# train_gen = datagen.flow(train_data, train_classes_cat, batch_size=BATCH_SIZE)
	test_datagen = T.ImageDataGenerator()
	test_datagen.config['random_crop_size'] = (224, 224)
	test_datagen.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
	test_generator = test_datagen.flow(test_classes, test_classes_cat, batch_size=BATCH_SIZE, seed=11,  pool=pool)
	 
	# test_gen = datagen.flow(test_data, test_classes_cat, batch_size=BATCH_SIZE)
	K.clear_session()
	callbacks = []
	callbacks.append(ModelCheckpoint(filepath='model4.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True))

	model = create_model(101, DROPOUT, shape)
	train_model(model, train_generator, test_generator, callbacks, train_data.shape[0] - 1, test_data.shape[0] - 1)
