import os
import json
from keras.utils.np_utils import to_categorical
from colorama import Fore, Back, Style
from colorama import init
from dotenv import load_dotenv, find_dotenv
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import Dense, Flatten
from keras.layers import Input
from keras.applications.inception_v3 import InceptionV3
from scipy.misc import imresize
import collections
from shutil import copyfile
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.regularizers import l2
import keras.backend as K
import image_gen_extended as T
import multiprocessing as mp
import pathlib

load_dotenv(find_dotenv())
init()

EPOCHS = float(os.environ.get("epochs"))
BATCH_SIZE = int(os.environ.get("batch_size"))
TRAIN_IMAGES_JSON = os.environ.get("train_images")
TEST_IMAGES_JSON = os.environ.get("test_images")
DROPOUT = float(os.environ.get("dropout"))
SUPPORTED_FOOD = os.environ.get("supported_food")
SHAPE = (224, 224, 3)
MODEL_LOCATION = os.environ.get('trained_model_url')
IMAGE_LOCATION = os.environ.get('image_url')

def exportImages(json_url):
	data = json.load(open(json_url))
	prepend_url = 'ordered_images/test/'
	if "test" not in json_url: 
		prepend_url = 'ordered_images/train/'

	pathlib.Path(prepend_url).mkdir(parents=True, exist_ok=True) 

	for food_obj in data.values():
		for food_url in food_obj:
			food_class = food_url.split('/')[0]
			if food_class not in classes:
				break
			pathlib.Path(prepend_url +  food_class).mkdir(parents=True, exist_ok=True) 
			print('copying file to ' + prepend_url + food_url + '.jpg')
			copyfile('images/' + food_url + '.jpg', prepend_url + food_url + '.jpg')


def load_image(img_path, dimensions, rescale=1. / 255):
    img = image.load_img(img_path, target_size=dimensions)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x *= rescale # rescale the same as when trained

    return x


def create_model(num_classes, dropout, shape):
	base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(
            shape=shape))

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(dropout)(x)
	predictions = Dense(num_classes, activation='softmax')(x)

	model_final = Model(inputs=base_model.input, outputs=predictions)

	return model_final


def load_model(weights_path, shape):
   model_final = create_model(4, 0, shape)
   model_final.load_weights(weights_path)

   return model_final

def train_model(model_final, train_generator, validation_generator, callbacks):
	model_final.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

	model_final.fit_generator(train_generator, validation_data=validation_generator,
                              epochs=EPOCHS, callbacks=callbacks,
                              steps_per_epoch=train_generator.samples,
                              validation_steps=validation_generator.samples)



if __name__ == '__main__': 

	classes = SUPPORTED_FOOD.split(',')
	class_to_num = dict(zip(classes, range(len(classes))))
	num_to_class = dict(zip(range(len(classes)), classes))
	class_to_num = {v: k for k, v in num_to_class.items()}
	sorted_class_to_num = collections.OrderedDict(sorted(class_to_num.items()))

	while True:
		print("\n1. Export images to train/test split")
		print("2. Create model")
		print("3. Train model")
		print("4. Load trained model")
		print("5. Load food & predict class and calories")
		try:
			selected = int(input("Enter number from menu: "))
			if(selected == 1):
				exportImages(TEST_IMAGES_JSON)
				exportImages(TRAIN_IMAGES_JSON)

			elif(selected == 2):	
				train_datagen = ImageDataGenerator(
			        rotation_range=40,
			        width_shift_range=0.2,
			        height_shift_range=0.2,
			        rescale=1. / 255,
			        shear_range=0.2,
			        zoom_range=0.2,
			        horizontal_flip=True,
			        fill_mode='nearest')

				test_datagen = ImageDataGenerator(rescale=1. / 255)

				train_generator = train_datagen.flow_from_directory(
			        'ordered_images/train',  # this is the target directory
			        target_size=SHAPE[:2],
			        batch_size=BATCH_SIZE)

				validation_generator = test_datagen.flow_from_directory(
			        'ordered_images/test', # this is the target directory
			        target_size=SHAPE[:2],
			        batch_size=BATCH_SIZE)


				model = create_model(train_generator.num_classes, DROPOUT, SHAPE)

			elif(selected == 3):
				callbacks = []
				callbacks.append(ModelCheckpoint(filepath='saved_models/model4.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True))
				train_model(model, train_generator, validation_generator, callbacks)

			elif(selected == 4):
				trained_model = load_model(MODEL_LOCATION, shape)

			elif(selected == 5):
				image = load_image(IMAGE_LOCATION, shape[:2])
				preds = trained_model.predict(image)
				print('\nThe image is: ' + num_classes[np.argmax(preds)])
		
			else:
				print('\nYou need to enter valid number (from 1 to 5)')


		except ValueError:
			print("\nThat's not even a number.. Try again")	


