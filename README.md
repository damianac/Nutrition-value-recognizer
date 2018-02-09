import os
import json
import Image
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
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

load_dotenv(find_dotenv())
init()
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
EPOCHS = os.environ.get("epochs")
BATCH_SIZE = os.environ.get("batch_size")
TRAIN_IMAGES_LOCATION = os.environ.get("train_images_url")
TEST_IMAGES_LOCATION = os.environ.get("test_images_url")
DROPOUT = os.environ.get("dropout")
shape = (224, 224, 3)


def load_image(img_path, dimensions, rescale=1. / 255):
    img = image.load_img(img_path, target_size=dimensions)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x *= rescale # rescale the same as when trained

    return x


def load_images(location):
	
	images = []
	classes = []
	data = json.load(open(location))

	i = 0
	for food_obj in data.values():
		i = i + 1
		if i > 2:
			break
		for food_url in food_obj:
			print('Loaded ' + 'images/' + food_url + '.jpg')
			image = cv2.imread('images/' + food_url + '.jpg')
            image = imresize(arr=image, size=shape)
			if image is not None:
                imageArray = np.asarray(image)
                images.append(imageArray)
                Y.append(wbcType)

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y


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


def load_model(model_final, weights_path, shape):
   model_final = create_model(101, 0, shape)
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


def sequence_image_generator(x, y, batch_size, generator, seq_len=4):
    new_y = numpy.repeat(y, seq_len, axis = 0)
    helper_flow = generator.flow(x.reshape((x.shape[0] * seq_len,
                                            x.shape[2],
                                            x.shape[3],
                                            x.shape[4])),
                                 new_y,
                                 batch_size=seq_len * batch_size)
    for x_temp, y_temp in helper_flow:
        yield x_temp.reshape((x_temp.shape[0] / seq_len, 
                              seq_len, 
                              x.shape[2],
                              x.shape[3],
                              x.shape[4])), y_temp[::seq_len,:]

print('Welcome to the ' + Back.GREEN + 'NutriValuer' + Style.RESET_ALL + ' - a machine-learning program for food recognition.')


class_to_num = {}
num_to_class = {}
with open('meta/classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]
    class_to_num = dict(zip(classes, range(len(classes))))
    num_to_class = dict(zip(range(len(classes)), classes))
    class_to_num = {v: k for k, v in num_to_class.items()}

test_data, test_classes = load_images(TEST_IMAGES_LOCATION)
train_data, train_classes = load_images(TRAIN_IMAGES_LOCATION)

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_classes = to_categorical(train_classes)
train_datagen.fit(train_data)
#train_gen = train_datagen.flow(train_data, train_classes)
#train_gen_sequence = sequence_image_generator(train_data, train_classes, train_datagen, 4)
#print(train_gen)
#test_datagen = ImageDataGenerator(rescale=1. / 255)
#test_gen = sequence_image_generator(test_data, test_classes, test_datagen, 4)

#callbacks = []
#callbacks.append(ModelCheckpoint(filepath='saved_models/food-101-epoch-{epoch:02d}.hdf5',verbose=1, save_best_only=True))

#model = create_model(2, 0.0, shape)
#train_model(model, train_gen, test_gen, callbacks)
