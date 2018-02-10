# Nutrition-value-recognizer
A machine-learning program for food recognition

## Motivation
We've found ourselves in a need of a program, that's going to detect nutrition value of food we take picture of, instead of manually adding food name
and checking those data by hand. This could be a must-have application on your phone in a near future, used by all the people that care about health and fitness
on daily basis.

## Few bumps on the road in the process of making
After few days of learning keras/tensorflow libs, we've noticed that we'd need a hell of a computer to get a proper results (accuracy).
That's mainly caused by not utilisating GPU, and to get proper results usage of both CPU and GPU is a must. 

## How-to
In order to successfuly run the project you'd need:
- Python 2.X/3.X version
- Keras
- Numpy
- Tensorflow as a backend
- scipy
- dotenv

## Dataset
As most of food-recognition projects, we are also using [Food-101 Dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/).
It contains 101 diffrent classes, which is fairsome amount for training. There are 1000 images of each food (around 750 for training, and 250 for testing)
which makes it in total of ~101,000 different images.

## Nutrition value database
We've used [MyFitnessPal](http://www.myfitnesspal.com/food/calorie-chart-nutrition-facts) for getting nutrition values for each of 101 diffrent food.
They have been added manually by our team member. The intention was to add this in an embedded database, but we ended up with json, and it'll do the job
since it has only 101 diffrent objects.

## Image processing
Dataset that we used offers train/test images in one folder, and two json files, train.json - which contains paths to train images, and test.json containing
paths to test images. We've parsed both json files, and extracted train and test images to train and test folder respectfully:

```
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

```

As for image processing, we are using build-in Keras generator [ImageDataGenerator](https://keras.io/preprocessing/image/)

## Creating model
So far, the best result we got is with [ResNet50](https://keras.io/applications/#resnet50), and that's what we used in code

## Training model
The CNN we used is GlobalAveragePooling by Keras

## Achieved accuracy
We have tested training model with 4 different food classes and the result we got is ~60% accuracy with only one epoche.