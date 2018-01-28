import os
from colorama import Fore, Back, Style
from colorama import init
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
init()

EPOCHS = os.environ.get("epochs")
print('Welcome to the ' + Back.GREEN + 'NutriValuer' + Style.RESET_ALL + ' - a machine-learning program for food recognition.')

