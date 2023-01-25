# Обработчик фото паспортов. Распознает некоторые области поворачивает и обрезает.

использует в работе 2 варианта алгоритма:

          1: Yolovo5 , openCV, 
          2: 1й вариант + OpenCV распознание лиц и remdg обрезка фона
          
 pastDetect.py - главный файл:
 
          1: !python pastDetect.py -f data/images/2015.jpg     обрабатывает один файл 
          2: !python pastDetect.py                             обрабатывает  папку 'data/images/'  



import cv2 

import os

import numpy as np

from importlib import reload 

from scipy.spatial import distance

import math

import time

import shutil

from rembg import remove

import matplotlib.pyplot as plt

import remdCvFind  # !!!!! Сопутствующий файл нужно взять из проэкта и загрузить в папку с исполняемым файлом

import detect      # !!!!! Нужен модифицированный detect.py  добавить в папку /yolovo5  иначе не вернет картинку и фитчи. (старый лучше сохранить)


# !cp best.pt /content/yolov5/  #обученная модель для паспортов
# !cp detect.py /content/yolov5/   #  экспортируем detect.py
