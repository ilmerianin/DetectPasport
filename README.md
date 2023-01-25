# Обработчик фото паспортов. Распознает некоторые области поворачивает и обрезает.

использует в работе 2 варианта алгоритма:

          1: Yolovo5 , openCV, 
          2: 1й вариант + OpenCV распознание лиц и remdg обрезка фона
          
 pastDetectYolovo5_v_1_0.py - главный файл



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
