# Обработчик фото паспортов. Распознает некоторые области поворачивает и обрезает.

использует в работе 2 варианта алгоритма:

          1: Yolovo5 с обученными весами, openCV, 
          2: 1й вариант + OpenCV распознание лиц и remdg обрезка фона

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

import remdCvFind  # !!!!! Сопутствующий файл нужно взять из проэкта и загрузить в папку yolovo

import detect      # !!!!! Нужен модифицированный detect.py  иначе не вернет картинку и фитчи.  взять из проэкта
