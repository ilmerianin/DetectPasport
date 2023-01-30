#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:51:48 2023

@author: wasilii

Набор методов для обработки фото паспортов.
     1. Вырезает фон с помощью ИИ rembg
     2. Обесцвечивает  для поиска границ openCV
     3. Ищет зону с границами, ост. обрезает numpy
     4. Распознает лицо openCV
     5. По расположению найденного лица через коэф. определяет правильность
     если все ок. возвращпет обрезанное изображение
      иначе np.array[16,0]

"""

import cv2 as cv
import shutil
import sys
import os
import time 
import numpy as np
#import matplotlib.pyplot as plt
from rembg import remove # Нейросеть  вырезает фон


# было в составе самостоятельного файла
cutFoto = 'cutFotoAfter_Mod_Pas' # название папки получателя не
pathFoto = 'data/images'   #'data/фото'  # папка источник файлов относительный путь
paperNew = 'newfotoCV'     # папка где будут новые файлы, вложенна  в pathFoto т.е новые файлы будут в папке 'data/фото/newfoto'

verbose = False # Показывать всплывающие окна для отладки 

def Find_cascade_fich(face_cascade, image):
    ''' ищет лица на фото  работает 13.05.22 cv.CascadeClassifier очень посредственно
    face_cascad 
    image frame 
    return:
        type(faces): <class 'numpy.ndarray'> (x, 4)
    время работы  wait time: 0.15 сек
     https://tproger.ru/translations/opencv-python-guide/'''

    assert not face_cascade.empty(), 'cv.CascadeClassifier( не нашёл файл haarcascade_frontalface_default.xml) '
    

    try:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # сделать серым
    except:
        gray = image

    faces = face_cascade.detectMultiScale(   #общая функция для распознавания как лиц, так и объектов. Чтобы функция искала именно лица, мы передаём ей соответствующий каскад.
        gray,              # Обрабатываемое изображение в градации серого.
        scaleFactor= 1.1,  # Параметр scaleFactor. Некоторые лица могут быть больше других, поскольку находятся ближе, чем остальные. Этот параметр компенсирует перспективу.
        minNeighbors= 5,   # Алгоритм распознавания использует скользящее окно во время распознавания объектов. Параметр minNeighbors определяет количество объектов вокруг лица. То есть чем больше значение этого параметра, тем больше аналогичных объектов необходимо алгоритму, чтобы он определил текущий объект, как лицо. Слишком маленькое значение увеличит количество ложных срабатываний, а слишком большое сделает алгоритм более требовательным.
        minSize=(10, 10)   # непосредственно размер этих областей
    )

    return faces

def DravRectangleImage(image, rectangle_NP):
    ''' рисует картинку и квадраты фич работает 13.05.22
        image: cv.imread
        rectangle_NP :<class 'numpy.ndarray'> (x, 4) 
    return 
        True - 
        Fault - не найдены фичи
    '''
    faces_detected = "Fich find: " + format(len(rectangle_NP))
    if len(rectangle_NP) == 0:
        print('не найдены фичи')
        return 0
    #print(type(rectangle_NP), rectangle_NP)
    
    image = np.ascontiguousarray(image, dtype=np.uint8)
    # Рисуем квадраты вокруг лиц
    #print(rectangle_NP)
    for (x, y, w, h) in rectangle_NP:
        
        #print( x, y, w, h)
        
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 20) # отрисовка квадратов

    return image

def findfith(image):
    ''' Поднотовка класификатора OpenCV '''
                    # Собственно этой командой мы загружаем уже обученные классификаторы cv.data.haarcascades+'haarcascade_frontalface_default.xml'
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    StartTime = time.time()
    faces = Find_cascade_fich(face_cascade,image) #  находит работает #поиск фитч "лиц"
    #print('Время распознания:', str(time.time()-StartTime), 'type:', type(faces), faces)
   
    return   faces   


def viewImage(image,waiK= 0, nameWindow = 'message windows', verbose = verbose):
    ''' Вывод в отдельное окошко 
    image - картинка numpy подобный массив
    waiK - int время ожидания окна если 0- будет ждать нажатия клавиши
    nameWindow - название окна лучше по английски иначе проблемы с размером
    verbose - показывать или нет True/False
    '''
    if verbose:
        cv.namedWindow(nameWindow, cv.WINDOW_NORMAL)
        cv.imshow(nameWindow, image)
        cv.waitKey(waiK)
        cv.destroyAllWindows()
    else:
        pass
    return


def windowClear(nparr):
    ''' проходит окошком и чистит от мусорра
     ширина окошка порога заданны в переменных 
     возможно лишняя опция'''
    xW  = 10
    yW = 10
    porog = 200
    for x in range(0, nparr.shape[0]-xW, 5):
        for y in range(0, nparr.shape[1]-yW, 5):
            if nparr[x: x + xW, y: y+yW].sum() < porog:
                nparr[x: x + xW, y: y+yW] = 0
    #viewImage(nparr, 'Подрезка окном')
    return nparr

def findContur(image):
    ''' превращение картинки в контуры удоьно убирает мусор '''
  # Convert to graycsale
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3,3), 0) 
     
    # Sobel Edge Detection
    sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    
    return edges     

def findBoard(npImage):
    ''' нахождение рамки путем поиска граничных не нулевых значений'''
                          # (2800, 1700, 4)  >> (2800, 1700)
    if len(npImage.shape)>2:
        #print('cut shape',npImage.shape) 
        npImage = np.sum(npImage, axis = 2) # суммирование цветных измерений в одно измерения
    x = npImage.shape[1]
    y = npImage.shape[0]
    xa = 0
    ya = 0
  
    npImage = windowClear(npImage) #проходит окошком и чистит от мусорра
    
    for i in range(npImage.shape[0]):
        ix = np.nonzero(npImage[i])
        
        if  ix[0].any() > 0:
            if x > ix[0][0]:  # левый край
                x = ix[0][0]
            if xa < ix[0][-1]:  # правый край
                xa = ix[0][-1]
            if y > i: # верхний край
                y = i
            if ya < i: # нижний край
                ya = i
            
            x1 = ix[0]
    #print('найденна рамка\n x,y' , x, y, '\n x1,y1=', xa,ya)
   
    return x,y,xa,ya

def choiseFace(rectFace):
    ''' выбор большей высоте картинки как лица 
    отсеивает заведомо мусорные загогулины, но может и фотку если есть большие фитчи'''
    sq0 = 0
    print( 'choiseFace len()', len(rectFace))
    
    #print(type(rectFace))
    print(rectFace)
    
    ind = rectFace[:,2].argsort() # Сортируем по высоте
    print(type(ind), ind)

    if len(ind) > 4:
        print(rectFace[ind[-4:], :] )
        return rectFace[ind[-4:], :]  # возвращаем 3 самых больших бокса
 
    elif len(ind) > 0:
         return rectFace[ind, :]  
    else:
        faceRect = np.array([[0,0,0,0]])
    return faceRect

def choiseFaceSq(rectFace):
    ''' выбор большей по площади картинки как лица 
    отсеивает заведомо мусорные загогулины, но может и фотку если есть большие фитчи'''
    sq0 = 0
    print( 'choiseFace len()', len(rectFace))
    #ind = rectFace[:,2].argsort()
    #print(type(rectFace))
    #print(rectFace)
    if len(rectFace)  > 0:
        if rectFace[0].shape[0]>1:
            for (x, y, w, h) in rectFace:
                sq = w*h
                if sq > sq0:
                    faceRect = np.array([[x,y,w,h]])
                    sq0 = sq
        else:
            faceRect = rectFace
    else:
        faceRect = np.array([[0,0,0,0]])
    return faceRect

def rotationNP(img):
    ''' вращение картинки методом numpy '''
    img = np.rot90(img, k=-1)

    return img

def rotation(img):
    ''' Вращение картинки методом openCV
    но много требует плясок и режет края numpy привычнее'''
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv.getRotationMatrix2D(center, -90, 1)
    rotated = cv.warpAffine(img, rotation_matrix, (w, h))
    return rotated

def giveListFiles(patn = pathFoto, pathGood = paperNew, pathBad =cutFoto ):
    ''' выдает списки обработанных файлов из папок
    goodfiles, badFiles '''
    try:
        goodfiles = os.listdir(os.path.join(patn, pathGood)) # получить список сделанных файлов
    except FileNotFoundError:
        goodfiles = []

    try:
        badFiles = os.listdir(os.path.join(patn, pathBad)) # получить список сделанных файло
    except FileNotFoundError:
        badFiles =[]
    return goodfiles, badFiles

def cutRemd(image):
    ''' 
     1. Вырезает фон с помощью ИИ rembg
     2. Обесцвечивает  для поиска границ openCV
     3. Ищет зону с границами, ост. обрезает numpy
     4. Распознает лицо openCV
     5. По расположению найденного лица через коэф. определяет правильность
     если все ок. возвращпет обрезанное изображение
      иначе np.array[16,0]
      '''
                     # fotogen = getFoto(path='data/фото') 
    countF =0
    flBadDetect = True   # флаг отсутствия распознавания
                # если файлы не обрабатывались ещё то                            
    rectFace = findfith(image)        #поиск лиц методом openCV
 
    viewImage(DravRectangleImage(image.copy(), rectFace), waiK=0, nameWindow = 'before all Koef' )
    
    if len(rectFace)>=1:  # если лиц много
    
        image_out = remove(image)   #удаление фона нейросетью   
        image_out = findContur(image_out) #мод картинки в контуры
        x,y,xa,ya = findBoard(image_out)  # поиск границ паспорта
        #print('повторный выбор ', len(rectFace), type(rectFace) )
        rectFace = choiseFace(rectFace)   # выбор лица если много картинок
        rectAll = np.append(rectFace, (x, y, xa-x, ya-y)).reshape(-1,4) # вывести планируемые границы    
        
        
        
        for i in range(rectFace.shape[0]):
            
            rect = np.concatenate((rectFace[i], [x,y,xa- x ,ya -y]), axis=0).reshape(-1,4)
            
            koef = np.divide(rect[1], rect[0])    # коэффициенты расположения фотографии
            koef1 = (xa-x)/(rect[0][0]-x)         # коэффициенты расположения фотографии
            koef2 = (ya-y)/ (y+ya-rect[0][1]-rect[0][3])  # коэффициенты расположения фотографии
           
            #if 4.5 < koef[2] and koef[2]< 7 and 6.36 < koef[3] and koef[3] < 9.25 and 6.8 < koef1 and koef1 <9.95 and 2.1 < koef2 and koef2 < 4.7:
            print(koef[2],koef[3], koef1, koef2 )
            viewImage(DravRectangleImage(image.copy(), rectAll), waiK=0, nameWindow = 'before Koef' )
            
            if 4.3 < koef[2] and koef[2]< 9.3 and 6.36 < koef[3] and koef[3] < 11.3 and 6.2 < koef1 and koef1 <13 and 1.6 < koef2 and koef2 < 5.1:
                    
                    if len(image.shape)==2: #если фотка дез 3го измерения
                        image = np.expand_dims(image, axis=-1) #добавить измерение
                    image = image[y:ya, x:xa, : ]  # срез нужной области картинки
                    viewImage(image,waiK= 0, nameWindow = 'Goood Koef')
                    return image
                    
    return np.array([0,0,0,0]*4)


if __name__ == '__main__':
    npar = cutRemd(True)    

