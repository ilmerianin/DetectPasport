#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
вырезанны лишние принты
Created on Wed Jan 18 22:55:59 2023

@author: wasilii

Пример запуска файла из строки !python pastDetect.py -f data/images/2015.jpg
        в папке файла будет создана папка paperNew = 'newYolovo'  или paperRemdg = 'newYolovoRemd'   куда помнщнен результат 

программа берет из  папки <pathFoto> файлы и если файлов с таким именем нет в папках  <paperNew> или <badFoto> запускает в анализ.
т.е если из этих папок удлить файл он будет запущен в анализ по новой. 
# clone the model
# =============================================================================
# !git clone https://github.com/ultralytics/yolov5
# %cd yolov5
# !pip install -r requirements.txt      # перенос файлов
# !cp /content/drive/MyDrive/imagePasp/*  /content/yolov5/data/images 
# !cp /content/drive/MyDrive/best.pt /content/yolov5/  #обученная модель для паспортов
# !cp /content/drive/MyDrive/Yolov5_files/detect.py /content/yolov5   #  экспортируем detect.py
# =============================================================================
# https://github.com/ilmerianin/DetectPasport.git

"""
import cv2 
import os
import sys
import numpy as np
from importlib import reload 
from scipy.spatial import distance
import math
import time
import shutil
from rembg import remove
import matplotlib.pyplot as plt
import remdCvFind  # !!!!! Сопутствующий файл нужно взять из проэкта и загрузить в папку yolovo
import detect  # !!!!!!    нужен модифицированный detect.py  иначе не вернет картинку и фитчи.  взять из проэкта

# ============================= переменные для настройки рабрты кода ============================
verbose = False # если нуны всплывающие окна установить в True
remd= True   #True/False флаг применения Неиросети remd поиск фона работает долго и менее стабильно
 
badFoto = 'badYolovoRemd'       # название папки получателя не распознанных фото вложенна  в pathFoto 
pathFoto = 'data/images/'       #'data/фото'  # папка источник файлов относительный путь
paperNew = 'newYolovo'          # папка где будут новые файлы, вложенна  в pathFoto т.е новые файлы будут в папке 'data/фото/newfoto'
paperRemdg = 'newYolovoRemd'    # папка где будут новые файлы при оборботке rembg, вложенна  в paperRemdg т.е новые файл

tempfile = 'tmp.jpg'    # название временный файл один можно удалить после работы

# ============================= переменные для настройки рабрты кода===================================

#==================== коэффициенты для позиционированния
x0_1 =  0.9373 #   0.9323 
y0_1 =  0.2442 
x0_2 =  0.9373 
y0_2 =  0.7427 
distO_O =  0.485  # между верх и нижн номерами пасп  изменяя раздвигаешь рамеку
distO_OL1_centrY =  0.255814 #
distO_OL1_centrX = 0.4325
x2 =  0.246 
y2 =  0.2209 
x3 =  0.5967 
y3 =  0.2226 
dist2_3 =  0.34  # между лев и прав дата и ном отд
dist2L3_centrX =  0.2545
dist2L3_centrY = -0.2790
#==================== коэффициенты для позиционированния

def cutFoto(image, fitchList):
    ''' Обрезка картинки по фитчам '''
    labels, fitchNp  = reraitLineToNp(fitchList) # сделать N
    ind0 = [index for (index, item) in enumerate(labels) if item == 0] # индексы 0 боксов
    ind2 = labels.index(2)  # индексы 2 боксов 
    ind3 = labels.index(3)  # индексы 3 боксов

    if len(ind0) == 2:  # Если такие 2 фитчи нашли
        if fitchNp[ind0[0], 1]< fitchNp[ind0[1], 1]: #разбори по вертикали фитч 01 02
            indOld = ind0[0]
            ind = ind0[1]
        else: 
            indOld = ind0[1]
            ind = ind0[0]
        kY = rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) / distO_O # коэфф перевода Y в новый размер x0*kY = нов точка
        kX = rasst(fitchNp[ind2, 0:2], fitchNp[ind3, 0:2]) / dist2_3 # коэфф перевода X в новый размер возс не нужен

        x0 = fitchNp[indOld,0] -( x0_1 * kY)

        print('kY', kY, 'kX', kX, ' x0;', x0)
        # print('x0_1 * kX)', x0_1 * kX)
        # print('x0_1', x0_1, 1-x0_1,  (1-x0_1) *kX) # возможно лучше ky
        # print('y0_1', y0_1, 1-y0_1, (1-y0_1) *kY)
        
        # print('fitchNp[indOld]', fitchNp[indOld])
        # )

        x0 = x0 if x0 > 0 else 0 

        # print('x0_1 * kY)', (1-x0_1) * kX)

        x1 = fitchNp[indOld,0] + ((1-x0_1) * kY)
                
        x1 = x1 if x1 < 1 else 1 
        y0 = fitchNp[indOld,1] -( y0_1 * kY)
        y0 = y0 if y0 > 0 else 0 
        y1 = fitchNp[indOld,1] + ((1-y0_1) * kY)
        y1 = y1 if y1 < 1 else 1 

        y0, y1 = reshapeX(y0, y1, image.shape[0] )
        x0, x1 = reshapeX(x0, x1, image.shape[1] )
        # print('image.shape:',image.shape)

        
        #print('new x0, x1 , y0, y1, x1-x0, y1-y0 \n',
        #          x0, x1, ' ',y0,' ', y1,' ', x1-x0,'  ', y1-y0)
        #                y       x
        #image1 = image[10:2000,500:2050, :]
        #image1 = image[y0:y1, x0:x1, :]
        #print(image1.shape)

        #print('y0;', y0)
        viewImage(DravRectangleImage(image.copy(), [[x0,y0,x1-x0,y1-y0]]))
        viewImage(image[y0:y1, x0:x1, :],waiK=700) 
        # plt.figure(figsize= (30, 20))
        # plt.imshow(img1)
        # plt.show() 

        return image[y0:y1, x0:x1, :]
    
def cutFoto2(image, fitchList):
    ''' Обрезка картинки по фитчам продвинутый вариант работает. '''
    labels, fitchNp  = reraitLineToNp(fitchList) # сделать Np
    
    ind0 = [index for (index, item) in enumerate(labels) if item == 0] # индексы 0 боксов
    indL11 = [index for (index, item) in enumerate(labels) if item == 11]  
    indL2 = [index for (index, item) in enumerate(labels) if item == 2]      # индексы 2 боксов!!!!!!!!!!!!! дубль проверить
    indL3 = [index for (index, item) in enumerate(labels) if item == 3]      # индексы 3 боксов!!!!!!!!!!!!! дубль проверить
  #  print('Подрезка Кол во фитч:', len(fitchList), ' кол во 0х фитч:', len(ind0), 
 #         'ind2 :', len(indL2), 'ind3: ', len(indL3), 'imdL11 photo:', len(indL11) )
        # Вариант обработки по 2номеров паспорта и фото и 1 дата паспорта
    if len(ind0) == 2 and len(indL11)==1 and len(indL2)== 1 :  # Если такие 2 фитчи нашли
        if fitchNp[ind0[0], 1]< fitchNp[ind0[1], 1]: #разбори по вертикали фитч 01 02
        
            indOld = ind0[0] # кто выше
            ind = ind0[1]
        else: 
            indOld = ind0[1] #
            ind = ind0[0]
        indFot = indL11[0]  # фотка
        
        kY = rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) / distO_O # коэфф перевода Y в новый размер x0*kY = нов точка

        x0 = fitchNp[indFot,0] -(fitchNp[indFot,2]* 0.85) # в лево от центра  фото отступить 0.8 ширины # отвечает за подрезку лево
        if x0 > (fitchNp[indL2[0], 0] - (fitchNp[indL2[0], 2]* 0.6)): # если боундБокс 2 левее взять левее  него
            x0 = fitchNp[indL2[0], 0] - (fitchNp[indL2[0], 2]* 0.6)

        x0 = x0 if x0 > 0 else 0 
        
        x1 = fitchNp[indOld,0] + (fitchNp[indOld,2]*2.2) # центр номера + ширина номера *3 # отвечает за подрезку аправо
        if x1 < fitchNp[ind,0] + (fitchNp[ind,2]* 1.5): # если нижний ББ правее взять по нему
            x1 = fitchNp[ind,0] + (fitchNp[ind,2]* 1.5)
                
        x1 = x1 if x1 < 1 else 1 
        y0 = fitchNp[indOld,1] -( y0_1 * kY)  # отвечает за подрезку верха
        y0 = y0 if y0 > 0 else 0 
        y1 = fitchNp[indFot,1] + (fitchNp[indFot,3] * 1.2) # центр фото + целое фото отвечает за подрезку низа можно играть коэффициентом
        y1 = y1 if y1 < 1 else 1 

        y0, y1 = reshapeX(y0, y1, image.shape[0] )
        x0, x1 = reshapeX(x0, x1, image.shape[1] )

        viewImage(DravRectangleImage(image.copy(), [[x0,y0,x1-x0,y1-y0]]), waiK=700, nameWindow=' cunInFoto') 
    
        if len(image.shape)==2: #если фотка дез 3го измерения
            image = np.expand_dims(image, axis=-1) #добавить измерение

        return image[y0:y1, x0:x1, :]
    # Вариант обработки по 2номеров паспорта  и 1 дата паспорта
    elif len(ind0) == 2 and len(indL2)== 1 :  # Если такие 2 фитчи нашли
        if fitchNp[ind0[0], 1]< fitchNp[ind0[1], 1]: #разбори по вертикали фитч 01 02
        
            indOld = ind0[0] # кто выше
            ind = ind0[1]
        else: 
            indOld = ind0[1] #
            ind = ind0[0]
        ind2 = indL2[0]        
        kY = rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) / distO_O # коэфф перевода Y в новый размер x0*kY = нов точка

        x0 = fitchNp[ind2, 0] - (fitchNp[ind2, 2] * 0.8) 
        x0 = x0 if x0 > 0 else 0 

        x1 = fitchNp[indOld,0] + (fitchNp[indOld,2]*2.5) # центр номера + ширина номера *3
        if x1 < fitchNp[ind,0] + (fitchNp[ind,2]* 2): # если нижний ББ правее взять по нему
            x1 = fitchNp[ind,0] + (fitchNp[ind,2]* 2)
                
        x1 = x1 if x1 < 1 else 1 
        y0 = fitchNp[indOld,1] -( y0_1 * kY)
        y0 = y0 if y0 > 0 else 0 
        y1 = fitchNp[ind,1] + (fitchNp[ind,3] * 1.4) # центр фото + целое фото
        y1 = y1 if y1 < 1 else 1 

        y0, y1 = reshapeX(y0, y1, image.shape[0] )
        x0, x1 = reshapeX(x0, x1, image.shape[1] )

                   # Вспывающее окно 
        viewImage(DravRectangleImage(image.copy(), [[x0,y0,x1-x0,y1-y0]]), waiK=700, nameWindow=' cunInFoto') 

        if len(image.shape)==2: #если фотка без 3го измерения
            image = np.expand_dims(image, axis=-1) #добавить измерение

        return image[y0:y1, x0:x1, :]  # отослать обрезанное фото
       # Вариант обработки по 2номеров паспорта и фото 
    elif len(ind0) == 2 and len(indL11)==1  :  # Если такие 2 фитчи нашли и фотку
        if fitchNp[ind0[0], 1]< fitchNp[ind0[1], 1]: #разбори по вертикали фитч 01 02
        
            indOld = ind0[0] # кто выше
            ind = ind0[1]
        else: 
            indOld = ind0[1] #
            ind = ind0[0]
        indFot = indL11[0]  # фотка
      
        kY = rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) / distO_O # коэфф перевода Y в новый размер x0*kY = нов точка
        x0 = fitchNp[indFot,0] -(fitchNp[indFot,2]* 0.85) # в лево от центра  фото отступить 0

        x0 = x0 if x0 > 0 else 0 

        x1 = fitchNp[indOld,0] + (fitchNp[indOld,2]*2.5) # центр номера + ширина номера *3
        if x1 < fitchNp[ind,0] + (fitchNp[ind,2]* 2): # если нижний ББ правее взять по нему
            x1 = fitchNp[ind,0] + (fitchNp[ind,2]* 2)
                
        x1 = x1 if x1 < 1 else 1 
        y0 = fitchNp[indOld,1] -( y0_1 * kY)
        y0 = y0 if y0 > 0 else 0 
        y1 = fitchNp[ind,1] + (fitchNp[ind,3] * 1.4) # центр фото + целое фото
        y1 = y1 if y1 < 1 else 1 

        y0, y1 = reshapeX(y0, y1, image.shape[0] )
        x0, x1 = reshapeX(x0, x1, image.shape[1] )

                   # Вспывающее окно 
        viewImage(DravRectangleImage(image.copy(), [[x0,y0,x1-x0,y1-y0]]), waiK=700, nameWindow=' cunInFoto') 

        if len(image.shape)==2: #если фотка без 3го измерения
            image = np.expand_dims(image, axis=-1) #добавить измерение

        return image[y0:y1, x0:x1, :]  # отослать обрезанное фото
#%
def findContur(image):
    ''' превращение картинки в контуры удоьно убирает мусор '''
  # Convert to graycsale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
     
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    
    return edges   
      
def cutFotoRemdBoard(image, fitchList):
    ''' обрезание с использованием сети rembg  '''
    image_out = remove(image)   #удаление фона нейросетью   
    x0,y0, x1, y1 = cutFoto_for_Remd(image_out, fitchList) # получение расчнтных  границ паспорта 
    
    x0 =20 if x0 > 20 else 0                        # расширение границ 
    x1 +=20 if x1+20 < image_out.shape[1] else image_out.shape[1]

    y0 =20 if y0 > 20 else 0                         # расширение границ 
    y1 +=20 if y1+20 < image_out.shape[0] else image_out.shape[0] 
        
    image_contur = findContur(image_out) #мод картинки в контуры
    
    image_contur[:y0,...]= 0      # зануление того что за пркрелами расчетной картинки
    image_contur[y1:,...]= 0
    image_contur[:,:x0,...]= 0
    image_contur[:, x1:,...]= 0
    
    x,y,xa,ya = findBoard(image_contur)  # поиск границ паспорта
      
    return image_out[y:ya, x:xa, :]  # отослать обрезанное фото 

def cutFoto_for_Remd(image, fitchList):
    ''' Обрезка картинки вырезание фона ИИ и обрезкапо фитчам продвинутый вариант работает. '''
    
    image_out = remove(image) # удаление фона нейросетью
    
    labels, fitchNp  = reraitLineToNp(fitchList) # сделать Np
    
    ind0 = [index for (index, item) in enumerate(labels) if item == 0] # индексы 0 боксов
    indL11 = [index for (index, item) in enumerate(labels) if item == 11]  
    indL2 = [index for (index, item) in enumerate(labels) if item == 2]      # индексы 2 боксов!!!!!!!!!!!!! дубль проверить
    indL3 = [index for (index, item) in enumerate(labels) if item == 3]      # индексы 3 боксов!!!!!!!!!!!!! дубль проверить
    #print('Подрезка Кол во фитч:', len(fitchList), ' кол во 0х фитч:', len(ind0), 
    #      'ind2 :', len(indL2), 'ind3: ', len(indL3), 'imdL11 photo:', len(indL11) )
        # Вариант обработки по 2номеров паспорта и фото и 1 дата паспорта
    if len(ind0) == 2 and len(indL11)==1 and len(indL2)== 1 :  # Если такие 2 фитчи нашли
        if fitchNp[ind0[0], 1]< fitchNp[ind0[1], 1]: #разбори по вертикали фитч 01 02
        
            indOld = ind0[0] # кто выше
            ind = ind0[1]
        else: 
            indOld = ind0[1] #
            ind = ind0[0]
        indFot = indL11[0]  # фотка
        
        kY = rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) / distO_O # коэфф перевода Y в новый размер x0*kY = нов точка
        #kX = rasst(fitchNp[ind2, 0:2], fitchNp[ind3, 0:2]) / dist2_3 # коэфф перевода X в новый размер возс не нужен

        x0 = fitchNp[indFot,0] -(fitchNp[indFot,2]* 0.85) # в лево от центра  фото отступить 0.8 ширины
        if x0 > (fitchNp[indL2[0], 0] - (fitchNp[indL2[0], 2]* 0.6)): # если боундБокс 2 левее взять левее  него
            x0 = fitchNp[indL2[0], 0] - (fitchNp[indL2[0], 2]* 0.6)

        x0 = x0 if x0 > 0 else 0 

        x1 = fitchNp[indOld,0] + (fitchNp[indOld,2]*2.2) # центр номера + ширина номера *3
        if x1 < fitchNp[ind,0] + (fitchNp[ind,2]* 1.5): # если нижний ББ правее взять по нему
            x1 = fitchNp[ind,0] + (fitchNp[ind,2]* 1.5)
                
        x1 = x1 if x1 < 1 else 1 
        y0 = fitchNp[indOld,1] -( y0_1 * kY)
        y0 = y0 if y0 > 0 else 0 
        y1 = fitchNp[indFot,1] + (fitchNp[indFot,3] * 1.4) # центр фото + целое фото
        y1 = y1 if y1 < 1 else 1 

        y0, y1 = reshapeX(y0, y1, image.shape[0] )
        x0, x1 = reshapeX(x0, x1, image.shape[1] )

        viewImage(DravRectangleImage(image.copy(), [[x0,y0,x1-x0,y1-y0]]), waiK=700, nameWindow=' cunInFoto') 
        #viewImage(image[y0:y1, x0:x1, :], waiK=0, nameWindow=' cunInFoto 2') 

        if len(image.shape)==2: #если фотка дез 3го измерения
            image = np.expand_dims(image, axis=-1) #добавить измерение

        return x0, y0, x1 ,y1
    
    # Вариант обработки по 2номеров паспорта  и 1 дата паспорта
    elif len(ind0) == 2 and len(indL2)== 1 :  # Если такие 2 фитчи нашли
        if fitchNp[ind0[0], 1]< fitchNp[ind0[1], 1]: #разбори по вертикали фитч 01 02
        
            indOld = ind0[0] # кто выше
            ind = ind0[1]
        else: 
            indOld = ind0[1] #
            ind = ind0[0]
        ind2 = indL2[0]        
        kY = rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) / distO_O # коэфф перевода Y в новый размер x0*kY = нов точка
        #kX = rasst(fitchNp[ind2, 0:2], fitchNp[ind3, 0:2]) / dist2_3 # коэфф перевода X в новый размер возс не нужен

        x0 = fitchNp[ind2, 0] - (fitchNp[ind2, 2] * 0.8) 
        x0 = x0 if x0 > 0 else 0 

        x1 = fitchNp[indOld,0] + (fitchNp[indOld,2]*2.5) # центр номера + ширина номера *3
        if x1 < fitchNp[ind,0] + (fitchNp[ind,2]* 2): # если нижний ББ правее взять по нему
            x1 = fitchNp[ind,0] + (fitchNp[ind,2]* 2)
                
        x1 = x1 if x1 < 1 else 1 
        y0 = fitchNp[indOld,1] -( y0_1 * kY)
        y0 = y0 if y0 > 0 else 0 
        y1 = fitchNp[ind,1] + (fitchNp[ind,3] * 1.4) # центр фото + целое фото
        y1 = y1 if y1 < 1 else 1 

        y0, y1 = reshapeX(y0, y1, image.shape[0] )
        x0, x1 = reshapeX(x0, x1, image.shape[1] )
                   # Вспывающее окно 
        viewImage(DravRectangleImage(image.copy(), [[x0,y0,x1-x0,y1-y0]]), waiK=700, nameWindow=' cunInFoto') 

        if len(image.shape)==2: #если фотка без 3го измерения
            image = np.expand_dims(image, axis=-1) #добавить измерение

        return x0, y0, x1 ,y1 # отослать обрезанное фото
       # Вариант обработки по 2номеров паспорта и фото 
    elif len(ind0) == 2 and len(indL11)==1  :  # Если такие 2 фитчи нашли и фотку
        if fitchNp[ind0[0], 1]< fitchNp[ind0[1], 1]: #разбори по вертикали фитч 01 02
        
            indOld = ind0[0] # кто выше
            ind = ind0[1]
        else: 
            indOld = ind0[1] #
            ind = ind0[0]
        indFot = indL11[0]  # фотка
      
        kY = rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) / distO_O # коэфф перевода Y в новый размер x0*kY = нов точка
        x0 = fitchNp[indFot,0] -(fitchNp[indFot,2]* 0.85) # в лево от центра  фото отступить 0

        x0 = x0 if x0 > 0 else 0 

        x1 = fitchNp[indOld,0] + (fitchNp[indOld,2]*2.5) # центр номера + ширина номера *3
        if x1 < fitchNp[ind,0] + (fitchNp[ind,2]* 2): # если нижний ББ правее взять по нему
            x1 = fitchNp[ind,0] + (fitchNp[ind,2]* 2)
                
        x1 = x1 if x1 < 1 else 1 
        y0 = fitchNp[indOld,1] -( y0_1 * kY)
        y0 = y0 if y0 > 0 else 0 
        y1 = fitchNp[ind,1] + (fitchNp[ind,3] * 1.4) # центр фото + целое фото
        y1 = y1 if y1 < 1 else 1 

        y0, y1 = reshapeX(y0, y1, image.shape[0] )
        x0, x1 = reshapeX(x0, x1, image.shape[1] )
                   # Вспывающее окно 
        viewImage(DravRectangleImage(image.copy(), [[x0,y0,x1-x0,y1-y0]]), waiK=700, nameWindow=' cunInFoto') 

        if len(image.shape)==2: #если фотка без 3го измерения
            image = np.expand_dims(image, axis=-1) #добавить измерение

        return x0, y0, x1 ,y1  # отослать обрезанное фото
    return 0, 0, image.shape[0], image.shape[1] # если не нашел граничы по другому то всю картинку 
#%Сервисные вычисления и функции


def getFoto(path=pathFoto):
    ''' Генератор фоток из папки по умолчанию фотки должны быть в папке с относительным путем <pathFoto>  'data/фото' 
    или переделайте путь в '''
    
    files = os.listdir(path)
    for file in files:
        
        if os.path.isfile(os.path.join(path, file)):

            image = cv2.imread(os.path.join(path, file), -1)
                                    # -1: Загружает картинку в том виде, в котором она есть, включая альфу.
            yield image, file


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
    
    image = np.ascontiguousarray(image, dtype=np.uint8)
    # Рисуем квадраты 
    for (x, y, w, h) in rectangle_NP:
         
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 20) # отрисовка квадратов

    return image

def compCentrRotate(fitchList):
    '''вычисление центра вращения 
    на вход лист фитчей из сетки
    возвращает x,y центр фитчей'''
    labels, fichNpRect = reraitLineToNp(fitchList) # преобразовать фитчи сети в np.массив

    xyMin = np.min(fichNpRect, axis = 0)
    xyMax = np.max(fichNpRect, axis = 0)
    indXma = np.argmax(fichNpRect[:, 0])
    indYma = np.argmax(fichNpRect[:, 1])
    indXmi = np.argmin(fichNpRect[:, 0])
    indYmi = np.argmin(fichNpRect[:, 1])
    x= (xyMax[1] + fichNpRect[indXma, 2]/2 - xyMin[1]- fichNpRect[indXmi, 2]/2)/ 2 + xyMin[1]
    y= (xyMax[0] + fichNpRect[indYma, 3]/2 - xyMin[0]- fichNpRect[indYmi, 3]/2)/ 2 + xyMin[0]
 
    return (int(x), int(y))

def rasst(a,b):
  ''' расстояние между точками 2D '''
  return math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )

def angelRotate(veca, vecb, xy = 0):
    ''' вычисление угла поворота изображения  ''' # computingRotate(fitchNp[8,0:2], fitchNp[6,0:2], kfY, distV) # высчитывание угла поворота
    a =vecb[1] - veca[1] # расчет катета Y
    b =vecb[0] - veca[0] # расчет катета X
    c =vecb[xy] - veca[xy] # расчет катета X
    if a < 0:
      return  round(np.arcsin(c/rasst(veca, vecb))*180/np.pi) #*0.7)  # деления катета на гипотенузу
    if a > 0: 
      return  round(np.arcsin(c/rasst(veca, vecb))*180/np.pi) #*0.7) 
    else :
      return 0


def compYolov_XY(shapeImg, xY, yY,  hY, wY):
    ''' Пересчет координат в формат вывода Yolov
    xY,  yY середина бокса, hY,wY размеры бокса'''
    h = hY* shapeImg[1]
    w = wY* shapeImg[0]
    return np.array([xY* shapeImg[1]-(h/2), yY* shapeImg[0]- (w/2), h, w], dtype='int')
    

def compXYWH_Yolov(shapeImg, x,y,h,w):
    ''' Пересчет  координат Yolov в XYHW'''
    hY = h/ shapeImg[1]
    wY = w/ shapeImg[0]
    return np.array([[x/shapeImg[1]+ (hY/2), y/shapeImg[0]+ (wY/2), h/shapeImg[1], w/shapeImg[0]]])

def reraitLineToNp(fitchList):
    '''преобразование  fitchList в массив координат np'''
    numCont = [[0,0,0,0]]
    labels =[]
    for cls, x, y, h, w, _ in fitchList:
        numCont  = np.append(numCont, [[x, y, h, w ]], axis=0)
        labels.append(int(cls))
    
    return labels, numCont[1:,:]

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


def mostRotate(fitchList):
    ''' Проверка на появление фитчей предположительно будут 
    появляться когда паспорт находится в меньше 45 градусов поворота 
    вход : список фитч из нейросети
    выход: угол если нашли и False если нет'''
    stepRotate = 45
    rotateAngel = stepRotate


    if len(fitchList)> 11: # если фитчей больше 11 то можно работать условия приблизительно идеальны
 
        rotateAngel = 180  # значит уже или 180 или до30
        labels, fitchNp  = reraitLineToNp(fitchList) # сделать NP
        ind0 = [index for (index, item) in enumerate(labels) if item == 0] # индексы 0 боксов
        indL11 = [index for (index, item) in enumerate(labels) if item == 11]  
        indL2 = [index for (index, item) in enumerate(labels) if item == 2]      # индексы 2 боксов!!!!!!!!!!!!! дубль проверить
        indL3 = [index for (index, item) in enumerate(labels) if item == 3]      # индексы 3 боксов!!!!!!!!!!!!! дубль проверить
        #print('Кол во фитч:', len(fitchList), ' кол во 0х фитч:', len(ind0),
        #      'ind2 :', len(indL2), 'ind3: ', len(indL3) ,
        #      'ind11 фото:', len(indL11))

        
        if len(ind0) == 2 and len(indL2)==1 and len(indL3)==1 :  # Если такие 2 фитчи нашли
            if fitchNp[ind0[0], 1]< fitchNp[ind0[1], 1]: #разбори по вертикали фитч 01 02
                indOld = ind0[0]
                ind = ind0[1]
            else: 
                indOld = ind0[1]
                ind = ind0[0]
            ind2 = indL2[0]    # фитчи 2 
            ind3 = indL3[0]    # и 3

            if fitchNp[ind0[0], 0] < fitchNp[ind2, 0] and fitchNp[ind0[0], 0]< fitchNp[ind3, 0]:# если фитчи 2 и 3 правее 0 то переворачиваем фото
                #print(' если фитчи 2 и 3 правее 0 переворачиваем фото')
                return 180
            
            if rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) > fitchNp[indOld, 2]: # если боксы не совпадают
            
            #kY = rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) / distO_O # коэфф перевода Y в новый размер x0*kY = нов точка
            #kX = rasst(fitchNp[ind2, 0:2], fitchNp[ind3, 0:2]) / dist2_3 # коэфф перевода X в новый размер возс не нужен
                angl23 = angelRotate(fitchNp[ind3, 0:2], fitchNp[ind2, 0:2], 1)
                ange00 = angelRotate(fitchNp[indOld, 0:2], fitchNp[ind, 0:2], 0)
                #print( 'улол по 0 0 :',ange00, '  угол по ', angl23, 'Среднее:', (ange00+angl23)/2 )
                
                return  (ange00+angl23)/2  # возвращаем среднее 2х углов
            
        if len(indL11) == 1  and len(ind0) == 2: #если одна фотка
            ind11  = indL11[0]
            
            if fitchNp[ind0[0], 1]< fitchNp[ind0[1], 1]: #разбори по вертикали фитч 01 02
                indOld = ind0[0]
                ind = ind0[1]
            else: 
                indOld = ind0[1]
                ind = ind0[0]
            if fitchNp[ind11,0] > fitchNp[indOld, 0] and fitchNp[ind11,0] > fitchNp[ind, 0]:
                return 180
            
            if rasst(fitchNp[indOld, 0:2], fitchNp[ind, 0:2]) > fitchNp[indOld, 2]: # если боксы не совпадают
            
                ange00 = angelRotate(fitchNp[indOld, 0:2], fitchNp[ind, 0:2], 0)
                print( 'улол по 0 0 :',ange00, )
                
                return  ange00  # возвращаем среднее 2х углов
                     
        # Вариант когда нет 2 и3     !!!!!!!!!!!!!!!!!!!
        if len(indL2)==1 and len(indL3)==1:  # Если такие 2 фитчи нашли
        
            ind2 = indL2[0]
            ind3 = indL3[0]

            if fitchNp[ind3, 0] < fitchNp[ind2, 0]:# если фитчи 2 и 3 правее 0 то переворачиваем фото
                print(' если фитчи 2 и 3 правее фото')
                return 180

            return angelRotate(fitchNp[ind2, 0:2], fitchNp[ind3, 0:2])
               
    return rotateAngel


def rotationNP(img):
    ''' вращение картинки методом numpy '''
    img = np.rot90(img, k=-1)

    return img

def rotation(img, center, angel):
    ''' Вращение картинки методом openCV
    но много требует плясок и режет края numpy привычнее'''
    (h, w) = img.shape[:2]
    #center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, -angel, 1)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    return rotated

def reshapeX(a,b,shapeA):
    return int(a*shapeA), int(b*shapeA)


def saveImage(file, path=pathFoto, finename= 'newfoto.jpg'):
    ''' Сохранение картинки по умолчанию относительный путь к папке источнику path=data/фото  + вложенная папка paper = 'newfoto'
    можно изменить на ваше усмотрение или передать новое при вызове'''
        
    print('saveImage Сохранаяю :', os.path.join(path, finename), file.shape )
    
    cv2.imwrite( os.path.join(path, finename), file)    
     
def viewImage(image,waiK= 500, nameWindow = 'message windows', verbose = verbose):
    ''' Вывод в отдельное окошко 
    image - картинка numpy подобный массив
    waiK - int время ожидания окна если 0- будет ждать нажатия клавиши
    nameWindow - название окна лучше по английски иначе проблемы с размером
    verbose - показывать или нет True/False
    '''
    if verbose:
        cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
        cv2.imshow(nameWindow, image)
        cv2.waitKey(waiK)
        cv2.destroyAllWindows()
    else:
        pass
    return

def checPaper(paper, path= pathFoto):
    checpaper = os.path.join(pathFoto, paper)
    if  not os.path.isdir(checpaper):
        os.makedirs(checpaper)
        print('папки нет создаю!', checpaper)
    else:
        print('папки есть')
            

def predFoto(file, pathFoto = pathFoto):
    ''' загрузчик yolovo5
    на вход: название файла
            если нужно путь, по умолчанию берет  pathFoto = pathFoto
    возвращает:  
        картинку с фитчами, 
        координаты фитч (класс, координаты, чтото похожее наплощадь не понял)'''
    im0 , line = detect.run(
            weights='best.pt',  # model path or triton URL названия файла весов
            source= pathFoto + file ,  # file/dir/URL/glob/screen/0(webcam)
            #data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold доверительный порог
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=True,  # save results to *.txt     
            save_conf=True,  # save confidences in --save-txt labels  текст данные
            save_crop=True,  # save cropped prediction boxes     сохранить обрезанные поля прогноза  отдельные фитчи   
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=True,  # augmented inference      расширенный вывод
            visualize=False,  # visualize features
            #update=False,  # update all models
            #project=ROOT / 'runs/detect',  # save results to project/name
            #name='exp',  # save results to project/name
            exist_ok=True,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    )
    return im0 , line 

def oneFileDetect(image, file, fileMod = tempfile, pathFoto = pathFoto, paperNew = paperNew):
    ''' Обработчик одного файла '''
                      # новое имя файла
    saveImage(image, finename = fileMod)  # созранить во временный для загрузки в неиронку ( для синзронизации ориентации)
    
    angel = 90            # первичный угол поворота
    n = 5                 # Ограничение по попыткам прохода

    long1 = 3 # счетчик зависания в 1 градусе поворота
    flBadDetect = True   # флаг отсутствия распознавания
                         # если файлы не обрабатывались ещё то
    while angel != 0:        # цикл работы с одной фото
        
        im0 , line  = predFoto(fileMod)#fileMod)   # пред паспорт
        
          # блок проверки синхронности ориентации изображений
        if image.shape[0] != im0.shape[0]: # если  прочитанное и выданное сетью изображения по разному ориентированны
            image = rotationNP(image)      # повернуть на 90 град прочитанное
            print(' Синхронизирую изображения поворотом NP   90 ')
        if image.shape[0] == im0.shape[0]:
            print(' Изображения синхронны')
        else:
            print(' Ошибка синхронизации !!!!!!!!!!!!')

        print('\n pred:>>> фото:', file,' Количество фитчей:', len(line))         # вывод начального изо

        viewImage(im0, nameWindow='pred')            
        angel = mostRotate(line) # получение угла вращения 
        
        if abs(angel) <=1.6: # контроль вечного  угла

            long1 -=1
            if long1==0:
                angel = 0
        else:
            long1 = 2
          
        print('angel:', angel)

        if 1 > abs(angel):  #  угол поворота меньше 1 гр. запустить обрезку и сохранить
                        
            print('подрезаю')
            
            if remd:
                viewImage(im0, nameWindow='before remd') 
                
                if len(image.shape)==3:          #если фотка без 3го измерения

                    nm = image.shape[2]    # взять размер 3 го измерения
                    ind = np.equal(image, 0).all(axis=2)
                    image[ind]= np.array([255]*nm)
                
                
                imageRemdCut = remdCvFind.cutRemd(image.copy())
                if imageRemdCut.shape[0] != 16: # если фотка не прошла то формат возврата такой
                    
                    viewImage(imageRemdCut, nameWindow='remd') 
                    filtmp = 'tmp1.jpg'               # сохраняем для загрузки в yolovo
                    saveImage(imageRemdCut, finename = filtmp)
                                    
                    imX , lineNow  = predFoto(filtmp)#fileMod)   # пред паспорт
     
                    viewImage(imX ,nameWindow='Remd Cut pred '+ str(len(line)- len(lineNow))+' '+str(len(lineNow))) # показать
                    flBadDetect = False # файл распознанн и сохраннен
                    angel = 0

                    print('\n  -------------------------------------------------------- \n rembg:', fileMod)
                    saveImage(imageRemdCut, path = os.path.join(pathFoto, paperRemdg) ,finename = file)
                    image= imX
                    break
                
            
            image = cutFoto2(image, line)
            flBadDetect = False # файл распознанн и сохраннен
            angel = 0
            
            saveImage(image, path = os.path.join(pathFoto, paperNew) , finename = file)
            
            #viewImage(image,waiK =700 ,nameWindow='Cut foto '+ str(angel)) # показать
                               # вывод в окно
            break # если угол найдет и отработан то следующая фотка

        elif 1 <= abs(angel) and angel < 44 :  #  если получен угол поворота
            print("Поворачиваю на ", angel )
            centrRotate = compCentrRotate(line)   # Поиск центра вращения
           
            image = rotation(image, centrRotate, round(angel))  # Вращение методом OpenCV применимо когда есть центр вращения
            #viewImage(image, nameWindow='rotate '+str(angel))

        elif 44 < abs(angel):  #  если угол поворота больше 
            if  abs(angel) == 180:
                print(' Rotation NP   90 + ', end=' ')
                
                image = rotationNP(image)                  
            print(' Rotation NP   90 ')
            image = rotationNP(image)                             # Вращение методом numpy ( не режет края а просто поворачивает матрицу)
            


        fileMod = tempfile              # новое имя файла
        saveImage(image, finename = fileMod)
        print('Сохраняю дла повтора :', fileMod)

        n -=1
        if n<= 0:            
            break
        
    #print(' Превышен лимит поворотов')
    return flBadDetect
    
def PaperDetect(test = False):   
    ''' Обработка прохода по папкам '''
    checPaper(paperNew)
    if paperNew != paperRemdg:
        checPaper(paperRemdg)
    checPaper(badFoto)
    
    
    nextFoto = True
    genfoto= getFoto()           # инициализация генератора фоток

    
    startTime = time.time()                  # время запуска
    countFiles = len(os.listdir(pathFoto))   # кол- во фото для расч времени
    countF = os.listdir(os.path.join(pathFoto, paperNew))
    countF.extend(os.listdir(os.path.join(pathFoto, badFoto)) )  #сумма обработанных файлов
    if paperNew != paperRemdg: # если папка отдельна renmdg
        countF.extend(os.listdir(os.path.join(pathFoto, paperRemdg)) )  #сумма обработанных файлов
        
    restFiles = countFiles - len(countF)                         # осталось файлов обработать
    count =0
    
    while nextFoto:              # цикл перебора фоток
        try:
            image, file = next(genfoto)
        except StopIteration:
            print(' Перебор фротографий закончил. Время работы: ', round((time.time() -  startTime)/ 60  ), ' минут')
            break
        
        if not '.jp' in file:
            continue
        
        if file in countF: # если фай есть в папках результтов
            print(' файл уже обработан :', file)
            continue
        count +=1
        print('\n Новое--------------- фото:', file, '  ', round((time.time() - startTime) / count * (restFiles - count)/ 60), ' min', type(image))  # file1)
        
        flBadDetect = oneFileDetect(image, file)
        
        if test: # тестовый вариант
            print('Итоговая:',tempfile)
            im0 , line  = predFoto(tempfile) # Проверка
            viewImage(im0, nameWindow='itog:') 
        
        if flBadDetect: # если файл не распознанн сохранить обрезанную копию для следующего алгоритма
               print('\n Файл , не распознанн,')
                       # сохраняем в отдельную папку image с фитчами
               #saveImage(image, paper= cutFoto,  finename = filename) 
                       # просто копируем эту фотку из папки без изменений
               shutil.copyfile(os.path.join(pathFoto, file),  os.path.join(pathFoto, badFoto, file))
    return
    
def main():
    arg_v = sys.argv
#print('begin prog')
#print('sys.argv:',arg_v,' len:',len(arg_v))
        
    if len(arg_v) > 1:
        if arg_v[1]=='-f': # один файл
            file = arg_v[2]
            print('Обработка одичного файла')
            if os.path.isfile(file):

                image = cv2.imread( file, -1)
                pathList = os.path.split(file)
                
                checPaper(os.path.join(pathList[0], paperNew)) # проверяем наличие папки
                
                flBadDetect = oneFileDetect(image, file = pathList[-1], pathFoto = pathList[0])
                
                if flBadDetect:
                    print('не получилось распознать')
                else:
                    print('Результат:', os.path.join(pathFoto, paperNew, pathList[-1]))
            else:
                print('файл не найден')
                
        if arg_v[1]=='-h': # один файл
            print('Пример запуска файла из строки !python pastDetect.py -f data/images/2015.jpg')
            print('Если аргументов нет обрабатывает папку:', pathFoto)

    else:
        print(' Обрабатываю файлы ы папке:', pathFoto)
        PaperDetect()
    return
        


if __name__ == '__main__':
    
    main()

