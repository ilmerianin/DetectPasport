# Обработчик фото паспортов. Распознает некоторые области поворачивает и обрезает.

использует в работе 2 варианта алгоритма:

          1: Yolovo5 , openCV, 
          2: 1й вариант + OpenCV распознание лиц и remdg обрезка фона
          
 pastDetect.py - главный файл:
 
          1: !python pastDetect.py -h                          подсказка
          2: !python pastDetect.py -f data/images/2015.jpg     обрабатывает один файл 
          2: !python pastDetect.py                             обрабатывает всю папку 'data/images/'  



remdCvFind.py          !!!!! Сопутствующий файл нужно взять из проэкта и загрузить в папку с исполняемым файлом

detect.py              !!!!! Нужен модифицированный detect.py  добавить в папку /yolovo5  иначе не вернет картинку и фитчи. (старый лучше сохранить)


          !cp best.pt /content/yolov5/     # обученная модель для паспортов должен быть у вас
          !cp detect.py /content/yolov5/   # экспортируем detect.py
