import cv2
import sys, os
import numpy as np
from matplotlib import pyplot as pl

folder_dir = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(folder_dir, "images/InputImage.jpg"))
img_result = np.zeros(img.shape, dtype='uint8')

#-------- Оптимизация изображения -------------
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_filter = cv2.bilateralFilter(img_gray, 15, 20, 90) # Убираем шум фото
img_edges = cv2.Canny(img_filter, 30, 50) # Находим углы изображения
img_optimased = img_edges # Результат оптимизации
img_optimased = cv2.bilateralFilter(img_optimased, 200, 20, 90) # Убираем шум фото
del img_gray, img_filter, img_edges
#----------------------------------------------


#Выделение конутров в изображении
con_img, hir = cv2.findContours(img_optimased, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_result, con_img, -1, (0, 0, 255), 1)

print(con_img)
cv2.imshow('result', img_result)



cv2.waitKey(0)

# Дополнительные операции, которые не использовались, но могут пригодиться ----------------------------------------------------------

#con_img = cv2.GaussianBlur(con_img, (3, 3), 0)
#img = cv2.GaussianBlur(img, (5, 5), 0)
#con_img = np.zeros(img.shape,dtype='uint8')
#cv2.rectangle(img, (0,0), (500, 500), (0, 201, 105), thickness=20)
#cv2.line(img, (0,0), (img.shape[0], img.shape[1]), (203, 192, 255), thickness=5)
#cv2.line(img,(img.shape[0], 0), (0,img.shape[1]), (203, 192, 255), thickness=5)
#cv2.putText(img, 'TA DA DAM \n TAM \n TA DA DAM \n TAM', (img.shape[0]//2, img.shape[1]//2), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 3)