import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import math


cap = cv2.VideoCapture('/home/borg/Desktop/atividade_3_joao_araujo_vitor_miada/line_following.mp4')


# Parameters to use when opening the webcam.


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1



#hsv1, hsv2 = aux.ranges('#012E95')
cor1_v1 = np.array([ 0, 0, 240], dtype=np.uint8)
cor2_v1 = np.array([255, 15, 255], dtype=np.uint8)

print("Press q to QUIT")

def auto_canny(image, sigma=0.02):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(img_hsv, cor1_v1, cor2_v1)
    #out = cv2.bitwise_or(frame,frame,mask=mascara)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(mascara)



    # Houghlines - detects lines using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-lines-images-using-opencv-hough-lines/
    lines = []
    angulos = []
    inicio = []
    lines = cv2.HoughLines(edges,1,np.pi/180,150,0,0)

    if lines is not None:        
        #lines = np.uint16(np.around(lines))
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            inicio.append((x0,y0,a,b))
            m = theta
            angulos.append(m)

            #cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

        min1 = inicio[0]
        max1 = inicio[0]
        for i in range(0, len(inicio)):
            if inicio[i][0] < min1[0]:
                min1 =  inicio[i]
            if inicio[i][0] < max1[0]:
                max1 = inicio[i]    
        pt1_max = (int(max1[0] + 1000*(-max1[3])), int(max1[1] + 1000*(max1[2])))
        pt2_max = (int(max1[0] - 1000*(-max1[3])), int(max1[1] - 1000*(max1[2])))
        pt1_min = (int(min1[0] + 1000*(-min1[3])), int(min1[1] + 1000*(min1[2])))
        pt2_min = (int(min1[0] - 1000*(-min1[3])), int(min1[1] - 1000*(min1[2])))
        cv2.line(frame, pt1_max, pt2_max, (0,0,255), 3, cv2.LINE_AA)
        cv2.line(frame, pt1_min, pt2_min, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow('saida',frame)
    #cv2.imshow('real',edges)
    cv2.imshow('eroded',edges)
    #cv2.imshow('Detector de circulos',bordas_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
