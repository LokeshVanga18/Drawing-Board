import cv2
import mediapipe as mp
import time
import numpy as np

##################

xp , yp = 0,0
drawColor = (255,0,255)
brushsize = 10
erasersize = 50

##################

cam = cv2.VideoCapture(0)
cam.set(3 , 860)
cam.set(4 , 640)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
Hand = mpHands.Hands()
tips = [8 , 12 , 16 , 20]

def findLandmarks(img):
    land_marks = []
    rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

    res = Hand.process(rgb)

    if res.multi_hand_landmarks:
        for hlms in res.multi_hand_landmarks:
            mpDraw.draw_landmarks(img , hlms , mpHands.HAND_CONNECTIONS)
            for id , lms in enumerate(hlms.landmark):
                h , w , c = img.shape
                x , y = int(lms.x * w) , int(lms.y * h)
                land_marks.append([id , x , y])
    
    return land_marks

def findCounter(lms):
    finger_count = []
    if lms[4][1] < lms[3][1]:
        finger_count.append(1)
    else:
        finger_count.append(0)
    for tip in tips:
        if lms[tip][2] < lms[tip-2][2]:
            finger_count.append(1)
        else:
            finger_count.append(0)
    
    return(finger_count)

imgCanvas = np.zeros((480 , 848 , 3) , np.uint8)

while True:
    _ , img = cam.read()
    img = cv2.flip(img , 1)

    land_marks = findLandmarks(img)
    
    if len(land_marks) != 0:

        x1 , y1 = land_marks[8][1:]
        x2 , y2 = land_marks[12][1:]

        count = findCounter(land_marks)
        
        # SELECTION MODE
        if count[1] and count[2]:
            xp , yp = 0 , 0
            cv2.rectangle(img , (x1 , y1) , (x2 , y2) , drawColor , cv2.FILLED)

        # DRAWING MODE
        elif count[1] and count[2] == False:
            cv2.circle(img , (x1 , y1) , 10 , drawColor , cv2.FILLED)
            if xp == 0 and yp == 0:
                xp , yp = x1 , y1
            cv2.line(imgCanvas , (xp , yp) , (x1 , y1) , drawColor , brushsize)
            xp , yp = x1 , y1

    gray = cv2.cvtColor(imgCanvas , cv2.COLOR_BGR2GRAY)
    _ , invImg = cv2.threshold(gray , 50 , 255 , cv2.THRESH_BINARY_INV)
    invImg = cv2.cvtColor(invImg , cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img , invImg)
    img = cv2.bitwise_or(img , imgCanvas)
    
    # cv2.imshow('loki' , imgCanvas)
    cv2.imshow('LOKESH' , img)
    key = cv2.waitKey(1)
    if key == ord('l'):
        break

cam.release()
cv2.destroyAllWindows()