import cv2 as cv
import mediapipe as mp
import time

from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS

class HandDetector():
    def __init__(self, static_img=False, max_num_hands=2, detection_con=0.5, tracking_con=0.5):
        self.static_img = static_img
        self.max_num_hands = max_num_hands
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_img, max_num_hands, detection_con, tracking_con)
        self.mpDraw = mp.solutions.drawing_utils

    def detectHands(self,frame):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for self.hand in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, self.hand, HAND_CONNECTIONS)

    def findPos(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] # Getting first hand detected
            
            for id, lm in enumerate(myHand.landmark): # For each lm in detected hand
                h, w, c = img.shape # Get w, h
                cx, cy = int(lm.x * w), int(lm.y * h) # Convert w, h to pixels
                lmList.append([id, cx, cy]) # Add updates values to lmList

                if draw:
                    cv.circle(img, (cx, cy), 2, (255, 0, 0), cv.FILLED)

        return lmList
