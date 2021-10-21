import cv2 as cv
import mediapipe as mp

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

    # For detecting hand
    def detectHands(self,img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if draw:
            if self.results.multi_hand_landmarks:
                for self.hand in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, self.hand, HAND_CONNECTIONS)

    # For finding lms(landmarks) position in detected hand
    def findPos(self, img, lmNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[lmNo] # Getting lm index
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape # Get w, h
                cx, cy = int(lm.x * w), int(lm.y * h) # Convert w, h to pixels
                lmList.append([id, cx, cy]) # Add updates values to lmList

                if draw:
                    cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)

        return lmList
