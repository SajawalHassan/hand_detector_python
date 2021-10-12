import cv2 as cv
import mediapipe as mp

from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Capturing vid (cange filename to 0 if need webcam)
capture = cv.VideoCapture("videos/hand_vid_test.3gp")

while True:
    # Reading currunt frame
    succses, frame = capture.read()

    # If can't read currunt frame, break loop
    if not succses:
        break

    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand, HAND_CONNECTIONS)

    cv.imshow("Video", frame)
    key = cv.waitKey(1)

    if key==27:
        break # If key is pressed, break loop

capture.release()
cv.destroyAllWindows()
