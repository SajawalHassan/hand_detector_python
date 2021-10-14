import cv2 as cv
import mediapipe as mp
import time
import hand_detector_module as hand_detector

cTime = 0
pTime = 0

# Capturing vid (cange filename to 0 if need webcam)
capture = cv.VideoCapture("videos/hand_vid_test.3gp")

while True:
    # Reading currunt frame
    succses, frame = capture.read()

    # If can't read currunt frame, break loop
    if not succses:
        break

    detector = hand_detector.HandDetector()

    detector.detectHands(frame)

    # Calculating fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Displaying fps
    cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)

    cv.imshow("Video", frame)
    key = cv.waitKey(1)

    if key==27:
        break # If key is pressed, break loop

capture.release()
cv.destroyAllWindows()
