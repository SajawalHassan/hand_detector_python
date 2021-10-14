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
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, hand, HAND_CONNECTIONS)


def main(fps=True):
    pTime = 0

    # Capturing vid (cange filename to 0 if need webcam)
    capture = cv.VideoCapture("videos/hand_vid_test.3gp")

    detector = HandDetector()

    while True:
        # Reading currunt frame
        succses, img = capture.read()

        # If can't read currunt frame, break loop
        if not succses:
            break

        detector.detectHands(frame=img)

        if fps:
            # Calculating fps
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # Displaying fps
            cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)

        cv.imshow("Video", img)
        key = cv.waitKey(1)

        if key==27:
            break # If key is pressed, break loop

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
