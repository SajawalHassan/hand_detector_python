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

    # Used for detecting hand
    def detectHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        self.face_coordinates = self.results.multi_hand_landmarks

        if draw:
            if self.face_coordinates:
                for self.hand in self.face_coordinates:
                    self.mpDraw.draw_landmarks(img, self.hand, HAND_CONNECTIONS)

    # Used for finding lms(landmarks) position in detected hand
    def findPos(self, img, lmNo=0, draw=True):

        lmList = []

        if self.face_coordinates:
            myHand = self.face_coordinates[lmNo] # Getting lm index
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape # Get w, h
                cx, cy = int(lm.x * w), int(lm.y * h) # Convert w, h to pixels
                lmList.append([id, cx, cy]) # Add updates values to lmList

                if draw:
                    cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)

        return lmList

def main():

    pTime = 0

    # Capturing vid
    capture = cv.VideoCapture(0)

    # Creating instanse of hand detector
    detector = HandDetector(detection_con=0.75)

    while True:
        # Reading current frame
        success, img = capture.read()

        # If can't read currunt frame, break loop
        if not success:
            break

        detector.detectHands(img)

        # Calculating fps
        cTime = time.time() # Getting current time
        fps = 1 / (cTime - pTime) # Minusing cTime to pTime
        pTime = cTime # Setting pTime to cTime

        # Displaying fps
        cv.putText(img, f"Fps: {int(fps)}", (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)

        cv.imshow("Video", img)
        key = cv.waitKey(1)

        if key == 27:
            break # If key is pressed, break loop


    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()