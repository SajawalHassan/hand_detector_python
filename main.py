import cv2 as cv
import time
import hand_detector_module as hand_detector

pTime = 0

# Capturing vid
capture = cv.VideoCapture(0)

# Creating instanse of hand detector
detector = hand_detector.HandDetector(detection_con=0.75)

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
