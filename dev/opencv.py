import cv2
import time

## init capture
frame_width = 1024
frame_height = 768

cap = cv2.VideoCapture(0)

time.sleep(1)

while (cap.isOpened()):
    _, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
      break

cap.release()
cv2.destroyAllWindows()