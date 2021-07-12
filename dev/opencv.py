import cv2
import time

## init capture
frame_width = 1024
frame_height = 768

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

time.sleep(1)
fps = 1 / 5

while (cap.isOpened()):
    time.sleep(fps)
    _, frame = cap.read()
    cv2.imshow('frame', frame)
    #print(frame.shape)

    if cv2.waitKey(fps) == 27:
      break

cap.release()
cv2.destroyAllWindows()