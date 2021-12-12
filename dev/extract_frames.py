import os
import cv2


filename = 'cat.mp4'
cap = cv2.VideoCapture(filename)

filename_string_length = len(filename)

foldername = filename[:filename_string_length - 4] # delete file extension from string

if not os.path.exists(foldername):
    os.mkdir(foldername)

count = 0
while True:
    ret, image = cap.read()

    if not ret:
        break

    cv2.imwrite(os.path.join(foldername, f"frame{count}.png"), image)     
    print(f"{count} frames extracted")
    count += 1

print(f"{count} images are extracted in {foldername}.")