import cv2
import os

foldername = 'test'

img = cv2.imread(f'{foldername}/frame0.png', 0)

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('video.avi', fourcc, 1, (img.shape[1], img.shape[0]))

files_in_folder = len([file for file in os.listdir(f'{foldername}/')])

for i in range(0, files_in_folder):
    img = cv2.imread(f'{foldername}/frame{i}.png')
    video.write(img)

cv2.destroyAllWindows()
video.release()