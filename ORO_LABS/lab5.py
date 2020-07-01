import tkinter as tk
from numba import njit
import cv2
import random as r
import numpy as np

def toEdges(img):
    vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
    horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
    n,m,d = img.shape
    edges_img = img.copy()
    for row in range(3, n-2):
        for col in range(3, m-2):
            local_pixels = img[row-1:row+2, col-1:col+2, 0]
            vertical_transformed_pixels = vertical_filter*local_pixels
            vertical_score = vertical_transformed_pixels.sum()/4
            horizontal_transformed_pixels = horizontal_filter*local_pixels
            horizontal_score = horizontal_transformed_pixels.sum()/4
            edge_score = (vertical_score**2 + horizontal_score**2)**.5
            edges_img[row, col] = [edge_score]*3

    return edges_img/edges_img.max()


@njit(nopython=True, parallel=True)
def toImpulse(img, coef=1.2): # img - > 2dim numpy array(bmp), returns impulse image
    sum_ = 0.0
    max_ = np.amax(img)
    final = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sum_ += img[i][j]
            if sum_ >= coef * max_:
                final[i][j] = 1
                sum_ -= coef * max_

    return final



def Path(str='Беркут_охота_1.avi'):
    return str

print("Enter the path to the video:")
path = input()
try:
    cap = cv2.VideoCapture(path)
except Exception:
    pass
cap = cv2.VideoCapture(Path())
success, frame = cap.read()
while success:
    scale_percent = 60  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width , height)
    # resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    #resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    canny = cv2.Canny(resized, 100, 200)
    cv2.imshow('Edges', canny)
    cv2.imshow('Original video', resized)
    cv2.imshow('Impulsed noise', toImpulse(np.average(resized, 2)))
    success, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



