import main
import numpy as np
import cv2

nn = main.ann()


# Creating image with all black pixels as canvas
# Filling in center as white to create drawing area
canvas = np.ones([600,600])*255
canvas[100:500,100:500] = 0

def draw_on_canvas(start_pos,end_pos):
    cv2.line(canvas,start_pos,end_pos,255,15)



