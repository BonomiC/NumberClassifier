import numpy as np
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

# ******************************************

from create_model import create_model
import tensorflow as tf

model = create_model()

model.load_weights('./checkpoint/saved_weights')

# ******************************************



import scipy.misc
# scipy.misc.toimage(image_array, cmin=0.0, cmax=...).save('outfile.jpg')

width = 200
height = 200
center = height//2
white = (255, 255, 255)
green = (0, 128, 0)

lastX = -1
lastY = -1

def save():
    filename = "image.png"
    print image1
    resized = image1.resize((28, 28))
    print resized
    gray = resized.convert("L")
    b = gray.getpixel((0, 0))
    print b

    pixVals = np.zeros((1, gray.width*gray.height))

    # pixVals = np.zeros((28, 28))

    print pixVals.size

    for i in range(gray.height):
        for j in range(gray.width):
            pixVals[0][i + j*gray.width] = gray.getpixel((i, j)) / 255.0 * 1

    # print pixVals

    # print pixVals.reshape((28, 28))

    gray.save(filename)

    prediction = model.predict(pixVals)
    predicted = np.argmax(prediction)

    scipy.misc.toimage(pixVals.reshape((28, 28)), cmin=0.0, cmax=1.0).save('outfile.jpg')

    print predicted    

    # image1.save(filename)

def paint(event):
    python_green = "#476042"
    # x1, y1 = (event.x - 1), (event.y - 1)
    # x2, y2 = (event.x + 1), (event.y + 1)
    # # cv.create_oval(x1, y1, x2, y2, fill="black", width=20)
    # cv.create_line(x1, y1, x2, y2)
    # draw.line([x1, y1, x2, y2], fill="black", width=20)

    global lastX
    global lastY

    x = event.x
    y = event.y
    if lastX == -1:
        cv.create_line(x-1, y-1, x+1, y+1, fill="white", width=20)
        draw.line([x-1, y-1, x+1, y+1], fill="white", width=20)
    else:
        cv.create_line(lastX, lastY, x, y, fill="white", width=20)
        draw.line([lastX, lastY, x, y], fill="white", width=20)

    lastX = x
    lastY = y

def reset(event):
    global lastX
    global lastY

    lastX = -1
    lastY = -1

def clear():
    cv.create_rectangle(0, 0, 300, 300, fill="black")
    draw.rectangle([0, 0, 300, 300], fill="black")

root = Tk()

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='black')
cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), (0,0,0))
draw = ImageDraw.Draw(image1)

# do the Tkinter canvas drawings (visible)
# cv.create_line([0, center, width, center], fill='green')

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

cv.bind("<ButtonRelease-1>", reset)

# do the PIL image/draw (in memory) drawings
# draw.line([0, center, width, center], green)

# PIL image can be saved as .png .jpg .gif or .bmp file (among others)
# filename = "my_drawing.png"
# image1.save(filename)
button = Button(text="save", command=save)
button2 = Button(text="clear", command=clear)
button.pack()
button2.pack()
root.mainloop()
