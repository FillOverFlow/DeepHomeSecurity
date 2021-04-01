import cv2
import glob
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

folder = 'img_convert/*'
img_array = []
video_name = 'fall_data'
#open file 
for filename in glob.glob(folder):
    print("filename => ", filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape 
    size = (height, width)
    img_array.append(img)


def visualize_predictions(img_array):
    frames = img_array
    fig, ax = plt.subplots()
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800, codec='mpeg4')


    def update(i):
        return plt.imshow(frames[i])

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(frames), 10), interval=1, repeat=False)
    anim.save('result.mp4', dpi=200, writer=writer)
    print("complete")

visualize_predictions(img_array)
