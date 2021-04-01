import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from utils.video_util import *


def visualize_clip(clip, convert_bgr=False, save_gif=False, file_path=None):
    num_frames = len(clip)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def update(i):
        if convert_bgr:
            frame = cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB)
        else:
            frame = clip[i]
        plt.imshow(frame)
        return plt

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, num_frames), interval=1)
    if save_gif:
        anim.save(file_path, dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()


def visualize_predictions(video_path, predictions, save_path):
    frames = get_video_frames(video_path)
    assert len(frames) == len(predictions)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800, codec='mpeg4')

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.set_tight_layout(True)

    fig_frame = plt.subplot(2, 1, 1)
    fig_prediction = plt.subplot(2, 1, 2)
    fig_prediction.set_xlim(0, len(frames))
    fig_prediction.set_ylim(0, 1.15)

    # print("show all frame ", frames)
    # print("show all predictions ", predictions)

    def update(i):
        print("update video frame")
        frame = frames[i]
        x = range(0, i)
        y = predictions[0:i]
        fig_prediction.plot(x, y, '-')
        fig_frame.imshow(frame)
        return plt

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(frames), 10), interval=1, repeat=False)
    anim.save('result.mp4', dpi=200, writer=writer)

    # if save_path:
    #     print("i will write this by myself...")
    #     plt.show()
    #     #anim.save(save_path, dpi=200, writer='imagemagick')
    # else:
    #     plt.show()

    


