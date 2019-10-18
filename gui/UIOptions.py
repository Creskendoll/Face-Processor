from tkinter import Checkbutton, CENTER, Canvas, IntVar, Scale, HORIZONTAL
from video import CaptureAsync


class UIOptions(object):
    def __init__(self, window, window_title):
        self.window = window
        self.window.bind('<Escape>', lambda e: self.stop())
        self.window.title(window_title)

        # open video source (by default this will try to open the computer webcam)
        self.vid = CaptureAsync()
        self.vid.start()

        # State variables
        self.draw_landmarks = IntVar(value=1)
        self.get_emotions = IntVar(value=0)
        self.draw_outline = IntVar(value=1)
        self.emotion_update_freq = IntVar(value=30)
        self.frame_index = 0

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(
            self.window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Slider
        Checkbutton(self.window, text="Emotions", variable=self.get_emotions).pack(
            anchor=CENTER, expand=True)
        Scale(self.window, variable=self.emotion_update_freq, orient=HORIZONTAL, from_=5, to=100).pack(anchor=CENTER)

        # Options
        Checkbutton(self.window, text="Landmarks", variable=self.draw_landmarks).pack(
            anchor=CENTER, expand=True)
        Checkbutton(self.window, text="Show Outline", variable=self.draw_outline).pack(
            anchor=CENTER, expand=True)
