from tkinter import Button, CENTER, Tk, NW
import cv2
import PIL.Image
import PIL.ImageTk
import time
from video import Emotion
from gui.ImageBuilder import ImageBuilder
from os.path import abspath
from gui.UIOptions import UIOptions
from misc import Plotter, Recorder

class App(UIOptions):
    def __init__(self, window, window_title):
        super().__init__(window, window_title)
        self.p = Plotter()
        self.emotion = Emotion()

        recording_file = abspath("./recording.csv")
        self.img_builder = ImageBuilder(
            abspath("./models/landmarks.dat"), recording_file)

        self.recorder = Recorder(recording_file)

        # Button that lets the user take a snapshot
        self.btn_snapshot = Button(
            self.window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=CENTER, expand=True)

        self.window.bind('s', lambda e: self.img_builder.toggleRecording())

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.update()

        # To keep track of how many images are being sent to the emotion API
        self.sending = 0

        self.window.mainloop()

    def livePlotEmotions(self, emotion_faces: list):
        """
            `emotion_faces` is a list of EmotionFace objects.

            Updates a matplotlib plot with the emotion values found in the emotion_faces argument.
            Refer to this link for more info on live plotting: 
            https://makersportal.com/blog/2018/8/14/real-time-graphing-in-python

            There should be one chart for each face and a single chart should plot all the emotions of a single face
            Each line will represent an individual emotion, example:
            https://pydatascience.org/2017/11/24/plot-multiple-lines-in-one-chart-with-different-style-python-matplotlib/
        """
        # Substract when response is received
        self.sending -= len(emotion_faces)
        for face in emotion_faces:
            # TODO-Kaan: Update a plt with emotion values
            self.p.updatePlot(face)
            print(face)

    def stop(self):
        self.vid.stop()
        self.window.quit()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            processed_img = self.img_builder.build(frame)
            cv2.imwrite(
                "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", processed_img)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            processed_img = self.img_builder.build(
                frame, draw_landmarks=self.draw_landmarks.get(), draw_outline=self.draw_outline.get())

            if self.get_emotions.get():
                self.frame_index += 1
                if self.frame_index % self.emotion_update_freq.get() == 0:
                    # Reset counter
                    self.frame_index = 0
                    face_images = self.img_builder.getFaceImages(frame, size=(128,128))
                    # Get emotions
                    self.emotion.getPredictionAsync(face_images, self.livePlotEmotions)
                    # Increment sending counter
                    self.sending += len(face_images)

                cv2.putText(processed_img, "Sending: {}".format(self.sending), (int(self.vid.width)-150, int(self.vid.height)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Convert from cv2 image to PIL IMAGE
            rgb_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGBA)
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(rgb_image))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        self.window.after(self.delay, self.update)

if __name__ == "__main__":
    # Create a window and pass it to the Application
    root = Tk()
    App(root, "App")
