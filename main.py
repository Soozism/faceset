import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk


class FaceDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Detection App")

        self.video_source = 0  # Default camera
        self.vid = cv2.VideoCapture(self.video_source)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.label = Label(master)
        self.label.pack()

        self.update()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Convert the frame to PhotoImage
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.label.after(10, self.update)

    def on_closing(self):
        self.vid.release()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
