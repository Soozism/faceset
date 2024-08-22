import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import face_recognition
import os
import numpy as np

class MonitorControl:
    def __init__(self, app):
        self.app = app
        self.black_screen = None  # Black screen window

    def simulate_screen_off(self):
        # Create a full-screen black window to simulate screen off
        if not self.black_screen:
            self.black_screen = tk.Toplevel(self.app.master)
            self.black_screen.attributes("-fullscreen", True)
            self.black_screen.config(bg='black')
            self.black_screen.attributes("-topmost", True)  # Ensure it's on top
            self.black_screen.overrideredirect(True)  # Remove window borders

        self.black_screen.deiconify()  # Show the black screen

    def simulate_screen_on(self):
        # Hide the full-screen black window and resume the feed
        if self.black_screen:
            self.black_screen.withdraw()  # Hide the black screen

    def handle_monitor(self, familiar_face_detected):
        if familiar_face_detected:
            # Turn the screen back on if a familiar face is detected
            self.simulate_screen_on()
        else:
            # Turn the screen off if no familiar face is detected
            self.simulate_screen_off()

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition App")

        self.video_source = 0  # Default camera
        self.vid = cv2.VideoCapture(self.video_source)

        self.known_face_encodings = {}  # Maps names to list of encodings
        self.load_known_faces("known_faces")

        self.label = Label(master)
        self.label.pack()

        self.monitor_control = MonitorControl(self)  # Create MonitorControl instance

        self.update()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_known_faces(self, folder):
        # Traverse each subfolder in the known_faces directory
        for person_name in os.listdir(folder):
            person_folder = os.path.join(folder, person_name)
            if os.path.isdir(person_folder):
                self.known_face_encodings[person_name] = []
                
                # Load all images in the person's folder
                for filename in os.listdir(person_folder):
                    if filename.lower().endswith((".jpg", ".png")):
                        image_path = os.path.join(person_folder, filename)
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        self.known_face_encodings[person_name].extend(encodings)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            familiar_face_detected = False
            recognized_name = "False"  # Default value if no familiar face is detected

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                threshold = 0.6  # Set a threshold to minimize false positives

                # Check if the face matches any known faces
                for person_name, encodings in self.known_face_encodings.items():
                    face_distances = face_recognition.face_distance(encodings, face_encoding)

                    if face_distances.size > 0:
                        closest_distance = min(face_distances)
                        if closest_distance < threshold:
                            familiar_face_detected = True
                            name = person_name
                            break

                # Draw a rectangle around the face and label it
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                # Print the recognized name in the terminal
                print(f"Recognized: {name}")

            if familiar_face_detected:
                recognized_name = name  # Update with the recognized name
            else:
                print("False")

            # Handle monitor based on recognition result
            self.monitor_control.handle_monitor(familiar_face_detected)

            # Convert the frame to PhotoImage
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        # Update every 30 ms for smoother video
        self.label.after(30, self.update)

    def on_closing(self):
        self.vid.release()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

