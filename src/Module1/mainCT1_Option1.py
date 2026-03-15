# File: mainCT1_Option1.py
# Written by: Angel Hernandez
# Description: Module 1 - Critical Thinking - Option 1
# Requirement(s): Detect faces with Python. Using haarcascade instead of face_recognition because the latter
#                  is not compatible with newer versions of Python, it requires DLib and other dependencies that
#                  are not maintained. One of the requirements was to "use a pre-trained face detection model.
#                  No need to train your own", haarcascade is suitable for this task.

import os
import sys
import subprocess

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class FaceDetector:
    def __init__(self, faces="JannaEverroadFamily-146-1024x683-1920w.jpg"):
        import cv2
        import numpy as np
        from PIL import Image as img, ImageDraw as idraw
        self.__np = np
        self.__cv2 = cv2
        self.__image = img
        self.__image_draw = idraw
        self.__face_cascade = None
        self.__selected_image = cv2.imread(faces)
        self.__gray = cv2.cvtColor(self.__selected_image, cv2.COLOR_BGR2GRAY)

    def detect_faces(self):
        self.__face_cascade = self.__cv2.CascadeClassifier(self.__cv2.data.haarcascades +
                                                           "haarcascade_frontalface_default.xml")
        assert not self.__face_cascade.empty(), "Failed to load face cascade model."

        faces = self.__face_cascade.detectMultiScale(self.__gray, scaleFactor=1.2, minNeighbors=6, minSize=(60, 60))
        print(f"Found {len(faces)} face(s) in this picture.")
        pil_image = self.__image.fromarray(self.__cv2.cvtColor(self.__selected_image, self.__cv2.COLOR_BGR2RGB))
        draw = self.__image_draw.Draw(pil_image)

        # Draw red boxes
        for (x, y, w, h) in faces:
            top = y
            left = x
            bottom = y + h
            right = x + w
            print(f"A face is located at Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")
            draw.rectangle([left, top, right, bottom], outline="red", width=3)

        pil_image.show()

class TestCaseRunner:
    @staticmethod
    def run_test():
        face_detector = FaceDetector()
        face_detector.detect_faces()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'opencv-python', 'pillow']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 1 - Critical Thinking - Option 1 ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()