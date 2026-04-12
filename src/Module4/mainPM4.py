# File: mainPM4.py
# Written by: Angel Hernandez
# Description: Module 4 - Portfolio Milestone - Option 1
# Requirement(s): Implementing Facial Recognition
#                 For this Portfolio Project Milestone, you will build on the programming requirements
#                 presented in Critical Thinking Assignment, Module 1, Option 1. In this Milestone, you will implement
#                 a Python program that uses facial recognition to determine if an individual face is present in a
#                 group of faces. For example, your program will use facial recognition to determine that
#                 the following individual (Man using a tablet  is not in the group of people shown here
#                       ---> Group of people standing in line)

import os
import sys
import subprocess

REQUIRED_PYTHON = (3, 9)

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' installed successfully.")

    @staticmethod
    def ensure_python_version():
        if sys.version_info[:2] != REQUIRED_PYTHON:
            print("\nERROR: Python 3.9 is required for face_recognition.")
            print(f"   You are running Python {sys.version_info.major}.{sys.version_info.minor}")
            print("\nPlease install Python 3.9 from:")
            print("https://www.python.org/downloads/release/python-390/")
            print("or run  winget install python.python.3.9 on Windows")
            print("\nAfter installing, run:")
            print("py -3.9 mainPM4.py\n")
            sys.exit(1)

class FaceRecognition:
    def __init__(self, query_faces, group_image):
        import numpy as np
        import face_recognition
        from PIL import Image, ImageDraw
        self.np = np
        self.Image = Image
        self.fr = face_recognition
        self.ImageDraw = ImageDraw
        self.query_status = {}
        self.query_encodings = []
        self.recognition_results = {}
        self.__load_images(query_faces, group_image)

    def __load_images(self, query_faces, group_image):
        print("Loading group image...")
        self.group_image = self.fr.load_image_file(group_image)
        print("Detecting faces in group image...")
        self.group_face_locations = self.fr.face_locations(self.group_image)
        self.group_encodings = self.fr.face_encodings(self.group_image, self.group_face_locations)
        # Load and encode all query faces
        for f in query_faces:
            print(f"Loading query face: {f}")
            img = self.fr.load_image_file(f)
            enc = self.fr.face_encodings(img)
            if len(enc) == 0:
                print(f"Warning: No face found in {f}")
                self.query_status[f] = "NO FACE DETECTED"
                continue
            self.query_encodings.append((f, enc[0]))
            self.query_status[f] = "FACE ENCODED"

    def recognize(self):
        pil_image = self.Image.fromarray(self.group_image)
        draw = self.ImageDraw.Draw(pil_image)
        print(f"\nFound {len(self.group_encodings)} face(s) in the group image.\n")
        print("Loaded query faces:")
        for f, status in self.query_status.items():
            print(f" - {f} → {status}")
        print("\nRecognition results:")
        # Initialize all results as NOT FOUND
        for f, _ in self.query_encodings:
            self.recognition_results[f] = "NOT FOUND"
        # Compare each group face to each query face
        for (top, right, bottom, left), group_encoding in zip(self.group_face_locations, self.group_encodings):
            best_color = "red"
            best_match = "NO MATCH"
            for filename, query_encoding in self.query_encodings:
                distance = self.fr.face_distance([query_encoding], group_encoding)[0]
                if distance < 0.6:
                    best_match = f"MATCH ({os.path.basename(filename)})"
                    best_color = "green"
                    self.recognition_results[filename] = "FOUND"
                    break
            draw.rectangle([(left, top), (right, bottom)], outline=best_color, width=3)
            draw.text((left, top - 10), best_match, fill=best_color)
        # Print recognition summary
        for f, result in self.recognition_results.items():
            print(f" - {f} → {result}")
        print("\nRecognition complete.\n")
        pil_image.show()

class TestCaseRunner:
    @staticmethod
    def run_test():
        query_faces = [
            "unknown_guy.jpg",   # man in blue shirt
            "guy_in_group.jpg"   # man with tomatoes basket
        ]
        group_image = "group.jpg"
        recognizer = FaceRecognition(query_faces, group_image)
        recognizer.recognize()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    DependencyChecker.ensure_python_version()
    dependencies = ['numpy', 'opencv-python', 'pillow', 'face_recognition']
    for d in dependencies: DependencyChecker.ensure_package(d)
    clear_screen()
    print("*** Module 4 - Portfolio Milestone – Face Recognition ***\n")
    TestCaseRunner.run_test()

if __name__ == '__main__': main()