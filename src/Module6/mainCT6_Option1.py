# File: mainCT6_Option1.py
# Written by: Angel Hernandez
# Description: Module 6 - Critical Thinking - Option 1
# Requirement(s): Implementation of CIFAR10 with CNNs Using TensorFlow.
#                 For this assignment, you will train a network to classify images from
#                 the CIFARLinks to an external site. dataset using a Convolutional Neural
#                 Network (CNN) built in TensorFlow.

import os
import sys
import subprocess
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class ArgumentDefinition(Generic[T]):
    def __init__(self, name: str, default_value: T, read_arg: bool = True):
        self.name = name
        self.read_arg = read_arg
        self.value: Optional[T] = None
        self.default_value = default_value
        self.caster = type(default_value)

    def read(self):
        value = input(f"{self.name} (default {self.default_value}): ")
        if value.strip() == "":
            self.value = self.default_value
        else:
            try:
                self.value = self.caster(value)
            except Exception:
                raise ValueError(f"Invalid value for {self.name}: {value}")

class CnnConfig:
    def __init__(self, images_per_class=10, epochs=10, normalization_factor=255.0, batch_size=64,
                 validation_split=0.2, verbosity=1, training_rounds=2, epochs_per_training=5):
        self.epochs = epochs
        self.verbosity = verbosity
        self.batch_size = batch_size
        self.training_rounds = training_rounds
        self.validation_split = validation_split
        self.images_per_class = images_per_class
        self.epochs_per_training = epochs_per_training
        self.normalization_factor = normalization_factor

        arguments = [("epochs", ArgumentDefinition("Epochs", self.epochs)),
                     ("batch_size", ArgumentDefinition("Batch Size", self.batch_size)),
                     ("verbosity", ArgumentDefinition("Verbosity", self.verbosity)),
                     ("normalization_factor", ArgumentDefinition("Normalization Factor", self.normalization_factor)),
                     ("validation_split", ArgumentDefinition("Validation Split", self.validation_split)),
                     ("epochs_per_training", ArgumentDefinition("Epochs per Training Round", self.epochs_per_training)),
                     ("training_rounds", ArgumentDefinition("Training Rounds", self.training_rounds)),
                     ("images_per_class", ArgumentDefinition("Images per class", self.images_per_class))]
        for attr_name, arg in arguments:
            if arg.read_arg:
                 arg.read()
            value = arg.value if arg.value is not None else arg.default_value
            setattr(self, attr_name, value )

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class Cifar10Cnn:
    def __init__(self, cnn_config = None):
        if cnn_config is None:
            cnn_config = CnnConfig()
        self.cnn_config = cnn_config

        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        self.__np = np
        self.__tf = tf
        self.__plt = plt
        self.model = None
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.__grid_images = []
        self.__class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.__load_dataset()

    @staticmethod
    def suppress_warnings():
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    def __load_dataset(self):
        print("\nLoading CIFAR10...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.__tf.keras.datasets.cifar10.load_data()
        self.y_train = self.y_train.flatten()
        self.y_test = self.y_test.flatten()
        print("Loading and normalizing dataset...")
        self.x_train = self.x_train.astype("float32") / self.cnn_config.normalization_factor
        self.x_test = self.x_test.astype("float32") / self.cnn_config.normalization_factor
        print("Dataset loaded and normalized.")

    def __build_model(self):
        print("Building CNN model...")
        self.model = self.__tf.keras.Sequential([
            self.__tf.keras.Input(shape=(32, 32, 3)),
            self.__tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            self.__tf.keras.layers.MaxPooling2D((2, 2)),
            self.__tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            self.__tf.keras.layers.MaxPooling2D((2, 2)),
            self.__tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            self.__tf.keras.layers.MaxPooling2D((2, 2)),
            self.__tf.keras.layers.Flatten(),
            self.__tf.keras.layers.Dense(128, activation='relu'),
            self.__tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model built successfully.")
        return self.model

    def __train_model(self, model, epochs=10):
        print(f"Training CNN for {epochs} epochs...\n")
        self.history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=self.cnn_config.batch_size,
                                     validation_split=self.cnn_config.validation_split, verbose=self.cnn_config.verbosity)
        print("Training complete.")
        return self.history

    def __evaluate_model(self, model):
        print("Evaluating model on test dataset...")
        loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
        return loss, accuracy

    def __predict_sample(self, model, index=0):
        print(f"Making prediction for test image #{index}...")
        prediction = model.predict(self.x_test[index:index + 1], verbose=0)
        predicted_class = self.__np.argmax(prediction)
        print(f"Predicted: {self.__class_names[predicted_class]}, Actual: {self.__class_names[self.y_test[index]]}")

    def __repeat_training(self, model, rounds=3, epochs=5):
        print(f"Repeating training {rounds} times to reduce loss...")
        for r in range(rounds):
            print(f"\nTraining Round: {r + 1}/{rounds}")
            self.__train_model(model, epochs=epochs)

    def train_and_classify(self):
        model = self.__build_model()
        # Initial training
        history = self.__train_model(model, epochs=self.cnn_config.epochs)
        # Repeat training to reduce loss
        self.__repeat_training(model, rounds=self.cnn_config.training_rounds, epochs=self.cnn_config.epochs_per_training)
        # Evaluate
        self.__evaluate_model(model)
        # Predict a sample
        self.__predict_sample(model, index=0)
        self.__show_images()

    def __show_images(self):
        for class_id in range(10):
            i = self.__np.where(self.y_train == class_id)[0][:self.cnn_config.images_per_class]
            self.__grid_images.append(self.x_train[i])
        self.__grid_images = self.__np.array(self.__grid_images)
        fig, axes = self.__plt.subplots(self.cnn_config.images_per_class, 10, figsize=(12, 12))
        fig.canvas.manager.set_window_title("*** Module 6 - Critical Thinking - Option 1 ***")
        for row in range(self.cnn_config.images_per_class):
            for col in range(10):
                axes[row, col].imshow(self.__grid_images[col, row])
                axes[row, col].axis("off")
        for col, label in enumerate(self.__class_names):
            axes[0, col].set_title(label, fontsize=10)
        self.__plt.tight_layout()
        self.__plt.show()

class TestCaseRunner:
    @staticmethod
    def run_scenario():
        cnn = Cifar10Cnn()
        cnn.train_and_classify()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        Cifar10Cnn.suppress_warnings()
        dependencies = ['numpy', 'pandas', 'matplotlib']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 6 - Critical Thinking - Option 1 ***\n')
        TestCaseRunner.run_scenario()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()