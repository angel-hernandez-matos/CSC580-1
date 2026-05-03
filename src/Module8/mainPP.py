# File: mainPP.py
# Written by: Angel Hernandez
# Description: Module 8 - Working with a Generative Adversarial Network - Option 1
# Requirement(s): Build a GAN using the Keras library.
#                 The dataset to use is the CIFAR10 Image dataset - https://www.cs.toronto.edu/~kriz/cifar.html

import os
import sys
import subprocess
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", message="The model does not have any trainable weights.")

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class Cifar10Gan:
    def __init__(self):
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten, Dropout,
                                             BatchNormalization, Activation, Conv2D,
                                             Conv2DTranspose, LeakyReLU)
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.optimizers import Adam

        self.__np = np
        self.__tf = tf
        self.__plt = plt
        self.__epochs = 0
        self.d_losses = []
        self.g_losses = []
        self.__adam = Adam
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.__model_class = Model
        self.__input = Input
        self.__dense = Dense
        self.__conv2d = Conv2D
        self.__grid_images = []
        self.__reshape = Reshape
        self.__flatten = Flatten
        self.__dropout = Dropout
        self.__leaky_relu = LeakyReLU
        self.__sequential = Sequential
        self.__activation = Activation
        self.__conv2d_transpose = Conv2DTranspose
        self.__batch_normalization = BatchNormalization
        self.__class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        for f in ["generated_images", "saved_models"]: os.makedirs(f, exist_ok=True)
        self.__load_dataset()
        self.__build_gan()

    @staticmethod
    def suppress_warnings():
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    def __load_dataset(self):
        print("Loading CIFAR10...")
        print(f"Classes available are {self.__class_names}")
        self.__class = input("Select class you'd like to use (0-9, default=6 [frog]): ")
        self.__class = 6 if not self.__class.isdigit() or not (0 <= int(self.__class) <= 9) else int(self.__class)
        print(f"Using class '{self.__class_names[self.__class]}'...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.__tf.keras.datasets.cifar10.load_data()
        self.x_train = self.x_train[self.y_train.flatten() == self.__class]  # Let's use selected class in CIFAR dataset
        # Normalize to [-1, 1]
        self.x_train = (self.x_train.astype("float32") / 127.5) - 1.0
        self.img_shape = (32, 32, 3)
        self.latent_dim = 100

    def __build_generator(self):
        model = self.__sequential()
        model.add(self.__input(shape=(self.latent_dim,)))
        model.add(self.__dense(8 * 8 * 256))
        model.add(self.__leaky_relu(0.2))
        model.add(self.__reshape((8, 8, 256)))
        model.add(self.__conv2d_transpose(128, kernel_size=4, strides=2, padding="same"))
        model.add(self.__leaky_relu(0.2))
        model.add(self.__conv2d_transpose(64, kernel_size=4, strides=2, padding="same"))
        model.add(self.__leaky_relu(0.2))
        model.add(self.__conv2d(3, kernel_size=3, padding="same", activation="tanh"))
        noise = self.__input(shape=(self.latent_dim,))
        return self.__model_class(noise, model(noise))

    def __build_discriminator(self):
        model = self.__sequential()
        model.add(self.__input(shape=self.img_shape))
        model.add(self.__conv2d(64, kernel_size=3, strides=2, padding="same"))
        model.add(self.__leaky_relu(0.2))
        model.add(self.__dropout(0.3))
        model.add(self.__conv2d(128, kernel_size=3, strides=2, padding="same"))
        model.add(self.__leaky_relu(0.2))
        model.add(self.__dropout(0.3))
        model.add(self.__flatten())
        model.add(self.__dense(1, activation="sigmoid"))
        img = self.__input(shape=self.img_shape)
        validity = model(img)
        return self.__model_class(img, validity)

    def __build_gan(self):
        print("Building GAN...")
        d_optimizer = self.__adam(0.0002, 0.5)
        g_optimizer = self.__adam(0.0002, 0.5)
        print("Creating Discriminator...")
        self.discriminator = self.__build_discriminator()
        # Compile while trainable = True
        self.discriminator.compile(loss="binary_crossentropy", optimizer=d_optimizer, metrics=["accuracy"])
        print("Creating Generator...")
        self.generator = self.__build_generator()
        # Freeze discriminator ONLY for the combined model
        self.discriminator.trainable = False
        z = self.__input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = self.discriminator(img)
        self.combined = self.__model_class(z, validity)
        self.combined.compile(loss="binary_crossentropy", optimizer=g_optimizer)

    def __save_images(self, epoch, folder="generated_images"):
        idx = 0
        noise = self.__np.random.normal(0, 1, (16, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = self.__plt.subplots(4, 4, figsize=(6, 6))
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(gen_imgs[idx])
                axs[i, j].axis("off")
                idx += 1
        fig.savefig(f"{folder}/epoch_{epoch}.png")
        self.__plt.close()

    def __plot_training_curves(self):
        self.__plt.plot(self.d_losses, label="Discriminator Loss")
        self.__plt.plot(self.g_losses, label="Generator Loss")
        self.__plt.legend()
        self.__plt.title("Training Loss Curves")
        self.__plt.xlabel("Epoch")
        self.__plt.ylabel("Loss")
        self.__plt.show()

    def train(self):
        batch_size = 32
        self.__epochs = input("Epochs (default 5000): ")
        self.__epochs = 5000 if not self.__epochs.isdigit() else int(self.__epochs)
        # Record start time
        start_time = datetime.now()
        print(f"Training started at: {start_time}")

        for epoch in range(1, self.__epochs + 1):
            # Train Discriminator
            idx = self.__np.random.randint(0, self.x_train.shape[0], batch_size)
            real_imgs = self.x_train[idx]
            noise = self.__np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_imgs = self.generator.predict(noise)
            valid = self.__np.ones((batch_size, 1))
            fake = self.__np.zeros((batch_size, 1))
            d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake)
            d_loss = 0.5 * self.__np.add(d_loss_real, d_loss_fake)
            # Train Generator
            g_loss = self.combined.train_on_batch(noise, valid)
            self.d_losses.append(d_loss[0])
            self.g_losses.append(g_loss)
            # Save first epoch and last epoch images
            if epoch == 1:
                self.__save_images(epoch)
            if epoch == self.__epochs:
                self.__save_images(epoch)
            if epoch % 500 == 0:
                print(f"Epoch {epoch} | D loss: {d_loss[0]:.4f} | G loss: {g_loss:.4f}")

        # Record end time
        end_time = datetime.now()
        print(f"Training ended at: {end_time}")
        # Duration
        duration = end_time - start_time
        print(f"Total training time: {duration}")
        # Let's save the models
        self.generator.save("saved_models/generator.keras")
        self.discriminator.save("saved_models/discriminator.keras")
        self.__plot_training_curves() # Show training curves

class TestCaseRunner:
    @staticmethod
    def run_scenario():
        gan = Cifar10Gan()
        gan.train()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        Cifar10Gan.suppress_warnings()
        dependencies = ['numpy', 'pandas', 'matplotlib', 'tensorflow']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 8 - Portfolio Project - Option 1 ***\n')
        TestCaseRunner.run_scenario()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()