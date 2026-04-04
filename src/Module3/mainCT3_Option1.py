# File: mainCT3_Option1.py
# Written by: Angel Hernandez
# Description: Module 3 - Critical Thinking - Option 2
# Requirement(s): Linear Regression Using TensorFlow
#                 In this assignment, you will use TensorFlow to predict the next output from a given set of random
#                 inputs. Start by importing the necessary libraries. You will use Numpy along with TensorFlow for
#                 computations and Matplotlib for plotting.
# Complete the following steps:
#
# 1) Plot the training data.
# 2) Create a TensorFlow model by defining the placeholders X and Y so that you can feed your training
# examples X and Y into the optimizer during the training process.#
# 3) Declare two trainable TensorFlow variables for the weights and bias and initialize them randomly.#
# 4) Define the hyperparameters for the model:#
#      learning_rate = 0.01
#      training_epochs = 1000
# 5) Implement Python code for:
#    * the hypothesis,
#    * the cost function,
#    * the optimizer.
# 6) Implement the training process inside a TensorFlow session.#
# 7) Print out the results for the training cost, weight, and bias.#
# 8) Plot the fitted line on top of the original data.

import os
import sys
import random
import subprocess
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class ArgumentDefinition(Generic[T]):
    def __init__(self, name: str, default_value: T):
        self.name = name
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

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class PredictorConfig:
    def __init__(self):
       self.epochs = 1000
       self.random_seed = 101
       self.learning_rate = 0.01
       self.linear_data_size = 50
       arguments = [("random_seed", ArgumentDefinition("Random Seed", 101)),
                    ("linear_data_size", ArgumentDefinition("Linear Data Size (Count)", 50)),
                    ("epochs", ArgumentDefinition("Epochs", 1000)),
                    ("learning_rate", ArgumentDefinition("Learning Rate", 0.01))]
       for attr_name, arg in arguments:
           arg.read()
           setattr(self, attr_name, arg.value)

class LinearRegression:
    def __init__(self, config=None):
        if config is None:
            config = PredictorConfig()
        self.__config = config
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        self.__np = np
        self.__tf = tf
        self.__x = None
        self.__y = None
        self.__plt = plt
        self.__x_std = None
        self.__y_std = None
        self.__x_norm = None
        self.__y_norm = None
        self.__x_mean = None
        self.__y_mean = None
        self.__tensor_X = None
        self.__tensor_Y = None
        self.__post_training = None
        self.__initialize_randomizer()

    def __initialize_randomizer(self):
        random.seed(self.__config.random_seed)
        self.__np.random.seed(self.__config.random_seed)
        self.__tf.random.set_seed(self.__config.random_seed)

    @staticmethod
    def suppress_warnings():
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    def __show_graph(self, is_training=True):
        if is_training:
            fig = self.__plt.figure()
            fig.canvas.manager.set_window_title("Module 3 - Critical Thinking - Option 1")
            # Show only the training data (single plot)
            self.__plt.scatter(self.__x, self.__y, label='Training Data')
            self.__plt.xlabel("x")
            self.__plt.ylabel("y")
            self.__plt.title("Training Data")
            self.__plt.show()
        else:
            # Show training data AND fitted line side-by-side
            fig, axes = self.__plt.subplots(1, 2, figsize=(12, 5))
            fig.canvas.manager.set_window_title("Module 3 - Critical Thinking - Option 1")
            # Left - original data
            axes[0].scatter(self.__x, self.__y)
            axes[0].set_title("Training Data")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            # Right - fitted line
            axes[1].scatter(self.__x, self.__y, label="Training Data")
            axes[1].plot(self.__x, self.__post_training['y_fit'], color='red', label="Fitted Line")
            axes[1].set_title("Linear Regression Fit")
            axes[1].legend()
            self.__plt.tight_layout()
            self.__plt.show()

    def __normalize_data(self):
        # The gradient for W in linear regression is:  dW = (2 / N) * Σ( (W * x_i + b - y_i) * x_i )
        # Because x_i can be as large as ~50 in our generated dataset,
        # this gradient becomes very large, causing exploding updates and eventually NaN values during training.
        # Normalizing x and y keeps values small (≈ -2 to +2),
        # which stabilizes the gradient and prevents numerical blow‑ups that translates into NaN
        self.__x_mean = self.__np.mean(self.__x)
        self.__x_std = self.__np.std(self.__x)
        self.__y_mean = self.__np.mean(self.__y)
        self.__y_std = self.__np.std(self.__y)
        self.__x_norm = (self.__x - self.__x_mean) / self.__x_std
        self.__y_norm = (self.__y - self.__y_mean) / self.__y_std

    def __denormalize_predictions(self, y_norm_pred):
        return y_norm_pred * self.__y_std + self.__y_mean

    def __generate_random_linear_data(self):
        size = self.__config.linear_data_size
        self.__x = self.__np.linspace(0, size, size)
        self.__y = self.__np.linspace(0, size, size)
        # Add noise
        self.__x += self.__np.random.uniform(-4, 4, size)
        self.__y += self.__np.random.uniform(-4, 4, size)
        # Normalize after generation
        self.__normalize_data()

    def __train(self):
        # Convert normalized data to tensors
        self.__tensor_X = self.__tf.constant(self.__x_norm, dtype=self.__tf.float32)
        self.__tensor_Y = self.__tf.constant(self.__y_norm, dtype=self.__tf.float32)
        # Safe random initialization (Otherwise gradients will be huge thus causing NaN)
        initializer = self.__tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
        weights = self.__tf.Variable(initializer(shape=()))
        bias = self.__tf.Variable(initializer(shape=()))

        # Training loop
        print()
        for epoch in range(self.__config.epochs):
            with self.__tf.GradientTape() as tape:
                y_pred = weights * self.__tensor_X + bias
                cost = self.__tf.reduce_mean(self.__tf.square(y_pred - self.__tensor_Y))
            dw, db = tape.gradient(cost, [weights, bias])
            weights.assign_sub(self.__config.learning_rate * dw)
            bias.assign_sub(self.__config.learning_rate * db)
            if (epoch + 1) % 100 == 0: # Let's show progress every 100 epochs
                print(f"Epoch {epoch + 1}, cost={cost.numpy()}")

        # Show results in console
        print("\n*** Training complete ***")
        print("Final cost:", cost.numpy())
        print("Weight (normalized):", weights.numpy())
        print(f"Bias (normalized):{bias.numpy()}\n")
        # Compute fitted line in normalized space
        y_fit_norm = weights.numpy() * self.__x_norm + bias.numpy()
        # Denormalize for plotting
        y_fit = self.__denormalize_predictions(y_fit_norm)
        self.__post_training = {"y_fit": y_fit}

    def run(self):
        self.__generate_random_linear_data()
        self.__show_graph()
        self.__train()
        self.__show_graph(is_training=False)

class TestCaseRunner:
    @staticmethod
    def run_test():
        linear_regression = LinearRegression()
        linear_regression.run()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        LinearRegression.suppress_warnings()
        dependencies = ['numpy', 'tensorflow', 'matplotlib']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 3 - Critical Thinking - Option 1 ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()