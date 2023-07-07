import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define the model
def create_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

while True:
    print("1. Create a new model")
    print("2. Train the model")
    print("3. Train the model from where it left off")
    print("4. Visualize the training process")
    print("5. Exit")
    choice = input("Enter your choice: ")
    if choice == '1':
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    elif choice == '2':
        history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
        tf.keras.models.save_model(model, "model.h5")
    elif choice == '3':
        model = tf.keras.models.load_model("model.h5")
        history = model.fit(x_train, y_train, epochs=5, initial_epoch=20, validation_data=(x_test, y_test))
    elif choice == '4':
        if 'history' in locals():
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            plt.plot(acc)
            plt.plot(val_acc)
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
        else:
            print("The model has not been trained yet.")
    elif choice == '5':
        break
    else:
        print("Invalid choice. Please enter a valid choice")
