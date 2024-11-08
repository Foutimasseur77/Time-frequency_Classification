import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

def generate_dataset():


    print("#--------------Dataset generation----------------#")
    dataset_dir = "AudioSpectro"

    data = []
    labels = []

    for class_name in ['bebop', 'membo']:
        class_path = os.path.join(dataset_dir,class_name)
        if os.path.isdir(class_path):
            label = 0 if class_name == "bebop" else 1
            for file_name in os.listdir(class_path):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(class_path,file_name)
                    data.append(np.load(file_path))
                    labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    data = data[..., np.newaxis]

    train_data, temp_data , train_labels, temp_labels = train_test_split(data, labels, test_size=0.3, random_state=42)

    val_data, test_data, val_labels, test_labels = train_test_split(temp_data,temp_labels,test_size=0.6, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data,test_labels))

    print(f"Taille du dataset d'entraînement : {len(train_dataset)} échantillons")
    print(f"Taille du dataset de validation : {len(val_dataset)} échantillons")
    print(f"Taille du dataset de test : {len(test_dataset)} échantillons")

    return train_dataset, val_dataset, test_dataset

def dataset_analysis(train_dataset, val_dataset, test_dataset):

    train_labels = np.array([label.numpy() for _, label in train_dataset])

    # Plot the label distribution
    plt.hist(train_labels, bins=2, rwidth=0.8)
    plt.xticks([0, 1], ['bebop', 'membo'])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Distribution of Labels in Train Dataset")
    plt.show()

    val_labels = np.array([label.numpy() for _, label in val_dataset])

    # Plot the label distribution
    plt.hist(val_labels, bins=2, rwidth=0.8)
    plt.xticks([0, 1], ['bebop', 'membo'])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Distribution of Labels in Validation Dataset")
    plt.show()

    test_labels = np.array([label.numpy() for _, label in test_dataset])

    # Plot the label distribution
    plt.hist(test_labels, bins=2, rwidth=0.8)
    plt.xticks([0, 1], ['bebop', 'membo'])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Distribution of Labels in Test Dataset")
    plt.show()

def model_cnn1(train_dataset, val_dataset, test_dataset):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(256,63,1)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    num_epochs=20
    history = model.fit(train_dataset,
                        batch_size=32,
                        epochs=num_epochs,
                        verbose=1,
                        validation_data=val_dataset,
                        shuffle=True)
    return model, history

def plot_history(history,num_epochs=20):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(num_epochs)

    plt.figure()
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()

    plt.figure()
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def evaluate_model(model, test_dataset):
    score = model.evaluate(test_dataset, verbose=1)
    print("Test Loss:", score[0])
    print("Test accuracy:", score[1])




if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = generate_dataset()
    dataset_analysis(train_dataset, val_dataset, test_dataset)
    model, history = model_cnn1(train_dataset, val_dataset, test_dataset)
    plot_history(history,num_epochs=20)
    evaluate_model(model, test_dataset)





"""
Modifier preprocessing pipeline pour sauvegarder les spectogramme plutôt que les npy
vérifier les dimensions des images

construire les dataset avec image_from_directory
puis modifier la taille dans input layer
applique le model cnn

"""
