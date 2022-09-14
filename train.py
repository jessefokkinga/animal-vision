import zipfile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def unzip_data(filename):

    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()


def create_data_loaders(train_dir, image_size=(224, 224)):

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, label_mode="categorical", image_size=image_size
    )

    return train_data


def create_model(input_shape, base_model, num_classes):

    # Apply data augmentation to increase generalization ability of our model
    data_augmentation = keras.Sequential(
        [
            preprocessing.RandomFlip("horizontal"),
            preprocessing.RandomRotation(0.2),
            preprocessing.RandomZoom(0.2),
            preprocessing.RandomHeight(0.2),
            preprocessing.RandomWidth(0.2),
        ],
        name="data_augmentation",
    )

    # Apply transfer learning: freeze all layers and add extra layer that
    # is finetuned based on our data
    base_model.trainable = False

    # Add input layer, augmentation and the base model
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)

    # Apply average pooling and add trainable dense layer
    x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="output_layer"
    )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model


def main(model_path="animal_classification_model.h5"):

    INPUT_SHAPE = (224, 224, 3)
    BASE_MODEL = tf.keras.applications.EfficientNetB0(include_top=False)

    train_data = create_data_loaders(train_dir="images")
    model = create_model(
        input_shape=INPUT_SHAPE,
        base_model=BASE_MODEL,
        num_classes=len(train_data.class_names),
    )

    model.fit(train_data, epochs=4, steps_per_epoch=len(train_data))

    model.save("models/" + model_path)


if __name__ == "__main__":
    main()
