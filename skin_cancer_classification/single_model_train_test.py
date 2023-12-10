"""
v6    
Uses datagen.flow_from_dataframe instead of datagen.flow_from_directory
v5
This version provides support to Resnet, and training some backend model layers
From:
https://www.apriorit.com/dev-blog/647-ai-applying-deep-learning-to-classify-skin-cancer-types
It does not use .h5 but the new format to save models.
v4
Code that saves all training history in a pickle file.
Saves AUC, ROC, etc as part of the history, in this pickle file.

Aldebaro, 2023
"""

#  tutorial in part from https://wandb.ai/sayakpaul/efficientnet-tl/reports/Transfer-Learning-With-the-EfficientNet-Family-of-Models--Vmlldzo4OTg1Nw

import argparse
import os
import pickle
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn.metrics
import tensorflow as tf
import tensorflow_hub as hub

# To avoid the warning in
# https://github.com/tensorflow/tensorflow/issues/47554
from absl import logging
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logging.set_verbosity(logging.ERROR)

def get_training_model_xception(url, trainable=False):
    # Load the Xception base model
    base = Xception(
        include_top=False, weights="imagenet", input_shape=(200, 200, 3), pooling="avg"
    )
    
    # Set the layers of the base model to trainable or not based on the parameter
    for layer in base.layers:
        layer.trainable = trainable
    
    # Build the custom model on top of the Xception base
    model = Sequential()
    model.add(base)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    
    return model

def get_training_model_resnet(trainable=False):
    # Load ResNet152 base model
    base_model = ResNet152(weights="imagenet", include_top=False)
    
    # Customize the top layers of the model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.25)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    # Print layer names for reference
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # Set the layers of the base model to trainable or not based on the parameter
    if not trainable:
        # Freeze all layers
        for layer in base_model.layers:
            layer.trainable = False
    else:
        # Unfreeze layers up to a given point
        layer_num = 483
        for layer in base_model.layers[:layer_num]:
            layer.trainable = False
        for layer in base_model.layers[layer_num:]:
            layer.trainable = True

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def get_training_model_effnet(url, trainable=False):
    # Load the respective EfficientNet model but exclude the classification layers
    extractor = hub.KerasLayer(
        url, input_shape=(img_size, img_size, 3), trainable=trainable
    )

    # Build the head of the model on top of the base model
    model = tf.keras.models.Sequential(
        [
            extractor,
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    
    # Uncomment the following lines to print layer names for reference
    # layers = model.weights
    # for i, layer in enumerate(layers):
    #     print(i, layer.name)
    
    return model


def get_training_model_fixed():
    img_width = 120
    img_height = 90
    input_shape = (img_width, img_height, 3)
    
    model = Sequential()
    
    if False:
        model.add(Conv2D(60, (8, 8), input_shape=input_shape, activation="swish"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))
        model.add(Conv2D(53, (5, 5), activation="swish"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(tf.keras.layers.Conv2D(30,(5, 5),strides=(1, 1),padding="valid",activation="relu",input_shape=(90, 120, 3)))
        model.add(Conv2D(30, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"))
        model.add(Conv2D(20, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
        model.add(Conv2D(15, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
        model.add(Conv2D(15, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"))
        model.add(Conv2D(10, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Normalization())
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


def decrease_num_negatives(df, desired_num_negative_examples):
    """
    Create dataframe with desired_num_rows rows from df
    """
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    neg_examples = shuffled_df[shuffled_df["target"] == "0"].copy()
    neg_examples = neg_examples.head(round(desired_num_negative_examples)).copy()

    pos_examples = shuffled_df[shuffled_df["target"] == "1"].copy()
    newdf = pd.concat([neg_examples, pos_examples], ignore_index=True)
    newdf = newdf.sample(frac=1).reset_index(drop=True)  # shuffle again
    return newdf


def get_balanced_dataframe(
    df, desired_num_negative_examples, desired_num_positive_examples
):
    """
    Create dataframe with desired_num_rows rows from df
    """
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    neg_examples = shuffled_df[shuffled_df["target"] == "0"].copy()
    neg_examples = neg_examples.head(round(desired_num_negative_examples)).copy()

    pos_examples = shuffled_df[shuffled_df["target"] == "1"].copy()
    pos_examples = pos_examples.head(round(desired_num_positive_examples)).copy()

    newdf = pd.concat([neg_examples, pos_examples], ignore_index=True)
    newdf = newdf.sample(frac=1).reset_index(drop=True)  # shuffle again
    return newdf


def lr_scheduler(epoch, lr, epochs=50):
    """Decrease learning rate over epochs"""
    initial = 1e-3
    if epoch < epochs * 0.1:
        return initial
    elif epoch > epochs * 0.1 and epoch < epochs * 0.25:
        lr *= tf.math.exp(-0.1)
        return lr
    else:
        lr *= tf.math.exp(-0.008)
        return lr


if __name__ == "__main__":
    print("=====================================")
    print("Train NN classifier")

    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument(
        "--num_desired_train_examples",
        type=int,
        help="Desired number of training examples (for both classes 0 and 1)",
        required=False,
        default=-1,
    )

    args = parser.parse_args()
    num_desired_train_examples = args.num_desired_train_examples

    plt.close("all")

    #%% CHECK ALL VARIABLES AND REPLACE VALUES AS NEEDED
    simulation_ID = 22  # model from Optuna, but using AUC as loss instead of accuracy
    # --------------------------------------------------------------------------
    # num_desired_train_examples = 40
    # img_size = 240
    # model_url = 'https://tfhub.dev/google/efficientnet/b1/feature-vector/1' #non-trainable
    # model_url = 'https://tfhub.dev/google/efficientnet/b1/classification/1'

    # From https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb?hl=vi#scrollTo=FlsEcKVeuCnf
    # Find models at https://tfhub.dev/google/collections/efficientnet_v2/1
    # Choose the model
    if True:
        # Model with 7 M parameters
        MODEL_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2"
        MODEL_NAME = "efficientnet_v2_imagenet1k_b1"
        NUM_PIXELS = 240
    else:
        # Model with 200 M parameters
        MODEL_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"
        MODEL_NAME = "efficientnet_v2_imagenet21k_ft1k_xl"
        NUM_PIXELS = 512  # Define the input shape of the images

    # MODEL_URL = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2'
    # use_augment = False
    # type_augment = "Albumentations"
    # test_id = str(simulation_ID) #test number
    # data_id = "unbal" #ratio
    # effnet_model = str(1)
    # --------------------------------------------------------------------------

    # Define the folders for train, validation, and test data
    # Laptop:
    # train_folder = 'D:/temp/test_50percent_teste4/train_50percent_teste4/'
    # validation_folder = 'D:/temp/test_50percent_teste4/val_50percent_teste4/'
    # test_folder = 'D:/temp/test_50percent_teste4/test_50percent_teste4'

    # Define the folders for train, validation, and test data
    train_folder = "../../data_ham1000/HAM10000_images_part_1/"
    validation_folder = "../../data_ham1000/HAM10000_images_part_1/"
    test_folder = "../../data_ham1000/HAM10000_images_part_1/"

    train_csv = "../../data_ham1000/train.csv"
    test_csv = "../../data_ham1000/test.csv"
    validation_csv = "../../data_ham1000/validation.csv"

    if False:
        print(traindf.info())
        print(traindf.head())
        print(traindf.describe())
        print(traindf.value_counts())

    if True:
        # do not remove header
        traindf = pd.read_csv(train_csv, dtype=str)
        testdf = pd.read_csv(test_csv, dtype=str)
        validationdf = pd.read_csv(validation_csv, dtype=str)
    else:
        # remove header
        traindf = pd.read_csv(train_csv, dtype=str, header=None)
        testdf = pd.read_csv(test_csv, dtype=str, header=None)
        validationdf = pd.read_csv(validation_csv, dtype=str, header=None)

    if num_desired_train_examples != -1:
        num_desired_negative_train_examples = np.floor(num_desired_train_examples / 2.0)
        desired_num_positive_examples = (
            num_desired_train_examples - num_desired_negative_train_examples
        )
        traindf = get_balanced_dataframe(
            traindf, num_desired_negative_train_examples, desired_num_positive_examples
        )

        # testdf = decrease_num_negatives(testdf, 184)
        # validationdf = decrease_num_negatives(validationdf, 76)

    print("train:")
    print(traindf["target"].value_counts())
    print("test:")
    print(testdf["target"].value_counts())
    print("validation:")
    print(validationdf["target"].value_counts())

    # Define the image size and other parameters
    img_width = 120
    img_height = 90
    image_size = (img_width, img_height)
    batch_size = 12
    epochs = 100
    num_classes = 2

    base_name = MODEL_NAME + str(num_desired_train_examples)

    # Important: output folder
    output_dir = os.path.join(
        "../outputs/train_test/id_" + str(simulation_ID), base_name
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created folder ", output_dir)

    #%%
    # copy script
    copied_script = os.path.join(output_dir, os.path.basename(sys.argv[0]))
    shutil.copy2(sys.argv[0], copied_script)
    print("Just copied current script as file", copied_script)

    # model = get_training_model_effnet(model_url, trainable=False)
    model = get_training_model_resnet(trainable=False)
    # model = get_training_model_fixed()
    print("just got the model")

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.005),
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    # metrics=["accuracy"])

    model.summary()

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=train_folder,
        x_col="image_name",
        y_col="target",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    # Loading and preprocessing the training, validation, and test data
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validationdf,
        directory=validation_folder,
        x_col="image_name",
        y_col="target",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=test_folder,
        x_col="image_name",
        y_col="target",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    # Count effective number of examples, to make sure
    # print(train_generator.classes)  # numpy array with all labels
    # actual_num_desired_train_examples = train_generator.classes.shape[0]  #flow from directory
    actual_num_desired_train_examples = train_generator.n  # flow from dataframe
    print("actual_num_desired_train_examples=", actual_num_desired_train_examples)
    if actual_num_desired_train_examples != num_desired_train_examples:
        print(
            "Updating num_desired_train_examples to", actual_num_desired_train_examples
        )
        num_desired_train_examples = actual_num_desired_train_examples
        # update base_name
        base_name = MODEL_NAME + str(num_desired_train_examples)

    # Define the EarlyStopping callback
    metric_to_monitor = "val_auc"  #'val_accuracy'
    metric_mode = "max"
    early_stopping = EarlyStopping(
        monitor=metric_to_monitor,
        patience=10,
        mode=metric_mode,
        restore_best_weights=True,
    )
    # early_stopping = EarlyStopping(monitor='val_auc', patience=5)

    # look at https://www.tensorflow.org/guide/keras/serialization_and_saving
    # do not use HDF5 (.h5 extension)
    # best_model_name = 'best_model_' + base_name + '.h5'
    best_model_name = "best_model_" + base_name
    best_model_name = os.path.join(output_dir, best_model_name)
    mcp_save = ModelCheckpoint(
        best_model_name,
        save_best_only=True,
        monitor=metric_to_monitor,
        mode=metric_mode,
    )
    # Choose the learning rate strategy
    if False:
        # reduce when chosen metric does not improve
        reduce_lr = ReduceLROnPlateau(
            monitor=metric_to_monitor,
            factor=0.5,
            patience=3,
            verbose=1,
            min_delta=1e-5,
            mode=metric_mode,
        )
    else:
        # reduce as training evolves over time
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    # Define Tensorboard as a Keras callback
    tensorboard = TensorBoard(
        log_dir=os.path.join("../outputs/tensorboard_logs", base_name),
        # log_dir= '.\logs',
        histogram_freq=1,
        write_images=True,
    )

    # Training the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, mcp_save, reduce_lr, tensorboard],
    )

    # Save the last model
    # look at https://www.tensorflow.org/guide/keras/serialization_and_saving
    # do not use HDF5 (.h5 extension)
    # last_model_name = 'last_model_' + base_name + '.h5'
    last_model_name = "last_model_" + base_name
    last_model_name = os.path.join(output_dir, last_model_name)
    print("Saving ", last_model_name, "...")
    model.save(last_model_name)

    # Read the best model
    # add custom_objects according to https://stackoverflow.com/questions/61814614/unknown-layer-keraslayer-when-i-try-to-load-model
    model = load_model(best_model_name, custom_objects={"KerasLayer": hub.KerasLayer})

    # Evaluating the best model on the test set
    test_loss, test_accuracy, test_auc = model.evaluate(test_generator)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)
    print("Test AUC:", test_auc)

    # Generate predictions --> labels: predicted and true
    predictions = model.predict(test_generator)

    # defining the metrics
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    # Plot the metrics over epochs
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    # plt.show()

    metrics_name = "metrics_" + base_name + ".png"
    # plt.title('Metrics - ' + title_id)
    plt.savefig(os.path.join(output_dir, metrics_name))

    # Convert predictions into values 0 or 1
    N = len(predictions)
    pred_labels = np.zeros((N,))
    my_threshold = 0.5  # assume a threshold given predictions are in range [0, 1]
    for i in range(N):
        if predictions[i] > my_threshold:
            pred_labels[i] = 1
        else:
            pred_labels[i] = 0
    true_labels = test_generator.classes

    true_labels = test_generator.classes

    fpr, tpr, thresholds = roc_curve(true_labels, predictions, pos_label=1)
    auc = auc(fpr, tpr)
    print("AUC:", auc)

    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    precis = precision_score(true_labels, pred_labels)

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        true_labels, predictions, pos_label=1
    )

    print("Recall: ", recall)
    print("Precision: ", precis)
    print("F1-score: ", f1)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Plot confusion matrix with Seaborn
    plt.figure()
    sn.heatmap(cm, annot=True, fmt="d")
    # plt.title('Confusion matrix - ' + title_id)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # plt.show()

    # save the cm
    cm_filename = "cm_" + base_name + ".png"
    plt.savefig(os.path.join(output_dir, cm_filename))

    # add to history
    history.history["num_desired_train_examples"] = num_desired_train_examples
    history.history["test_accuracy"] = test_accuracy
    history.history["test_loss"] = test_loss
    history.history["test_confusion_matrix"] = cm
    history.history["recall"] = recall
    history.history["f1"] = f1
    history.history["precis"] = precis
    history.history["auc"] = auc
    history.history["roc_fpr"] = fpr
    history.history["roc_tpr"] = tpr
    history.history["roc_thresholds"] = thresholds
    history.history["pr_precision"] = pr_precision
    history.history["pr_recall"] = pr_recall
    history.history["pr_thresholds"] = pr_thresholds

    # https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
    pickle_file_path = os.path.join(output_dir, "trainHistoryDict.pickle")
    with open(pickle_file_path, "wb") as file_pi:
        pickle.dump(history.history, file_pi)

    # Write accuracies and losses to a text file
    file_name = "classification_output_" + base_name + ".txt"
    with open(os.path.join(output_dir, file_name), "w") as f:
        f.write("validation_acc\n")
        f.write(str(val_acc))
        f.write("\ntrain_acc\n")
        f.write(str(train_acc))
        f.write("\nvalidation_loss\n")
        f.write(str(val_loss))
        f.write("\ntrain_loss\n")
        f.write(str(train_loss))
        f.write("\ntest_acc\n")
        f.write(str(test_accuracy))
        f.write("\nnum_desired_train_examples\n")
        f.write(str(num_desired_train_examples))
