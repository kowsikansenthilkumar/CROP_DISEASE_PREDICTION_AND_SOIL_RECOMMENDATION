import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

os_path="D:/ML/PythonDataScience/RiceDiseasePrediction/RiceDiseasePrediction (1)/RiceDiseasePrediction"
def plot_hist(hist, metric="accuracy"):
    plt.plot(hist.history[metric])
    plt.plot(hist.history[f"{metric}"])
    plt.title(f"model {metric}")
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 10
TRAINING_DIR = Path(os_path+'/Dataset/train')
TEST_DIR = Path(os_path+'/Dataset/test')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale the images by 1/255
    rotation_range=30,  # Rotate the images by up to 30 degrees
    width_shift_range=0.05,  # Shift the images horizontally by up to 5%
    height_shift_range=0.05,  # Shift the images vertically by up to 5%
    zoom_range=0.10,  # Zoom in on the images by up to 10%
    horizontal_flip=True,  # Flip the images horizontally
    fill_mode="nearest",  # Fill in the empty pixels with the nearest pixel
    validation_split=0.2,  # Use 20% of the images for validation
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale the images by 1/255
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),  # Resize the images to (224, 224)
    batch_size=BATCH_SIZE,  # Use a batch size of 32
    class_mode="categorical",  # Use categorical labels
    subset="training",  # Use the training subset
    shuffle=True, seed=13
)

validation_generator = valid_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),  # Resize the images to (224, 224)
    batch_size=BATCH_SIZE,  # Use a batch size of 32
    class_mode="categorical",  # Use categorical labels
    subset="validation",  # Use the validation subset}
    shuffle=True, seed=13
)


class_indices = {v: k for k, v in train_generator.class_indices.items()}
print("Class Details",class_indices)

num_classes = len(class_indices)
print("Number of Classes",num_classes)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

print(f"Training Samples: {train_generator.samples}")
print(f"Validation Samples: {validation_generator.samples}")
print(f"Test Samples: {test_generator.samples}")

first_batch = train_generator.next()
first_batch_images = first_batch[0]
first_batch_labels = first_batch[1]
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 8)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = first_batch_images[ind]
        ax2.imshow(image_data)
        label_index = first_batch_labels[ind].argmax()
        ax2.set_title(f"{label_index} - {class_indices[label_index]}")
        ind += 1

fig.suptitle('First Batch of Training Images') 
plt.show()

# Count the number of samples in each class
class_counts = np.unique(train_generator.classes, return_counts=True)[1]

# Print the percentage of samples that belong to each class
print("Training set class distribution:")
for i, count in enumerate(class_counts):
    print(f'Class {i}: {count/len(train_generator.classes):.2%} ({count})')
print(f"Total samples: {train_generator.samples}")

first_batch = validation_generator.next()
first_batch_images = first_batch[0]
first_batch_labels = first_batch[1]
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 8)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = first_batch_images[ind]
        ax2.imshow(image_data)
        label_index = first_batch_labels[ind].argmax()
        ax2.set_title(f"{label_index} - {class_indices[label_index]}")
        ind += 1

fig.suptitle('First Batch of Validation Images') 
plt.show()

# Count the number of samples in each class
class_counts = np.unique(validation_generator.classes, return_counts=True)[1]

# Print the percentage of samples that belong to each class
print("Validation set class distribution:")
for i, count in enumerate(class_counts):
    print(f'Class {i}: {count/len(validation_generator.classes):.2%} ({count})')
print(f"Total samples: {validation_generator.samples}")

first_batch = test_generator.next()
first_batch_images = first_batch[0]
first_batch_labels = first_batch[1]
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 8)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = first_batch_images[ind]
        ax2.imshow(image_data)
        label_index = first_batch_labels[ind].argmax()
        ax2.set_title(f"{label_index} - {class_indices[label_index]}")
        ind += 1

fig.suptitle('First Batch of Test Images') 
plt.show()

# Count the number of samples in each class
class_counts = np.unique(test_generator.classes, return_counts=True)[1]

# Print the percentage of samples that belong to each class
print("Test set class distribution:")
for i, count in enumerate(class_counts):
    print(f'Class {i}: {count/len(test_generator.classes):.2%} ({count})')
print(f"Total samples: {test_generator.samples}")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.summary()


model.compile(
    loss="categorical_crossentropy",
    # optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"],
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer


def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):

  if multi_class:
    # If class is one hot encoded, transform to categorical labels to use compute_class_weight   
    if one_hot_encoded:
      class_series = np.argmax(class_series, axis=1)
  
    # Compute class weights with sklearn method
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    return dict(zip(class_labels, class_weights))
  else:
    # It is neccessary that the multi-label values are one-hot encoded
    mlb = None
    if not one_hot_encoded:
      mlb = MultiLabelBinarizer()
      class_series = mlb.fit_transform(class_series)

    n_samples = len(class_series)
    n_classes = len(class_series[0])

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1
    
    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))
# class_weights = generate_class_weights(train_generator.classes, multi_class=True, one_hot_encoded=False)
# class_weights
class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(train_generator.classes), y=train_generator.classes)
# Convert to dictionary
class_weights = dict(enumerate(class_weights))
print("",class_weights)


early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto', verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // BATCH_SIZE, 
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // BATCH_SIZE,
    verbose=1,
    callbacks=[early_stopping_cb, reduce_lr_cb],
    class_weight=class_weights,
    # use_multiprocessing=True,
    # workers=4,
)
model.save("model.h5")

plot_hist(history, "accuracy")
plot_hist(history, "loss")

test_loss, test_acc = model.evaluate(test_generator)
print(f"test_loss: {test_loss}, test_acc: {test_acc}")

y_pred = model.predict(test_generator)  
y_pred = y_pred.argmax(axis=1)
y_true = test_generator.classes

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

cmd = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=class_indices.values())
cmd.plot(xticks_rotation="vertical")
plt.show()