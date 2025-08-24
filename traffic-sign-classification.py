import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.keras.utils import plot_model

# === konfiguracija ===
DATASET_DIR = r"C:\Users\MATKO\Desktop\vjezba\traffic_data\Train"
IMG_SIZE = (64, 64)
NUM_CLASSES = 43
BATCH_SIZE = 64
EPOCHS = 30
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# === Augmentacija ===
def apply_lighting(image):
    gamma = np.random.uniform(0.5, 1.5)
    return np.clip(np.power(image, gamma), 0, 1)



def apply_blur(image):
    if np.random.rand() < 0.3:
        k = np.random.choice([3, 5])
        return cv2.GaussianBlur(image, (k, k), 0)
    return image

def augment_image(image):
    image = image.astype(np.float32) / 255.0
    if np.random.rand() < 0.7:
        image = apply_lighting(image)
    if np.random.rand() < 0.5:
        image = apply_blur(image)
    return image

#Učitavnje dataset-a 
def load_dataset(path):
    images, labels = [], []
    for class_id in range(NUM_CLASSES):
        class_path = os.path.join(path, str(class_id))
        for img_path in glob(os.path.join(class_path, '*')):
            try:
                img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(class_id)
            except:
                continue
    return np.array(images), np.array(labels)

print("Unos dataset-a")
all_images, all_labels = load_dataset(DATASET_DIR)

# === SPLIT===
x_train, x_val, y_train, y_val = train_test_split(
    all_images, all_labels, test_size=0.15, stratify=all_labels, random_state=SEED
)

# ===Augmaentacija trening seta===
x_train_aug = np.array([augment_image(img) for img in x_train], dtype=np.float32)
x_val = x_val.astype(np.float32) / 255.0

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)

# === Blansiranje klasa===
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# === MODEL ===
def create_simple_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),

        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = create_simple_cnn()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#early stoppage i pohrana modela
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ModelCheckpoint("model_4.h5", save_best_only=True, monitor='val_accuracy')
]

# === Trening modela ===
history = model.fit(
    x_train_aug, y_train_cat,
    validation_data=(x_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# === Prikaz diajgramam preciznosit i gubitka ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Validacija')
plt.title('Točnost')
plt.xlabel('Epoha')
plt.ylabel('Točnost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Validacija')
plt.title('Gubitak')
plt.xlabel('Epoha')
plt.ylabel('Gubitak')
plt.legend()

plt.tight_layout()
plt.show()



# Predikcija nad test setom
y_val_pred_prob = model.predict(x_val)
y_val_pred = np.argmax(y_val_pred_prob, axis=1)

# prikz f11 rezultata
f1 = f1_score(y_val, y_val_pred, average='macro')
acc = accuracy_score(y_val, y_val_pred)
final_train_acc = history.history['accuracy'][-1]
final_train_loss = history.history['loss'][-1]
print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation Macro F1 Score: {f1:.4f}")


print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")

# matrica zbunjenosti
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title("Matrica Zbunjenosti")
plt.xlabel("Index Predikcije")
plt.ylabel("Pravi indeks")
plt.show()
