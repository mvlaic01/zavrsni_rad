from tkinter import Tk, Canvas, Button, Label, filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# Constants
CANVAS_SIZE = (300, 300)
MODEL_PATH = "model_2.h5"
IMG_SIZE = (64, 64)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = [
    'Ograničenje brzine (20km/h)', 'Ograničenje brzine (30km/h)', 'Ograničenje brzine (50km/h)',
    'Ograničenje brzine (60km/h)', 'Ograničenje brzine (70km/h)', 'Ograničenje brzine (80km/h)',
    'Kraj Ograničenja brzine (80km/h)', 'Ograničenje brzine (100km/h)', 'Ograničenje brzine (120km/h)',
    'Zabrana preticanja', 'Zabrana pretjecanja za teretne automobile', 'Raskrižje sa sporednom cestom pod pravim kutom',
    'Cesta s prednošću prolaza', 'Križanjem s cestom s prednošću prolaza', 'Stop', 'Zabrana prometa u oba smjera',
    'Zabrana prometa za teretne automobile', 'Zabrana prometa u jednom smjeru', 'Opasnost na cesti',
    'Zavoj u lijevo', 'Zavoj u desno', 'Dvosruki zavoj ili više uzastopnih zavoja',
    'Neravan kolnik (uzastopne izbočine i ulegnuća)', 'Sklizak kolnik', 'Suženje ceste s desne strane',
    'Radovi na cesti', 'Nailazak na prometna svjetla', 'Obilježeni pješački prijelaz', 'Djeca na cesti',
    'Biciklisti na cesti', 'Poledica', 'Divljač na cesti', 'Prestanak svih zabrana',
    'Obvezan smjer-Desno', 'Obvezan smjer-Lijevo', 'Obvezan smjer-Ravno',
    'Dopušteni smjerovi-Ravno ili Desno', 'Dopušteni smjerovi-Ravno ili lijevo',
    'Obvezno obilaženje s desne strane', 'Obvezno obilaženje s lijeve strane', 'Kružni tok prometa',
    'Prestanak zabrane pretjecanja svih vozila na motorni pogon osim mopeda',
    'Prestanak zabrane pretjecanja za teretne automobile'
]

def to_rgb(cv_img):
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

def preprocess_roi(roi):
    roi_resized = cv2.resize(roi, IMG_SIZE)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_norm = roi_rgb.astype('float32') / 255.0
    return np.expand_dims(roi_norm, axis=0)

def get_red_blue_mask(hsv):
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    combined = cv2.bitwise_or(red_mask1, red_mask2)
    combined = cv2.bitwise_or(combined, blue_mask)
    return combined

def predict_multi(image_path):
    img = cv2.imread(image_path)
    output_img = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_mask = get_red_blue_mask(hsv)

    morph_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, morph_shape)
    color_mask = cv2.dilate(color_mask, morph_shape, iterations=1)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predictions = []
    seen_classes = set()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        area = cv2.contourArea(cnt)

        if w > 20 and h > 20 and area > 500 and 0.6 < aspect_ratio < 1.4:
            pad = int(0.1 * max(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)

            roi = img[y1:y2, x1:x2]
            input_tensor = preprocess_roi(roi)
            preds = model.predict(input_tensor, verbose=0)[0]
            class_id = np.argmax(preds)
            confidence = preds[class_id]

            if confidence > 0.5 and class_id not in seen_classes:
                seen_classes.add(class_id)
                label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
                predictions.append((label, (x1, y1, x2, y2)))
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not predictions:
        input_tensor = preprocess_roi(img)
        preds = model.predict(input_tensor, verbose=0)[0]
        class_id = np.argmax(preds)
        confidence = preds[class_id]
        label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
        predictions.append((label, (0, 0, img.shape[1], img.shape[0])))
        cv2.rectangle(output_img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 255), 2)

    return img, output_img, predictions

# GUI
root = Tk()
root.title("Aplikacija za prepoznavanje prometnih znakova")

canvas_orig = Canvas(root, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])
canvas_orig.grid(row=0, column=0, padx=10, pady=10)

canvas_result = Canvas(root, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])
canvas_result.grid(row=0, column=1, padx=10, pady=10)

label_result = Label(root, text="Potrebno učitati sliku", font=("Arial", 14))
label_result.grid(row=1, column=0, columnspan=2)

def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not file_path:
        return

    orig_cv, result_cv, preds = predict_multi(file_path)

    orig_pil = Image.fromarray(to_rgb(orig_cv))
    orig_pil.thumbnail(CANVAS_SIZE)
    orig_tk = ImageTk.PhotoImage(orig_pil)

    result_pil = Image.fromarray(to_rgb(result_cv))
    result_pil.thumbnail(CANVAS_SIZE)
    result_tk = ImageTk.PhotoImage(result_pil)

    canvas_orig.image = orig_tk
    canvas_orig.create_image(CANVAS_SIZE[0] // 2, CANVAS_SIZE[1] // 2, image=orig_tk)

    canvas_result.image = result_tk
    canvas_result.create_image(CANVAS_SIZE[0] // 2, CANVAS_SIZE[1] // 2, image=result_tk)

    if preds:
        msg = "\n".join(label for label, _ in preds)
        label_result.config(text=f"Prepoznato:\n{msg}")
    else:
        label_result.config(text="Nema detektiranih znakova.")

btn_load = Button(root, text="Učitaj sliku", command=open_image)
btn_load.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()
