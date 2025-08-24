import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

#Učitavanje modela 
MODEL_PATH = "model_2.h5"
IMG_SIZE = (64, 64)
NUM_CLASSES = 43
model = tf.keras.models.load_model(MODEL_PATH)
#Imena klasa
class_names = [ 'Ograničenje brzine (20km/h)', 'Ograničenje brzine (30km/h)', 'Ograničenje brzine (50km/h)',
    'Ograničenje brzine (60km/h)', 'Ograničenje brzine (70km/h)', 'Ograničenje brzine (80km/h)',
    'Kraj Ograničenja brzine (80km/h)', 'Ograničenje brzine (100km/h)', 'Ograničenje brzine (120km/h)',
    'Zabrana preticanja', 'Zabrana pretjecanja za teretne automobile', 'Pravo prolaska na sljedećem raskrižju',
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
    'Prestanak zabrane pretjecanja za teretne automobile']


def preprocess_roi(roi):
    roi_resized = cv2.resize(roi, IMG_SIZE)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_norm = roi_rgb.astype('float32') / 255.0
    return np.expand_dims(roi_norm, axis=0)

# Hsv maska za plavu i crvenu boju
def get_red_blue_mask(hsv):
    #Crvena
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Plava
    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([130, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    combined = cv2.bitwise_or(red_mask1, red_mask2)
    combined = cv2.bitwise_or(combined, blue_mask)

    return combined

# detekcija i predikcija
def predict_multi(image_path):
    img = cv2.imread(image_path)
    output_img = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_mask = get_red_blue_mask(hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predictions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        if w > 30 and h > 30 and 0.6 < aspect_ratio < 1.4:
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

            if confidence > 0.94:
                label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
                predictions.append((label, (x1, y1, x2, y2)))

                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, output_img, predictions

# === GUI Aplikacija ===
root = tk.Tk()
root.title("Program za prepoznavanje prometnih znakova")

canvas_orig = tk.Canvas(root, width=300, height=300)
canvas_orig.grid(row=0, column=0, padx=10, pady=10)
canvas_result = tk.Canvas(root, width=300, height=300)
canvas_result.grid(row=0, column=1, padx=10, pady=10)

label_result = tk.Label(root, text="Potrebno učitati sliku", font=("Arial", 14))
label_result.grid(row=1, column=0, columnspan=2)

def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not file_path:
        return

    orig_cv, result_cv, preds = predict_multi(file_path)

    orig_rgb = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)

    orig_pil = Image.fromarray(orig_rgb)
    orig_pil.thumbnail((300, 300))
    orig_tk = ImageTk.PhotoImage(orig_pil)

    result_pil = Image.fromarray(result_rgb)
    result_pil.thumbnail((300, 300))
    result_tk = ImageTk.PhotoImage(result_pil)

    canvas_orig.image = orig_tk
    canvas_orig.create_image(150, 150, image=orig_tk)

    canvas_result.image = result_tk
    canvas_result.create_image(150, 150, image=result_tk)

    if preds:
        msg = "\n".join([p[0] for p in preds])
        label_result.config(text=f"Prepoznato:\n{msg}")
    else:
        label_result.config(text="Nema detektiranih znakova.")

btn_load = tk.Button(root, text="Učitaj sliku", command=open_image)
btn_load.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()
