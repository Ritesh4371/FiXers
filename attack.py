"""
FGSM Adversarial Attack for YOUR Fashion-MNIST Classifier
Requires:
    fashion_classifier.h5
    class_names (1).txt
    images.zip containing test images
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import zipfile
import matplotlib.pyplot as plt


# ======================================================
# 1. Load model + class names
# ======================================================

MODEL_PATH = "fashion_classifier.h5"
CLASS_FILE = "class_names (1).txt"
ZIP_FILE = "images-20251126T134926Z-1-001.zip"
EXTRACT_DIR = "att_images/"

model = load_model(MODEL_PATH)

with open(CLASS_FILE, "r") as f:
    class_names = [line.strip() for line in f]


# ======================================================
# 2. Extract images
# ======================================================

os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_FILE, "r") as z:
    z.extractall(EXTRACT_DIR)

    print("Extracted files:")
for root, dirs, files in os.walk(EXTRACT_DIR):
    for f in files:
        print(os.path.join(root, f))

import os

VALID_EXT = (".png", ".jpg", ".jpeg")

# Recursively search all folders inside EXTRACT_DIR
image_paths = []
for root, dirs, files in os.walk(EXTRACT_DIR):
    for f in files:
        if f.lower().endswith(VALID_EXT):
            image_paths.append(os.path.join(root, f))

# Debugging print
print("Total images found:", len(image_paths))
for p in image_paths:
    print("  ", p)

if len(image_paths) == 0:
    raise Exception("No images found — check extensions or directory tree.")



# ======================================================
# 3. Image preprocessing
# ======================================================

def load_and_preprocess(path):
    img = load_img(path, color_mode="grayscale", target_size=(28, 28))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)   # shape (1,28,28,1)
    return arr


# ======================================================
# 4. FGSM Attack
# ======================================================

loss_object = tf.keras.losses.CategoricalCrossentropy()

def fgsm_attack(model, image, label, eps=0.15):
    """
    image: shape (1,28,28,1)
    label: one-hot encoded label
    """
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = loss_object(label, prediction)

    # gradient wrt input
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)

    adversarial = image + eps * signed_grad
    adversarial = tf.clip_by_value(adversarial, 0, 1)

    return adversarial.numpy()


# ======================================================
# 5. Attack all images
# ======================================================

for idx, path in enumerate(image_paths):
    img = load_and_preprocess(path)

    # predicted label
    pred = model.predict(img, verbose=0)
    orig_class = np.argmax(pred)
    orig_name = class_names[orig_class]

    # convert to one-hot
    label = tf.one_hot(orig_class, 10)[None, :]


    # generate adversarial image
    adv = fgsm_attack(model, img, label, eps=0.20)

    adv_pred = model.predict(adv, verbose=0)
    adv_class = np.argmax(adv_pred)
    adv_name = class_names[adv_class]

    print(f"\nImage {idx+1}: {os.path.basename(path)}")
    print(f"  Original prediction : {orig_name}")
    print(f"  Adversarial prediction : {adv_name}  {'<<-- MISCLASSIFIED' if adv_class != orig_class else ''}")

    # save visualization
    plt.figure(figsize=(4,2))
    plt.subplot(1,2,1)
    plt.title(orig_name)
    plt.imshow(img[0].reshape(28,28), cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title(adv_name)
    plt.imshow(adv[0].reshape(28,28), cmap="gray")
    plt.axis("off")

    out_path = f"adv_result_{idx}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"  Saved: {out_path}")

print("\nAttack completed.")
