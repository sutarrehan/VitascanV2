from PIL import Image
import numpy as np
import cv2

# ---------------------------------------------
# FACE DETECTION FUNCTION
# ---------------------------------------------
def is_face_present(image_path):
    try:
        # Load face detection model
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        img = cv2.imread(image_path)

        # If image not read properly
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        # If at least 1 face found → True
        return len(faces) > 0

    except Exception as e:
        print("Face detection error:", e)
        return False


# ---------------------------------------------
# IMAGE PREPROCESSING (FOR MODEL)
# ---------------------------------------------
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')

        # Resize for AlexNet
        img = img.resize((227, 227))

        img_array = np.array(img).astype("float32")

        # Mean normalization (VERY IMPORTANT)
        mean = np.array([123.68, 116.779, 103.939])
        img_array = img_array - mean

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print("Preprocessing error:", e)
        return None