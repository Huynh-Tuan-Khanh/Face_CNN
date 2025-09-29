import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# === Load CNN model ===
model = load_model("face_recognition_cnn.h5")

# === Load class mapping (folder → index) ===
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Đảo ngược: index → tên class
idx_to_class = {v: k for k, v in class_indices.items()}

# === Hàm dự đoán khuôn mặt ===
def predict_face(img_gray):
    img = cv2.resize(img_gray, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=-1)   # (128,128,1)
    img = np.repeat(img, 3, axis=-1)     # (128,128,3)
    img = np.expand_dims(img, axis=0)    # (1,128,128,3)

    pred = model.predict(img)
    pred_class = np.argmax(pred)
    return idx_to_class[pred_class]   # trả về tên class

# === Haar cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

# === Nhận diện từ webcam ===
def recognize_camera():
    cap = cv2.VideoCapture(0)
    print("📷 Camera đang chạy... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label = predict_face(face_roi)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition - Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_camera()
