

## 💤 Real-Time Drowsiness Detection Using CNN & OpenCV

This project implements a real-time eye state detection system (open vs closed) using a **Convolutional Neural Network (CNN)** trained on eye images and integrated with **OpenCV** for webcam-based inference. It can serve as a foundation for a **driver drowsiness alert system**.

---

### 📸 Demo

![Real-Time Eye Detection](./Screenshot%202025-07-30%20235616.png)

---

### 🧠 Model Summary

* Input: Grayscale eye image (24×24)
* Output: Eye state — `Open` or `Closed`
* Accuracy: **\~99.91%** on test set

---

### 🗂️ Dataset

* **Source**: Kaggle
* **Structure**:

  ```
  train/
  ├── Closed_Eyes/
  └── Open_Eyes/
  ```
* Total Images: **4,000**
* Image Type: Grayscale `.png`

[Kaggle Dataset Link](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)

---

### 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

---

### 🚀 How It Works

1. **Train CNN** on labeled eye images.
2. **Capture webcam video** using OpenCV.
3. For each frame:

   * Convert to grayscale.
   * Resize and normalize.
   * Predict eye state with CNN.
   * Display label in real time.

---

### 🧪 Training Code Snippet

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

---

### 📹 Real-Time Detection Code

```python
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (24, 24))
    reshaped = resized.reshape(1, 24, 24, 1) / 255.0
    prediction = model.predict(reshaped)[0][0]
    label = "Open" if prediction > 0.5 else "Closed"
    cv2.putText(frame, f"Eye: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Real-time Eye Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 📁 Project Structure

```
📦 DrowsinessDetection/
├── eye_state_model.h5
├── train/
│   ├── Closed_Eyes/
│   └── Open_Eyes/
├── drowsiness_detector.py
├── Screenshot 2025-07-30 235616.png
└── README.md
```

---

### ✨ Future Work

* Add **face + eye detection** using Haar Cascades
* Play alert sound if eyes are closed for X seconds
* Extend to full drowsiness detection with blink rate, yawning, head tilt

---

Would you like this saved as a downloadable `README.md` file?
