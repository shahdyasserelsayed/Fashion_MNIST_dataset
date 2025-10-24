# Fashion MNIST Classification Project 
This project applies both **Machine Learning** and **Deep Learning** models to the Fashion-MNIST dataset to classify clothing images.

---

##  Models Implemented

- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **PCA + SVM**
- **Fully Connected Deep Learning Model (DNN)**  


## ▶️ How to Run

### 1️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate    # Windows PowerShell
```

### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 3️⃣ Train All Models
```bash
python main.py
```

 Outputs:
- `models/fashion_model.h5` — trained Deep Learning model
- Printed accuracies for all ML & DL models

### 4️⃣ Run Flask Web App
```bash
python app.py
```
 Open web browser : http://127.0.0.1:5000


---

##  Web App Feature
Upload a fashion item image → Model predicts category

Example output:
```
Prediction: Ankle boot
Confidence: 98.52%
```

Supported Predictions:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## Requirements
- Python 3.9+
- TensorFlow
- Scikit-Learn
- Flask
- NumPy, Matplotlib, Pandas

(Already included in requirements.txt )

---

## ✨ Author
Shahd Yasser Elsayed
