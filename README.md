
# 🌾 Rice Grain Classifier using EfficientNetB4 and Flask

This project classifies five different types of rice grains using deep learning with EfficientNetB4 and provides a web-based interface built with Flask.

## 📂 Project Structure

```
rice_grain_classifier/
├── model/
│   └── rice_grain_model.h5             # Trained model file
│
├── templates/
│   └── index.html                      # Web UI template (Flask)
│
├── static/
│   └── style.css                       # UI styling (CSS)
│
├── uploads/                            # Temporarily stores uploaded images (ignored in Git)
│
├── app.py                              # Flask backend script
├── model(1).ipynb                      # Jupyter notebook for training the model
├── Pseudocode_rice_grains.txt          # Pseudocode description of the project
├── flow_diagram.png                    # Flow diagram image
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── .gitignore                          # Ignore rules for Git

```

## 🔍 Features
- Transfer Learning using EfficientNetB4
- Image augmentation and validation split
- Real-time prediction via a simple Flask web app
- Drag-and-drop UI with preview and confidence display

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- Flask
- HTML/CSS/JavaScript

## 🚀 How to Run

## 📁 Dataset

Due to size constraints, the dataset (~70,000 images) is not included in this repository.

You can download it from [[link-to-dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)] and place it in a folder named `data/` at the project root:



### 1. Train the model (optional if you have the `.h5` model)
```
python model(1).ipynb
```

### 2. Start the Flask app
```
python app.py
```

### 3. Open in browser
```
http://127.0.0.1:5000/
```

Upload a rice grain image and get an instant prediction.

## 📊 Supported Classes
- Arborio
- Basmati
- Ipsala
- Jasmin
- Karacadag

## 📁 Dataset Format

Your dataset folder should look like this (with rice type subfolders):
```
Rice_Image_Dataset/
├── Arborio/
├── Basmati/
├── Ipsala/
├── Jasmin/
└── Karacadag/
```

## 📌 Requirements

Install dependencies with:
```
pip install tensorflow flask pillow numpy matplotlib
```

---

## 🧩 Flow Diagram

1. User uploads rice grain image via web UI.
2. Flask app receives and saves the image.
3. Image is preprocessed and passed to the model.
4. Model predicts the rice grain type.
5. Web app displays prediction result with confidence.

---

## 🔄 Pseudocode

```
START
LOAD pre-trained EfficientNetB4 (without top layer)
FREEZE base layers for transfer learning

ADD:
  - Global Average Pooling
  - Dropout for regularization
  - Dense layer with 256 ReLU units
  - Dropout layer
  - Final softmax layer for classification

COMPILE model with Adam optimizer

TRAIN model with augmented image data (train/val split)
SAVE best model as .h5

FLASK APP:
  - Load trained model
  - Accept image upload via form
  - Preprocess uploaded image (resize, normalize)
  - Predict rice class using model
  - Show result on UI with confidence percentage
END
```

---

## 📬 Contact

Built by Tushar Desai  
For more, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/tushar-desai-14a711259/)

