# 🦜 Indian Bird Species Detector  

This project is a **deep learning-based image classification web application** that detects **Indian bird species** from uploaded images. It uses **PyTorch, EfficientNet-B0**, and **Streamlit** to provide a simple and interactive UI for users to upload images and get predictions.

---

## 🚀 Features  
✅ **Classifies 25 different Indian bird species**  
✅ **Uses EfficientNet-B0 for accurate predictions**  
✅ **Streamlit-powered web interface** for easy interaction  
✅ **Utilizes transfer learning** to improve model efficiency  
✅ **Automatically resumes training** from the last checkpoint  

---

## 📂 **Dataset**
The dataset used for training can be downloaded from Kaggle:
🔗 [Indian Birds Dataset (25 Classes)](https://www.kaggle.com/datasets/ichhadhari/indian-birds?select=Birds_25)
After downloading, place the dataset inside the `preprocessed-dataset` folder.

## 🛠️ Technologies Used  

| Technology | Purpose |
|------------|---------|
| **Python** | Backend logic |
| **PyTorch** | Deep learning & image classification |
| **Torchvision** | Pretrained models & image transformations |
| **Streamlit** | Web interface |
| **Pillow (PIL)** | Image handling |
| **NumPy** | Numerical computations |

---

## 📥 Installation  

### 1️⃣ **Clone the Repository**  
```bash
mkdir project-folder && cd project-folder
```
```bash
git clone https://github.com/PriyankMoon/common-indian-birds.git
```
```bash
cd common-indian-birds
```



### 2️⃣ **Install Required Libraries**  
```bash
pip install torch torchvision torchaudio
```
```bash
pip install numpy pillow streamlit
```

### 3️⃣ **Project Structure**  
```
project-folder/
│── common-indian-birds/
│   │── models/checkpoint.pth  # Folder to store trained models
│   │── preprocessed-dataset/ # Training & testing dataset
│   │   |── test (20% images for testing)
│   │   |── train (80% images for training)
│   │── main.py            # Training script for the model
|   |── app.py              # Streamlit web app for predictions
│   │── README.md           # Project Documentation

```

## 🎯 Running the Application

### 1️⃣ **Train the Model (Optional)**
If you want to train the model from scratch, run:
```
python main.py
```

### 2️⃣ **Run the Web Application**
After training, or if you're using a pre-trained model, start the app:
```
streamlit run app.py
```
This will open a web interface in your browser, where you can upload an image and get predictions. 🎉

## 🔧 Additional Notes
- Ensure you have Python 3.7+ installed.
- Modify train.py if you want to train on a different dataset.
- If any library is missing, install it manually using pip install <package-name>.

## 📜 License
This project is licensed under the MIT License. Feel free to use and modify it! 🚀

## ⭐ Contributions
- Contributions are welcome! Feel free to fork the repo and submit a pull request.
