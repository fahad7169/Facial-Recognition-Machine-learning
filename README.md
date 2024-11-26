# Facial Recognition and Attendance System  

A Python-based machine learning project to recognize faces and mark attendance using a camera. This project trains a CNN model on facial data and provides a GUI for recording attendance.  

## Table of Contents  
- [Features](#features)  
- [Setup](#setup)  
- [Usage](#usage)  
- [Libraries Used](#libraries-used)  
- [Project Workflow](#project-workflow)  
- [Author](#author)  

---

## Features  
- Detect and recognize faces from images or camera feed.  
- Mark attendance of recognized individuals in an Excel file with details like name, course, and timing.  
- Includes data preprocessing steps to crop faces from input images for better model accuracy.  

---

## Setup  

### Step 1: Install Git (if not already installed)  
Before cloning the repository, ensure Git is installed on your system:  
- **Windows**: Download and install Git from [git-scm.com](https://git-scm.com/).  
- **Mac/Linux**: Open a terminal and run:  
  ```bash
  git --version
  ```
  
### Step 2: Clone the Repository  
First, clone this repository to your local machine:  
```bash  
git clone https://github.com/fahad7169/Facial-Recognition-Machine-learning.git  

cd Facial-Recognition-Machine-learning
```
### Step 3: Install Required Libraries 
Install all the required libraries using the **requirements.txt** file
```bash
pip install -r requirements.txt
```
**Caution**: You Must Have Python version of 3.7-3.10 to ensure all libraries works

### Step 4: Prepare the Dataset
1. Create a folder named ```dataset``` in the root directory.
2. Inside ```dataset```, create subfolders for each person.
   * Name each subfolder with the person's name (e.g., ```Fahad```).
   * Upload 10-20 images of each person into their respective subfolders.
   * Take pictures from different angles and environments. **More pictures = Better accuracy.**
  

## Usage

### Step 1: Preprocess the Images
Run the ```crop.py``` script to process and crop images:

```bash
python crop.py
```
  * A new folder named resized_images will be created.
  * It will contain subfolders similar to dataset, but with cropped face images only.

### Step 2: Train the Model
Train the CNN model using the preprocessed data:
```bash
python train.py  
```
* Once training is completed:
     * The trained model will be saved as face_recognition_model.keras.
     * The label encoder will be saved as label_encoder.pkl.

### Step 3: Test the Model
Test the model by running ```predict.py```:
```bash
python predict.py  
```
* This will open the camera feed.
* Bring the trained individuals in front of the camera to test the model's performance.

### Step 4: Use the Attendance System
To mark attendance, run the ```main.py``` script:
```bash
python main.py  
```
1. Enter the details prompted (Instructor name, Course name, Class timing).
2. The camera will open to recognize students.
3. An Excel file will be created, logging the attendance with:
   * Student name
   * Course name
   * Attendance timing
   * Instructor name
   * Class Timing
   * Attendance Date

## Libraries Used
The following libraries were used in this project:
* [OpenCV](https://pypi.org/project/opencv-python/) (for image processing and face detection)
* [NumPy](https://numpy.org/) (for numerical computations)
* [Pandas](https://pandas.pydata.org/) (for handling attendance data in Excel files)
* [TensorFlow](https://www.tensorflow.org/) (for building and training the CNN model)
* [scikit-learn](https://scikit-learn.org/stable/) (for encoding labels)
* [Tkinter](https://docs.python.org/3/library/tkinter.html) (for GUI)

## Project Workflow
1. **Dataset Preparation**: Organize images in ```dataset`` folder, ensuring diverse samples for each individual.
2. **Preprocessing**: ```crop.py``` to extract and resize facial regions for training.
3. **Model Training**: Train a CNN using ```train.py``` to recognize faces.
4. **Testing**: Test model predictions on live camera feed using ```predict.py```.
5. **Attendance System**: Use ```main.py``` to automate attendance marking.

## Author
This project was developed by Fahad as part of a hands-on machine learning and facial recognition system project.

Feel free to contribute, suggest improvements, or raise issues!
