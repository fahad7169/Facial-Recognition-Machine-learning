#main.py
import os
import cv2
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from predict import predict_image
from tensorflow.keras.models import load_model
import pickle
import threading


# Load the trained model and label encoder
model = load_model('face_recognition_model.keras')

def load_label_encoder(encoder_path='label_encoder.pkl'):
    """
    Load a label encoder from a pickle file.

    Args:
        encoder_path (str): The path to the pickle file containing the label encoder. 
                            Defaults to 'label_encoder.pkl'.

    Returns:
        LabelEncoder: The loaded label encoder object.
    """
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

label_encoder = load_label_encoder()


def log_attendance(student_name, course_name, instructor_name, class_time):
    # Format current date and time
    """
    Logs a student's attendance in an Excel file.

    Parameters:
        student_name (str): Name of the student.
        course_name (str): Name of the course.
        instructor_name (str): Name of the instructor.
        class_time (str): Time of the class.

    Notes:
        The attendance file is saved in a directory named "attendance_records".
        The file name is formatted as "YYYY-MM-DD_HHMMSS_attendance.xlsx".
        If the file does not exist, it is created with the headers.
        If the file exists, the new entry is appended to the existing data.
    """
    date_now = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%I:%M:%S %p')

    # Define the attendance file path
    attendance_file = f"{date_now}_{class_time.replace(':', '')}_attendance.xlsx"
    
    # Check if the directory exists, if not, create it
    attendance_dir = "attendance_records"
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)
        
    # Full path to the attendance file
    attendance_file_path = os.path.join(attendance_dir, attendance_file)

    # Data to write
    data = {
        'Student Name': [student_name],
        'Date': [date_now],
        'Time Marked': [time_now],
        'Instructor': [instructor_name],
        'Course': [course_name],
        'Class Timing': [class_time]
    }
    new_entry = pd.DataFrame(data)

    # Check if the file exists
    if os.path.exists(attendance_file_path):
        # Load existing attendance data
        existing_data = pd.read_excel(attendance_file_path)
        # Append new entry to the existing data
        updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
    else:
        # No existing file, create new data with headers
        updated_data = new_entry

    # Write updated data to Excel file, overwriting the file each time
    with pd.ExcelWriter(attendance_file_path, mode='w') as writer:
        updated_data.to_excel(writer, index=False, header=True)

def start_attendance_logging(student_name, course_name, instructor_name, class_time):
    # Start a separate thread for logging attendance to keep the main UI responsive
    """
    Starts a separate thread for logging attendance to keep the main UI responsive.

    Parameters:
        student_name (str): Name of the student.
        course_name (str): Name of the course.
        instructor_name (str): Name of the instructor.
        class_time (str): Time of the class.
    """
    log_thread = threading.Thread(target=log_attendance, args=(student_name, course_name, instructor_name, class_time))
    log_thread.start()

def recognize_and_log(course_name, instructor_name, class_time):
    
    """
    Recognizes faces in a video stream and logs attendance.

    This function captures a video stream from the default camera, detects faces in the stream, and recognizes the students
    using the pre-trained model. If a student is recognized with a confidence above a certain threshold, they are marked
    as present and their name is displayed on the camera feed. If a student is already marked present, a dialog box is
    shown indicating that they are already marked.

    Parameters:
        course_name (str): Name of the course.
        instructor_name (str): Name of the instructor.
        class_time (str): Time of the class.

    Returns:
        None
    """
    cap = cv2.VideoCapture(0)
    marked_students = set()  # To track students who are already marked

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
         # Define a confidence threshold to consider a prediction valid
        confidence_threshold = 0.6

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            student_name, confidence = predict_image(face_img)

            if confidence >= confidence_threshold:

                label = f"{student_name} ({confidence * 100:.2f}%)"

                if student_name not in marked_students:
                    # Mark the student as present
                    start_attendance_logging(student_name, course_name, instructor_name, class_time)
                    marked_students.add(student_name)
                    messagebox.showinfo("Marked Present",f" Marked {student_name} present at {datetime.now().strftime('%I:%M:%S %p')}")
                else:
                     # Show a dialog box indicating that the student is already marked
                    messagebox.showinfo("Already Marked", f"{student_name} is already marked present.")
                   
            else:
                label = "Unknown"

            # Display the label on the camera feed
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def start_attendance():
    # Get values from input fields
    """
    Starts the attendance logging process.

    Retrieves the values from the input fields for course name, instructor name, and class time.
    Checks that all fields are filled, and if not, shows an error message and returns.
    If all fields are filled, calls recognize_and_log to start the attendance logging process.

    Returns:
        None
    """
    
    course_name = course_entry.get()
    instructor_name = instructor_entry.get()
    class_time = time_entry.get()
    
    # Ensure all fields are filled
    if not course_name or not instructor_name or not class_time:
        messagebox.showwarning("Input Error", "Please fill in all fields before proceeding.")
        return
    
    recognize_and_log(course_name, instructor_name, class_time)

# Set up the GUI window
root = tk.Tk()
root.title("Attendance System")
root.geometry("400x250")
root.configure(bg="lightgray")

# Add labels and input fields
tk.Label(root, text="Course Name:", bg="lightgray").grid(row=0, column=0, padx=10, pady=10, sticky="e")
course_entry = tk.Entry(root, width=30)
course_entry.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Instructor Name:", bg="lightgray").grid(row=1, column=0, padx=10, pady=10, sticky="e")
instructor_entry = tk.Entry(root, width=30)
instructor_entry.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Class Timing (e.g., 9:00-10:15):", bg="lightgray").grid(row=2, column=0, padx=10, pady=10, sticky="e")
time_entry = tk.Entry(root, width=30)
time_entry.grid(row=2, column=1, padx=10, pady=10)

# Add Start Attendance button
start_button = tk.Button(root, text="Start Attendance", command=start_attendance, bg="blue", fg="white", font=("Arial", 12, "bold"))
start_button.grid(row=3, column=0, columnspan=2, pady=20)

root.mainloop()
