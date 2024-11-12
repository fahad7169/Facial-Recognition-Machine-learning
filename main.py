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
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

label_encoder = load_label_encoder()


def log_attendance(student_name, course_name, instructor_name, class_time):
    # Format current date and time
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
    df = pd.DataFrame(data)

    # Check if the file exists and choose the mode
    file_exists = os.path.exists(attendance_file_path)

    # Write attendance data to Excel file without blocking the main thread
    with pd.ExcelWriter(attendance_file_path, mode='a' if file_exists else 'w') as writer:
        df.to_excel(writer, index=False, header=not file_exists)  # Append data if file exists

def start_attendance_logging(student_name, course_name, instructor_name, class_time):
    # Start a separate thread for logging attendance to keep the main UI responsive
    log_thread = threading.Thread(target=log_attendance, args=(student_name, course_name, instructor_name, class_time))
    log_thread.start()

def recognize_and_log(course_name, instructor_name, class_time):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            student_name, confidence = predict_image(face_img)

            # Log attendance if confidence is high
            if confidence > 0.6:  # Adjust threshold as needed
                start_attendance_logging(student_name, course_name, instructor_name, class_time)
                cv2.putText(frame, f"{student_name} ({confidence*100:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"Marked {student_name} present at {datetime.now().strftime('%H:%M:%S')}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def start_attendance():
    # Get values from input fields
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
