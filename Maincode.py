#!/usr/bin/env python
# coding: utf-8

# In[51]:


import cv2
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import mediapipe as mp
import os

# In[52]:


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create tkinter window
window = tk.Tk()
window.title("Gait Analysis")
window.geometry("400x200")


# In[54]:


# Variables to store calculated parameters
step_lengths = []
step_widths = []
stride_lengths = []
hip_joint_angle_extension = []
hip_joint_angle_flexion = []
stance_time = []
swing_time = []


# In[55]:


# Open video file
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if file_path:
        video_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        calculate_parameters(file_path)


# In[56]:


# Calculate gait parameters
def calculate_parameters(file_path):
    global step_lengths, step_widths, stride_lengths, hip_joint_angle_extension, hip_joint_angle_flexion, knee_joint_angle, step_time, stance_time, swing_time

    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
   # print(fps)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for i in range(frame_count):
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get required coordinates
                Left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                
                # Calculate step length
                step_length = math.sqrt((Left_heel[1] - Left_heel[0]) ** 2 + (right_heel[1] - right_heel[0]) ** 2) * 100
                step_lengths.append(step_length)
                
                # Calculate step width
                step_width = math.sqrt((left_foot_index[1] - right_foot_index[1]) ** 2 + (left_foot_index[0] - right_foot_index[0]) ** 2) * 100
                step_widths.append(step_width)
                
                # Calculate stride length
                stride_lengths.append(step_length * 2)
                
                
                # Calculate swing time and stance time
                #print(i)
              
                if i>0:
                    
                    prev_landmarks = prev_results.pose_landmarks.landmark
                    prev_left_toe = [prev_landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                     prev_landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                    prev_right_toe = [prev_landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                      prev_landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                    # Check if current frame is in swing or stance phase
                    is_swing = ((left_foot_index[1] > prev_left_toe[1] and right_foot_index[1] > prev_right_toe[1])
                               or (left_foot_index[1] < prev_left_toe[1] and right_foot_index[1] < prev_right_toe[1]))
                    
                    
                    kunal = is_swing
                    
                    if kunal:
                        #print(1/2)
                        swing_time.append((1/fps)*10)
                       # print(swing_time)
                        
                    else:
                        stance_time.append((1/fps)*10)
                        
                    
                    

                # Calculate hip joint angle extension
                hip_joint_angle_ext = math.degrees(math.acos(
                    (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x - landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x) /
                    (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y - landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)))
                hip_joint_angle_extension.append(hip_joint_angle_ext)

                # Calculate hip joint angle flexion
                hip_joint_angle_flex = math.degrees(math.acos(
                    (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x - landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x) /
                    (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y - landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)))
                hip_joint_angle_flexion.append(hip_joint_angle_flex)

               
                prev_results = results

                
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('Gait Analysis', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Calculate mean values of parameters
        step_length_mean = np.mean(step_lengths)
        step_width_mean = np.mean(step_widths)
        stride_length_mean = step_length_mean * 2

        if step_length_mean>44 and step_length_mean <70 :
            messagebox.showinfo("Gait Analysis",
                            f"You are normal!")
        else:
            messagebox.showinfo("Gait Analysis",
                            f"You are not normal!")
            
        # Calculate total swing time and stance time
        total_swing_time = sum(swing_time)
        total_stance_time = sum(stance_time)
        
        # Display parameter values
        messagebox.showinfo("Gait Analysis",
                            f"Step Length: {step_length_mean:.2f} cm\nStep Width: {step_width_mean:.2f} cm\nStride Length: {stride_length_mean:.2f} cm")


        # Save parameters to an Excel file
        username = username_entry.get()
        data = {'Step Length': step_lengths,
                'Step Width': step_widths,
                'Stride Length': stride_lengths}
        df = pd.DataFrame(data)
        df.to_excel(f"{username}.xlsx", index=False)
        
        # save angle values
        
        dataAngles = {'Hip joint angle extension': hip_joint_angle_extension,
                'Hip joint angle flexion': hip_joint_angle_flexion}
        dfA = pd.DataFrame(dataAngles)
        dfA.to_excel(f"{username}_Angles.xlsx", index=False)

        
        # Save video skeleton
        cap_skeleton = cv2.VideoCapture(file_path)
        skeleton_output = cv2.VideoWriter(f"{username}_Skeleton.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (image.shape[1], image.shape[0]))
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for i in range(frame_count):
                ret, frame = cap_skeleton.read()
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                skeleton_output.write(image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
        cap_skeleton.release()
        skeleton_output.release()
        cv2.destroyAllWindows()
        
        
        # Reset parameters
        step_lengths = []
        step_widths = []
        stride_lengths = []
        hip_joint_angle_extension = []
        hip_joint_angle_flexion = []
        swing_time = []
        stance_time = []

        messagebox.showinfo("Gait Analysis", "Your Skeleton Video is Saved!")
    
    cap.release()
    cv2.destroyAllWindows()



# In[57]:


# Username entry
username_label = tk.Label(window, text="Enter Your Name:")
username_label.pack()
username_entry = tk.Entry(window)
username_entry.pack()
# Open file button
open_file_button = tk.Button(window, text="Open Video File", command=open_file)
open_file_button.pack()

window.mainloop()


# In[ ]:




