#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# For calculating the time that the condition has been met
start_time = None
error_displayed_time = None
#triggered = False

# Define the duration threshold (in seconds) that the condition must be met
duration_threshold = 2
error_display_duration = 1


# Capture video from the first camera device
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image and find the pose landmarks
        results = pose.process(image)
        
        # Convert the image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            # Extract the landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Get the coordinates for the shoulders and elbow
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the angle for the left arm
            angle = calculate_angle(shoulder_left, elbow, wrist)
            
            # Display the angle on the video feed
            cv2.putText(image, str(angle), 
                        tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            
            #Check the height of the hands relative to the elbows
            
            if wrist[1] > elbow[1]: # if the hand is lower than elbow
                point_text = "0 point"
            elif angle < 150:
                point_text = "1 point"
            else:
                point_text = "2 point"
 
            # Visualize text
            cv2.putText(image, point_text, 
                        (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
            
            
            
             # Calculate the frame's width to convert from normalized coordinates to pixel values
            frame_width = image.shape[1]
            
            # Calculate the horizontal distance between the shoulders in pixels
            shoulder_diff = abs(shoulder_left[0] - shoulder_right[0]) * frame_width
            
            # Set a threshold for displaying the warning
            threshold = 0.1 * frame_width  # 10% of the frame height
            
            # current time
            current_time = time.time()

            if shoulder_diff > threshold:
                if start_time is None:
                    start_time = current_time
                elif current_time - start_time > duration_threshold:
                    # Condition has been met for over the duration threshold
                    if error_displayed_time is None :
                        error_displayed_time = current_time
            else:
                # If the condition is no longer met, we check if the error has been displayed long enough
                if error_displayed_time is not None:
                    if current_time - error_displayed_time > error_display_duration:
                        # Error message has been displayed long enough, reset timers
                        start_time = None
                        error_displayed_time = None
            
            #Check if we need to display the error message
            if error_displayed_time is not None:
                # Even if the condition is no longer met, display the error
                cv2.putText(image, "Error Watch Front !", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                #If the error message has been displayed for long enough,resset error displayed time
                if current_time - error_displayed_time > error_display_duration:
                    error_displayed_time = None # reset after message is displayed
                
                
            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Display the image
        cv2.imshow('Mediapipe FMA', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:


# In[ ]:




