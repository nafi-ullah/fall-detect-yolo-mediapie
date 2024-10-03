import numpy as np
import cv2
import base64
import time
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for fall detection
isFall = False
fallTime = ""
imageLink = ""

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return int(angle) 

counter = 0
counter_two = 0
counter_three = 0
counter_four = 0
stage = None

def process_frame(frame):
    global isFall, fallTime, imageLink
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_hight, image_width, _ = image.shape

        try:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            # ----------------------   DOT   ----------------------           
            
    
            # dot - NOSE
                   
            dot_NOSE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width)
            dot_NOSE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight)
                               
            # dot - LEFT_SHOULDER
                   
            dot_LEFT_SHOULDER_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width)
            dot_LEFT_SHOULDER_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_hight)
            
            # dot - RIGHT_SHOULDER
                   
            dot_RIGHT_SHOULDER_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width)
            dot_RIGHT_SHOULDER_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_hight)
            
            # dot - LEFT_ELBOW
                   
            dot_LEFT_ELBOW_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width)
            dot_LEFT_ELBOW_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_hight)
                        
            # dot - RIGHT_ELBOW
                   
            dot_RIGHT_ELBOW_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width)
            dot_RIGHT_ELBOW_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_hight)
            
            # dot - LEFT_WRIST
                   
            dot_LEFT_WRIST_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width)
            dot_LEFT_WRIST_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_hight)
            
            # dot - RIGHT_WRIST
                   
            dot_RIGHT_WRIST_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width)
            dot_RIGHT_WRIST_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_hight)
            
            
            #2작업
            
            
            # dot - LEFT_HIP
                   
            dot_LEFT_HIP_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width)
            dot_LEFT_HIP_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_hight)
            
            # dot - RIGHT_HIP
                   
            dot_RIGHT_HIP_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width)
            dot_RIGHT_HIP_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_hight)
            
            # dot - LEFT_KNEE
                   
            dot_LEFT_KNEE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width)
            dot_LEFT_KNEE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_hight)
                        
            # dot - RIGHT_KNEE
                   
            dot_RIGHT_KNEE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width)
            dot_RIGHT_KNEE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_hight)
            

            # dot - LEFT_ANKLE
                   
            dot_LEFT_ANKLE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width)
            dot_LEFT_ANKLE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_hight)
                        
            
            # dot - RIGHT_ANKLE
                   
            dot_RIGHT_ANKLE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width)
            dot_RIGHT_ANKLE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_hight)
            
            # dot - LEFT_HEEL
                   
            dot_LEFT_HEEL_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x * image_width)
            dot_LEFT_HEEL_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * image_hight)
           
           
            # dot - RIGHT_HEEL
                   
            dot_RIGHT_HEEL_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x * image_width)
            dot_RIGHT_HEEL_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y * image_hight)
            
                                  
            
            # dot - LEFT_FOOT_INDEX
                   
            dot_LEFT_FOOT_INDEX_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image_width)
            dot_LEFT_FOOT_INDEX_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image_hight)
           
        
            # dot - LRIGHTFOOT_INDEX
                   
            dot_RIGHT_FOOT_INDEX_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * image_width)
            dot_RIGHT_FOOT_INDEX_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * image_hight)
        
            # dot - NOSE
            
                     
            dot_NOSE = [ dot_NOSE_X,dot_NOSE_Y]
            
        
        
        
            # dot - LEFT_ARM_WRIST_ELBOW
            
            dot_LEFT_ARM_A_X = int( (dot_LEFT_WRIST_X+dot_LEFT_ELBOW_X) / 2)
            dot_LEFT_ARM_A_Y = int( (dot_LEFT_WRIST_Y+dot_LEFT_ELBOW_Y) / 2)
            
            LEFT_ARM_WRIST_ELBOW = [dot_LEFT_ARM_A_X,dot_LEFT_ARM_A_Y]
            
            
            # dot - RIGHT_ARM_WRIST_ELBOW
            
            dot_RIGHT_ARM_A_X = int( (dot_RIGHT_WRIST_X+dot_RIGHT_ELBOW_X) / 2)
            dot_RIGHT_ARM_A_Y = int( (dot_RIGHT_WRIST_Y+dot_RIGHT_ELBOW_Y) / 2)
            
            RIGHT_ARM_WRIST_ELBOW = [dot_LEFT_ARM_A_X,dot_LEFT_ARM_A_Y]
            
            
            # dot - LEFT_ARM_SHOULDER_ELBOW
            
            dot_LEFT_ARM_SHOULDER_ELBOW_X = int( (dot_LEFT_SHOULDER_X+dot_LEFT_ELBOW_X) / 2)
            dot_LEFT_ARM_SHOULDER_ELBOW_Y = int( (dot_LEFT_SHOULDER_Y+dot_LEFT_ELBOW_Y) / 2)
            
            LEFT_ARM_SHOULDER_ELBOW = [    dot_LEFT_ARM_SHOULDER_ELBOW_X   ,     dot_LEFT_ARM_SHOULDER_ELBOW_Y     ]
            
           
            # dot - RIGHT_ARM_SHOULDER_ELBOW
            
            dot_RIGHT_ARM_SHOULDER_ELBOW_X = int( (dot_RIGHT_SHOULDER_X+dot_RIGHT_ELBOW_X) / 2)
            dot_RIGHT_ARM_SHOULDER_ELBOW_Y = int( (dot_RIGHT_SHOULDER_Y+dot_RIGHT_ELBOW_Y) / 2)
            
            RIGHT_ARM_SHOULDER_ELBOW = [    dot_RIGHT_ARM_SHOULDER_ELBOW_X   ,     dot_RIGHT_ARM_SHOULDER_ELBOW_Y     ]
            
            
            # dot - BODY_SHOULDER_HIP
            
            dot_BODY_SHOULDER_HIP_X = int( (dot_RIGHT_SHOULDER_X+dot_RIGHT_HIP_X+dot_LEFT_SHOULDER_X+dot_LEFT_HIP_X) / 4)
            dot_BODY_SHOULDER_HIP_Y = int( (dot_RIGHT_SHOULDER_Y+dot_RIGHT_HIP_Y+dot_LEFT_SHOULDER_Y+dot_LEFT_HIP_Y) / 4)
            
            BODY_SHOULDER_HIP = [    dot_BODY_SHOULDER_HIP_X   ,     dot_BODY_SHOULDER_HIP_Y     ]
            
            
            # dot - LEFT_LEG_HIP_KNEE
            
            dot_LEFT_LEG_HIP_KNEE_X = int( (dot_LEFT_HIP_X+dot_LEFT_KNEE_X) / 2)
            dot_LEFT_LEG_HIP_KNEE_Y = int( (dot_LEFT_HIP_Y+dot_LEFT_KNEE_Y) / 2)
            
            LEFT_LEG_HIP_KNEE = [    dot_LEFT_LEG_HIP_KNEE_X   ,     dot_LEFT_LEG_HIP_KNEE_Y     ]
            
            
            # dot - RIGHT_LEG_HIP_KNEE
            
            dot_RIGHT_LEG_HIP_KNEE_X = int( (dot_RIGHT_HIP_X+dot_RIGHT_KNEE_X) / 2)
            dot_RIGHT_LEG_HIP_KNEE_Y = int( (dot_RIGHT_HIP_Y+dot_RIGHT_KNEE_Y) / 2)
            
            RIGHT_LEG_HIP_KNEE = [    dot_RIGHT_LEG_HIP_KNEE_X   ,     dot_RIGHT_LEG_HIP_KNEE_Y     ]
            
            
            # dot - LEFT_LEG_KNEE_ANKLE
            
            dot_LEFT_LEG_KNEE_ANKLE_X = int( (dot_LEFT_ANKLE_X+dot_LEFT_KNEE_X) / 2)
            dot_LEFT_LEG_KNEE_ANKLE_Y = int( (dot_LEFT_ANKLE_Y+dot_LEFT_KNEE_Y) / 2)
            
            LEFT_LEG_KNEE_ANKLE = [   dot_LEFT_LEG_KNEE_ANKLE_X   ,     dot_LEFT_LEG_KNEE_ANKLE_Y     ]

        
            # dot - RIGHT_LEG_KNEE_ANKLE
            
            dot_RIGHT_LEG_KNEE_ANKLE_X = int( (dot_RIGHT_ANKLE_X+dot_RIGHT_KNEE_X) / 2)
            dot_RIGHT_LEG_KNEE_ANKLE_Y = int( (dot_RIGHT_ANKLE_Y+dot_RIGHT_KNEE_Y) / 2)
            
            RIGHT_LEG_KNEE_ANKLE = [   dot_RIGHT_LEG_KNEE_ANKLE_X   ,     dot_RIGHT_LEG_KNEE_ANKLE_Y     ]
            
            
            # dot - LEFT_FOOT_INDEX_HEEL
            
            dot_LEFT_FOOT_INDEX_HEEL_X = int( (dot_LEFT_FOOT_INDEX_X+dot_LEFT_HEEL_X) / 2)
            dot_LEFT_FOOT_INDEX_HEEL_Y = int( (dot_LEFT_FOOT_INDEX_Y+dot_LEFT_HEEL_Y) / 2)
            
            LEFT_FOOT_INDEX_HEEL = [    dot_LEFT_FOOT_INDEX_HEEL_X   ,    dot_LEFT_FOOT_INDEX_HEEL_Y    ]
            
                        
            # dot - RIGHT_FOOT_INDEX_HEEL
            
            dot_RIGHT_FOOT_INDEX_HEEL_X = int( (dot_RIGHT_FOOT_INDEX_X+dot_RIGHT_HEEL_X) / 2)
            dot_RIGHT_FOOT_INDEX_HEEL_Y = int( (dot_RIGHT_FOOT_INDEX_Y+dot_RIGHT_HEEL_Y) / 2)
            
            RIGHT_FOOT_INDEX_HEEL = [    dot_RIGHT_FOOT_INDEX_HEEL_X   ,    dot_RIGHT_FOOT_INDEX_HEEL_Y    ]
            
            
            
            
            
            
            
            # dot _ UPPER_BODY
            
            dot_UPPER_BODY_X = int((dot_NOSE_X+dot_LEFT_ARM_A_X+dot_RIGHT_ARM_A_X+dot_LEFT_ARM_SHOULDER_ELBOW_X+dot_RIGHT_ARM_SHOULDER_ELBOW_X+dot_BODY_SHOULDER_HIP_X)/6)
            dot_UPPER_BODY_Y = int((dot_NOSE_Y+dot_LEFT_ARM_A_Y+dot_RIGHT_ARM_A_Y+dot_LEFT_ARM_SHOULDER_ELBOW_Y+dot_RIGHT_ARM_SHOULDER_ELBOW_Y+dot_BODY_SHOULDER_HIP_Y)/6)
            
            
            UPPER_BODY = [      dot_UPPER_BODY_X    ,     dot_UPPER_BODY_Y      ]
            
                            
            # dot _ LOWER_BODY
            
            dot_LOWER_BODY_X = int( (dot_LEFT_LEG_HIP_KNEE_X+dot_RIGHT_LEG_HIP_KNEE_X+dot_LEFT_LEG_KNEE_ANKLE_X+ dot_RIGHT_LEG_KNEE_ANKLE_X+dot_LEFT_FOOT_INDEX_HEEL_X+dot_RIGHT_FOOT_INDEX_HEEL_X )/6 )
            dot_LOWER_BODY_Y = int( (dot_LEFT_LEG_HIP_KNEE_Y+dot_RIGHT_LEG_HIP_KNEE_Y+dot_LEFT_LEG_KNEE_ANKLE_Y+ dot_RIGHT_LEG_KNEE_ANKLE_Y+dot_LEFT_FOOT_INDEX_HEEL_Y+dot_RIGHT_FOOT_INDEX_HEEL_Y )/6 )
            
            
            LOWER_BODY = [      dot_LOWER_BODY_X    ,     dot_LOWER_BODY_Y      ]
            
            # dot _ BODY
            
            dot_BODY_X = int( (dot_UPPER_BODY_X + dot_LOWER_BODY_X)/2 )
            dot_BODY_Y = int( (dot_UPPER_BODY_Y + dot_LOWER_BODY_Y)/2 )
            
            BODY = [      dot_BODY_X    ,     dot_BODY_Y      ]

           # ---------------------------  COOLDINATE  ---------------------- 
            
            
            
            
            
            # Get coordinates - elbow_l
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Get coordinates - elbow_r
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Get coordinates - shoulder_l
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # Get coordinates - shoulder_r
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Get coordinates - hip_l
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            # Get coordinates - hip_r
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            # Get coordinates - knee_l
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Get coordinates - knee_r
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            
            
                    
            
            
            # Calculate angle - elbow_l
            angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            
            # Calculate angle - elbow_r
            angle_elbow_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            
            # Calculate angle - shoulder_l
            angle_shoulder_l = calculate_angle(elbow_l, shoulder_l, hip_l)
            
            # Calculate angle - shoulder_r
            angle_shoulder_r = calculate_angle(elbow_r, shoulder_r, hip_r)
            
            # Calculate angle - hip_l
            angle_hip_l = calculate_angle(shoulder_l, hip_l, knee_l)
            
            # Calculate angle - hip_r
            angle_hip_r = calculate_angle(shoulder_r, hip_r, knee_r)
            
            # Calculate angle - knee_l
            angle_knee_l = calculate_angle(hip_l, knee_l, ankle_l)
            
            # Calculate angle - knee_r
            angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)
            
            
            
            
            
             #발 사이값
            Point_of_action_LEFT_X = int( 
                ((dot_LEFT_FOOT_INDEX_X +  dot_LEFT_HEEL_X)/2) )
            
            Point_of_action_LEFT_Y = int( 
                ((dot_LEFT_FOOT_INDEX_Y+   dot_LEFT_HEEL_Y)/2) )
            
               
            Point_of_action_RIGHT_X = int( 
                ((dot_RIGHT_FOOT_INDEX_X +  dot_RIGHT_HEEL_X)/2) )
            
            Point_of_action_RIGHT_Y = int( 
                ((dot_RIGHT_FOOT_INDEX_Y+   dot_RIGHT_HEEL_Y)/2) )           
            
                       
            
           #발 사이값 평균
        
            Point_of_action_X = int ( (Point_of_action_LEFT_X +  Point_of_action_RIGHT_X)/2 )
            
            Point_of_action_Y = int ( (Point_of_action_LEFT_Y +  Point_of_action_RIGHT_Y)/2 )
            
            
            #발 사이값 좌표
            Point_of_action = [Point_of_action_X , Point_of_action_Y]
            
           
            #fall case
            fall = int(Point_of_action_X - dot_BODY_X )
            
          
             #--------------------------   여기까지                     
            #case falling and standa
            
            falling = abs(fall) > 50
            standing = abs(fall) < 50
            
            x = Point_of_action_X
            y = -(1.251396648*x) + 618
            

            
            
            
            if falling:
                stage="falling"
          #   if Point_of_action_X <  320 and Point_of_action_Y > 240 and standing and stage == 'falling':     #count3            
          #       cv2.putText(image, 'fall' , ( 320,240 ),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA )
          #       stage = "standing"
          #       counter_three +=1
                isFall = True
                fallTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                imageLink = './outputs/fallImage.png'
                cv2.imwrite(imageLink, image)
                cv2.putText(image, 'Fall Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if Point_of_action_X < 320 and Point_of_action_X > 100 and  Point_of_action_Y > 390 and Point_of_action_Y < 480 and  standing and stage == 'falling':     #count3            
                cv2.putText(image, 'fall' , ( 320,240 ),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA )
                stage = "standing"
                counter_three +=1
                isFall = False
                print(Point_of_action, y)
            if Point_of_action_X >=  320 and Point_of_action_X < 520 and Point_of_action_Y > 380 and Point_of_action_Y < 480 and standing and stage == 'falling':     #count4                
                cv2.putText(image, 'fall' , ( 320,240 ),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA )
                stage = "standing"
                counter_four +=1
                
               
        except:
              pass
        
            #-------------------------------
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2) 
                                 )    
        

        
           
            


@socketio.on('video_frame')
def handle_video_stream(data):
    """Receive base64-encoded frame from frontend, process it, and send back the fall data."""
    global isFall, fallTime, imageLink
    base64_frame = data['frame']
    nparr = np.frombuffer(base64.b64decode(base64_frame), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame for fall detection
    processed_frame = process_frame(frame)

    # Send fall detection data to frontend
    emit('fall_update', {'isFall': isFall, 'time': fallTime, 'imageLink': imageLink})


@socketio.on('get_fall_data')
def handle_fall_data_request():
    """Handle request for the latest fall data."""
    global isFall, fallTime, imageLink
    emit('fall_update', {'isFall': isFall, 'time': fallTime, 'imageLink': imageLink})


if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("Shutting down...")
