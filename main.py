import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import numpy as np
from collections import deque

class GestureController:
    def __init__(self):
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Frame settings
        self.frame_width = 640
        self.frame_height = 480
        
        # Performance settings
        self.process_fps = 15  # Process frames at this rate
        self.frame_interval = 1.0 / self.process_fps
        
        # Gesture detection settings
        self.debounce_time = 0.3  # Reduced debounce time for more responsive controls
        self.last_gesture_time = time.time()
        self.gesture_history = deque(maxlen=10)  # Store recent gestures for smoothing
        
        # Movement thresholds
        self.swipe_threshold = 40  # Pixels movement required to detect swipe
        self.vertical_swipe_threshold = 35  # Threshold for vertical swipes
        
        # Hand tracking state
        self.prev_hand_position = None
        self.prev_gesture = None
        self.motion_tracker = deque(maxlen=5)  # Track recent hand movements
        
        # Initialize gestures dict (gesture name -> function)
        self.gestures = {
            "OPEN_PALM": self._handle_open_palm,
            "FIST": self._handle_fist,
            "SWIPE_LEFT": self._handle_swipe_left,
            "SWIPE_RIGHT": self._handle_swipe_right,
            "SWIPE_UP": self._handle_swipe_up,
            "SWIPE_DOWN": self._handle_swipe_down,
            "VICTORY": self._handle_victory,
            "THUMBS_UP": self._handle_thumbs_up,
            "POINT": self._handle_point
        }
        
        # Current recognized gesture and confidence
        self.current_gesture = "No Gesture"
        self.gesture_confidence = 0.0
        
        # Debug mode flag
        self.debug_mode = True
        
    def start(self):
        """Initialize and start the webcam capture and processing"""
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Start webcam capture thread
        self.webcam_thread = WebcamCaptureThread(self.cap, self.frame_width, self.frame_height)
        self.webcam_thread.daemon = True  # Thread will close when main program exits
        self.webcam_thread.start()
        
        # Initialize the MediaPipe hand tracking solution
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,  # 0=Lite, 1=Full
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Start the main processing loop
        self._process_frames()
        
    def _process_frames(self):
        """Main processing loop for gesture recognition"""
        last_frame_time = time.time()
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        
        try:
            while True:
                # Control frame rate processing
                current_time = time.time()
                if current_time - last_frame_time < self.frame_interval:
                    continue
                
                # FPS calculation
                fps_counter += 1
                if current_time - fps_timer > 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_timer = current_time
                
                last_frame_time = current_time
                
                # Get the latest frame from webcam thread
                if self.webcam_thread.frame is None:
                    continue
                    
                frame = self.webcam_thread.frame.copy()
                frame = cv2.flip(frame, 1)  # Mirror image for intuitive feedback
                
                # Process frame for hand detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Process hand landmarks if detected
                if results.multi_hand_landmarks:
                    self._process_hand_landmarks(frame, results.multi_hand_landmarks[0])
                else:
                    self.current_gesture = "No Hand Detected"
                    self.prev_hand_position = None
                
                # Display information on frame
                self._display_info(frame, current_fps)
                
                # Show the frame
                cv2.imshow("Advanced Gesture Controller", frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Clean up resources
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            
    def _process_hand_landmarks(self, frame, hand_landmarks):
        """Process the detected hand landmarks for gesture recognition"""
        # Draw hand landmarks on frame
        self.mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Extract landmark positions
        landmarks = []
        for id, lm in enumerate(hand_landmarks.landmark):
            cx, cy, cz = int(lm.x * self.frame_width), int(lm.y * self.frame_height), lm.z
            landmarks.append((id, cx, cy, cz))
            
            # Draw key points with different colors for important landmarks
            if id in [4, 8, 12, 16, 20]:  # Fingertips
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            elif id in [0, 5, 9, 13, 17]:  # Base of fingers
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # Detect current gesture
        if len(landmarks) == 21:  # Full hand detected
            finger_states = self._get_finger_states(landmarks)
            
            # Get palm center for movement tracking
            palm_center = self._get_palm_center(landmarks)
            
            # Detect basic gestures
            gesture = self._recognize_gesture(finger_states, landmarks)
            
            # Track hand position for movement/swipe detection
            if self.prev_hand_position is not None:
                dx = palm_center[0] - self.prev_hand_position[0]
                dy = palm_center[1] - self.prev_hand_position[1]
                
                # Track motion
                self.motion_tracker.append((dx, dy))
                
                # Detect swipes if enough time has passed since last gesture
                if time.time() - self.last_gesture_time > self.debounce_time:
                    if abs(dx) > self.swipe_threshold and abs(dx) > abs(dy):
                        if dx > 0:
                            gesture = "SWIPE_RIGHT"
                        else:
                            gesture = "SWIPE_LEFT"
                    elif abs(dy) > self.vertical_swipe_threshold and abs(dy) > abs(dx):
                        if dy > 0:
                            gesture = "SWIPE_DOWN"
                        else:
                            gesture = "SWIPE_UP"
            
            # Store current palm position for next comparison
            self.prev_hand_position = palm_center
            
            # Execute gesture if different from previous or enough time has passed
            if (gesture != self.prev_gesture or 
                time.time() - self.last_gesture_time > self.debounce_time):
                
                if gesture in self.gestures and gesture != "No Gesture":
                    self.gestures[gesture]()
                    self.last_gesture_time = time.time()
                
                self.current_gesture = gesture
                self.prev_gesture = gesture
                self.gesture_history.append(gesture)
    
    def _get_finger_states(self, landmarks):
        """Determine which fingers are extended"""
        fingers = []
        
        # Special case for thumb
        if landmarks[4][1] < landmarks[3][1]:  # Thumb is extended if tip is left of knuckle (in flipped view)
            fingers.append(1)
        else:
            fingers.append(0)
            
        # For four fingers
        tips = [8, 12, 16, 20]  # Fingertips
        mcp = [5, 9, 13, 17]    # Metacarpophalangeal joints
        pip = [6, 10, 14, 18]   # Proximal interphalangeal joints
        
        for tip, base in zip(tips, pip):
            # Finger is extended if tip is higher than base (lower y value)
            if landmarks[tip][2] < landmarks[base][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
        
    def _recognize_gesture(self, finger_states, landmarks):
        """Recognize the hand gesture based on finger states"""
        # Extract individual finger states
        thumb, index, middle, ring, pinky = finger_states
        
        # Basic gesture recognition
        if all(finger_states):  # All fingers extended
            return "OPEN_PALM"
        elif not any(finger_states):  # All fingers closed
            return "FIST"
        elif index == 1 and middle == 1 and thumb == 0 and ring == 0 and pinky == 0:
            return "VICTORY"
        elif index == 1 and all(f == 0 for f in [thumb, middle, ring, pinky]):
            return "POINT"
        elif thumb == 1 and all(f == 0 for f in [index, middle, ring, pinky]):
            return "THUMBS_UP"
        
        return "No Gesture"
        
    def _get_palm_center(self, landmarks):
        """Calculate the center of the palm from landmarks"""
        # Use wrist and base of middle finger to estimate palm center
        wrist = landmarks[0][1:3]
        middle_mcp = landmarks[9][1:3]
        
        palm_center = ((wrist[0] + middle_mcp[0]) // 2, 
                       (wrist[1] + middle_mcp[1]) // 2)
        return palm_center
        
    def _display_info(self, frame, fps):
        """Display debug information on frame"""
        # Display FPS
        cv2.putText(frame, f'FPS: {fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                   
        # Display current gesture
        cv2.putText(frame, f'Gesture: {self.current_gesture}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                   
        # Add instruction box
        if self.debug_mode:
            instructions = [
                "OPEN_PALM: Jump (Space)",
                "FIST: Duck (Down)",
                "SWIPE_LEFT/RIGHT: Move Left/Right",
                "POINT: Click",
                "VICTORY: Double-click",
                "THUMBS_UP: Enter",
                "Press 'Q' to quit"
            ]
            
            # Draw background for instructions
            cv2.rectangle(frame, (10, 100), (300, 270), (0, 0, 0, 0.7), -1)
            
            # Add each instruction line
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (15, 130 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Gesture handlers
    def _handle_open_palm(self):
        pyautogui.press('space')
        
    def _handle_fist(self):
        pyautogui.press('down')
        
    def _handle_swipe_left(self):
        pyautogui.press('left')
        
    def _handle_swipe_right(self):
        pyautogui.press('right')
        
    def _handle_swipe_up(self):
        pyautogui.press('up')
        
    def _handle_swipe_down(self):
        pyautogui.press('down')
        
    def _handle_point(self):
        pyautogui.click()
        
    def _handle_victory(self):
        pyautogui.doubleClick()
        
    def _handle_thumbs_up(self):
        pyautogui.press('enter')


class WebcamCaptureThread(threading.Thread):
    def __init__(self, cap, frame_width, frame_height):
        threading.Thread.__init__(self)
        self.cap = cap
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame = None
        self.running = True
        
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Resize frame to improve performance
            self.frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # Let the thread sleep a tiny bit to reduce CPU usage
            time.sleep(0.01)
            
    def stop(self):
        self.running = False


if __name__ == "__main__":
    # Create and start the gesture controller
    controller = GestureController()
    controller.start()