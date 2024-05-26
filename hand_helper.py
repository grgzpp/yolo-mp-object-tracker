class HandHelper:

    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.right_hand_landmarks = []
        self.left_hand_landmarks = []

    def register_hands_landmarks(self, right_hand_landmarks, left_hand_landmarks):
        self.right_hand_landmarks = right_hand_landmarks
        self.left_hand_landmarks = left_hand_landmarks

    def get_tips_midpoints(self):
        tips_midpoints = []
        for i in range(2):
            if i == 0:
                hand_landmarks = self.right_hand_landmarks
            else:
                hand_landmarks = self.left_hand_landmarks
            
            thumb_tip_landmark = hand_landmarks[4]
            thumb_tip = (thumb_tip_landmark.x, thumb_tip_landmark.y, thumb_tip_landmark.z)
            index_finder_tip_landmark = hand_landmarks[8]
            index_finder_tip = (index_finder_tip_landmark.x, index_finder_tip_landmark.y, index_finder_tip_landmark.z)

            tips_midpoint = (int((thumb_tip[0] + index_finder_tip[0])/2*self.image_width), 
                             int((thumb_tip[1] + index_finder_tip[1])/2*self.image_height))
            tips_midpoints.append(tips_midpoint)
        
        return tips_midpoints
    
    def get_hand_centers(self):
        hand_centers = []
        for i in range(2):
            if i == 0:
                hand_landmarks = self.right_hand_landmarks
            else:
                hand_landmarks = self.left_hand_landmarks
            
            wrist_landmark = hand_landmarks[0]
            wrist = (wrist_landmark.x, wrist_landmark.y, wrist_landmark.z)
            middle_finger_mcp_landmark = hand_landmarks[9]
            middle_finger_mcp = (middle_finger_mcp_landmark.x, middle_finger_mcp_landmark.y, middle_finger_mcp_landmark.z)

            hand_center = (int((wrist[0] + middle_finger_mcp[0])/2*self.image_width), 
                           int((wrist[1] + middle_finger_mcp[1])/2*self.image_height))
            hand_centers.append(hand_center)
        
        return hand_centers
