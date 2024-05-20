class TrackedObject:
    def __init__(self, yolo_object, tracker_id, frame_index):
        self.yolo_object = yolo_object
        self.tracker_id = tracker_id
        self.last_seen_frame_index = frame_index
        self.frames_persistence = 1
        self.in_hand_frames_persistence = 0
        self.is_visible = True
        self.is_moving = False

    def __repr__(self):
        return f"TrackedObject(yolo_object.label_id={self.yolo_object.label_id}, tracker_id={self.tracker_id}, " \
               f"is_visible={self.is_visible}, is_moving={self.is_moving})"
