import numpy as np

from tracked_object import TrackedObject

class ObjectTracker:
    TRACKING_DISTANCE_PERCENTAGE_OF_WIDTH = 0.08
    MOVING_DISTANCE_PERCENTAGE_OF_WIDTH = 0.01
    IN_HAND_DISTANCE_PERCENTAGE_OF_WIDTH = 0.05
    BACK_TO_TRACK_DISTANCE_PERCENTAGE_OF_WIDTH = 0.15

    FALSE_SEEN_FRAMES_PATIENCE = 5
    EXPIRATION_FRAMES_PATIENCE = 20
    PATIENT_COEFFICIENT_NOT_SEEN_IN_HAND = 2
    STABLE_IN_HAND_FRAMES_PATIENCE = 5

    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.tracked_objects = []
        self.expired_objects = []
        self.frame_index = 0
        self.next_object_tracker_id = 0
        self.right_hand_tracked_object = None
        self.left_hand_tracked_object = None

        self.tracking_distance_threshold = int(ObjectTracker.TRACKING_DISTANCE_PERCENTAGE_OF_WIDTH*self.image_width)**2
        self.moving_distance_threshold = int(ObjectTracker.MOVING_DISTANCE_PERCENTAGE_OF_WIDTH*self.image_width)**2
        self.in_hand_distance_threshold = int(ObjectTracker.IN_HAND_DISTANCE_PERCENTAGE_OF_WIDTH*self.image_width)**2
        self.back_to_track_distance_threshold = int(ObjectTracker.BACK_TO_TRACK_DISTANCE_PERCENTAGE_OF_WIDTH*self.image_width)**2
        
    def register_seen_objects(self, seen_yolo_objects, tips_midpoints):
        yolo_objects_to_track = list(seen_yolo_objects)
        already_tracked_object_ids = []
        
        # Visible objects logic
        visible_tracked_objects = []
        visible_tracked_objects_hands_distances = []
        visible_tracked_objects_closest_hands = []
        visible_object_indices = []
        for i, yolo_object in enumerate(yolo_objects_to_track):
            closest_tracked_object, distance = self._get_closest_tracked_object(yolo_object, already_tracked_object_ids)
            if closest_tracked_object is not None:
                if distance < self.tracking_distance_threshold:
                    closest_tracked_object.yolo_object = yolo_object
                    closest_tracked_object.last_seen_frame_index = self.frame_index
                    if closest_tracked_object.frames_persistence < ObjectTracker.FALSE_SEEN_FRAMES_PATIENCE:
                        closest_tracked_object.frames_persistence += 1
                    closest_tracked_object.is_visible = True
                    closest_tracked_object.is_moving = False if distance < self.moving_distance_threshold else True

                    visible_tracked_objects.append(closest_tracked_object)
                    right_hand_distance = self._get_distance_between_object_centers(tips_midpoints[0], closest_tracked_object.yolo_object.get_center())
                    left_hand_distance = self._get_distance_between_object_centers(tips_midpoints[1], closest_tracked_object.yolo_object.get_center())
                    if right_hand_distance <= left_hand_distance:
                        visible_tracked_objects_hands_distances.append(right_hand_distance)
                        visible_tracked_objects_closest_hands.append(0)
                    else:
                        visible_tracked_objects_hands_distances.append(left_hand_distance)
                        visible_tracked_objects_closest_hands.append(1)

                    already_tracked_object_ids.append(closest_tracked_object.tracker_id)
                    visible_object_indices.append(i)

        for i in sorted(visible_object_indices, reverse=True):
            yolo_objects_to_track.pop(i)

        # Hand association
        for i in np.argsort(visible_tracked_objects_hands_distances):
            if visible_tracked_objects_hands_distances[i] < self.in_hand_distance_threshold:
                if visible_tracked_objects_closest_hands[i] == 0:
                    in_hand_visible_tracked_object = visible_tracked_objects[i]
                    if self.right_hand_tracked_object is not None and self.right_hand_tracked_object.tracker_id == in_hand_visible_tracked_object.tracker_id: # Object already in the hand
                        if self.right_hand_tracked_object.in_hand_frames_persistence < ObjectTracker.STABLE_IN_HAND_FRAMES_PATIENCE:
                            self.right_hand_tracked_object.in_hand_frames_persistence += 1
                        break
                    elif self.right_hand_tracked_object is None or self.right_hand_tracked_object.in_hand_frames_persistence < ObjectTracker.STABLE_IN_HAND_FRAMES_PATIENCE: # New object in the hand
                        self.right_hand_tracked_object = in_hand_visible_tracked_object
                        self.right_hand_tracked_object.in_hand_frames_persistence = 1
                        break
            else:
                break

        for i in np.argsort(visible_tracked_objects_hands_distances):
            if visible_tracked_objects_hands_distances[i] < self.in_hand_distance_threshold:
                if visible_tracked_objects_closest_hands[i] == 1:
                    in_hand_visible_tracked_object = visible_tracked_objects[i]
                    if self.left_hand_tracked_object is not None and self.left_hand_tracked_object.tracker_id == in_hand_visible_tracked_object.tracker_id: # Object already in the hand
                        if self.left_hand_tracked_object.in_hand_frames_persistence < ObjectTracker.STABLE_IN_HAND_FRAMES_PATIENCE:
                            self.left_hand_tracked_object.in_hand_frames_persistence += 1
                        break
                    elif self.left_hand_tracked_object is None or self.left_hand_tracked_object.in_hand_frames_persistence < ObjectTracker.STABLE_IN_HAND_FRAMES_PATIENCE: # New object in the hand
                        self.left_hand_tracked_object = in_hand_visible_tracked_object
                        self.left_hand_tracked_object.in_hand_frames_persistence = 1
                        break
            else:
                break

        # Back to track from hand hide logic or create a new tracked object
        for yolo_object in yolo_objects_to_track:
            back_to_track_tracked_object = None
            if self.right_hand_tracked_object is not None \
               and not self.right_hand_tracked_object.is_visible \
               and (not any(tips_midpoints[0]) or self._get_distance_between_object_centers(tips_midpoints[0], yolo_object.get_center()) < self.back_to_track_distance_threshold) \
               and self.right_hand_tracked_object.yolo_object.label_id == yolo_object.label_id:
                back_to_track_tracked_object = self.right_hand_tracked_object
            elif self.left_hand_tracked_object is not None \
               and not self.left_hand_tracked_object.is_visible \
               and (not any(tips_midpoints[1]) or self._get_distance_between_object_centers(tips_midpoints[1], yolo_object.get_center()) < self.back_to_track_distance_threshold) \
               and self.left_hand_tracked_object.yolo_object.label_id == yolo_object.label_id:
                back_to_track_tracked_object = self.left_hand_tracked_object

            if back_to_track_tracked_object is not None:
                back_to_track_tracked_object.yolo_object = yolo_object
                back_to_track_tracked_object.last_seen_frame_index = self.frame_index
                if back_to_track_tracked_object.frames_persistence < ObjectTracker.FALSE_SEEN_FRAMES_PATIENCE:
                    back_to_track_tracked_object.frames_persistence += 1
                back_to_track_tracked_object.is_visible = True
                back_to_track_tracked_object.is_moving = True
                
                already_tracked_object_ids.append(back_to_track_tracked_object.tracker_id)
            else:
                double_detected_object = False
                for tracked_object in self.tracked_objects:
                    if tracked_object.tracker_id in already_tracked_object_ids and tracked_object.yolo_object.label_id == yolo_object.label_id and tracked_object.frames_persistence >= ObjectTracker.FALSE_SEEN_FRAMES_PATIENCE:
                        distance = self._get_distance_between_object_centers(tracked_object.yolo_object.get_center(), yolo_object.get_center())
                        if distance < self.tracking_distance_threshold:
                            double_detected_object = True
                            break
                if not double_detected_object:
                    new_tracked_object = self._register_new_tracked_object(yolo_object)
                    already_tracked_object_ids.append(new_tracked_object.tracker_id)
        
        # Hidden objects logic (possible hidden from hand track logic)
        hidden_objects = []
        hidden_objects_hands_distances = []
        hidden_objects_closest_hands = []
        for tracked_object in self.tracked_objects:
            if tracked_object.tracker_id not in already_tracked_object_ids:
                tracked_object.is_visible = False
                tracked_object.is_moving = True

                hidden_objects.append(tracked_object)
                right_hand_distance = self._get_distance_between_object_centers(tips_midpoints[0], tracked_object.yolo_object.get_center())
                left_hand_distance = self._get_distance_between_object_centers(tips_midpoints[1], tracked_object.yolo_object.get_center())
                if right_hand_distance <= left_hand_distance:
                    hidden_objects_hands_distances.append(right_hand_distance)
                    hidden_objects_closest_hands.append(0)
                else:
                    hidden_objects_hands_distances.append(left_hand_distance)
                    hidden_objects_closest_hands.append(1)

                already_tracked_object_ids.append(tracked_object.tracker_id)

        if self.right_hand_tracked_object is None or self.right_hand_tracked_object.in_hand_frames_persistence < ObjectTracker.STABLE_IN_HAND_FRAMES_PATIENCE:
            for i in np.argsort(hidden_objects_hands_distances):
                if hidden_objects_hands_distances[i] < self.in_hand_distance_threshold:
                    if hidden_objects_closest_hands[i] == 0:
                        self.right_hand_tracked_object = hidden_objects[i]
                        if self.right_hand_tracked_object.in_hand_frames_persistence < ObjectTracker.STABLE_IN_HAND_FRAMES_PATIENCE:
                            self.right_hand_tracked_object.in_hand_frames_persistence += 1
                        break
                else:
                    break
        if self.left_hand_tracked_object is None or self.left_hand_tracked_object.in_hand_frames_persistence < ObjectTracker.STABLE_IN_HAND_FRAMES_PATIENCE:
            for i in np.argsort(hidden_objects_hands_distances):
                if hidden_objects_hands_distances[i] < self.in_hand_distance_threshold:
                    if hidden_objects_closest_hands[i] == 1:
                        self.left_hand_tracked_object = hidden_objects[i]
                        if self.left_hand_tracked_object.in_hand_frames_persistence < ObjectTracker.STABLE_IN_HAND_FRAMES_PATIENCE:
                            self.left_hand_tracked_object.in_hand_frames_persistence += 1
                        break
                else:
                    break

        # Hand release logic
        if any(tips_midpoints[0]) \
           and self.right_hand_tracked_object is not None \
           and self.right_hand_tracked_object.is_visible \
           and self._get_distance_between_object_centers(tips_midpoints[0], self.right_hand_tracked_object.yolo_object.get_center()) >= self.in_hand_distance_threshold:
            self.right_hand_tracked_object.in_hand_frames_persistence = 0
            self.right_hand_tracked_object = None

        if any(tips_midpoints[1]) \
           and self.left_hand_tracked_object is not None \
           and self.left_hand_tracked_object.is_visible \
           and self._get_distance_between_object_centers(tips_midpoints[1], self.left_hand_tracked_object.yolo_object.get_center()) >= self.in_hand_distance_threshold:
            self.left_hand_tracked_object.in_hand_frames_persistence = 0
            self.left_hand_tracked_object = None

    def increment_frame_index(self):
        self._check_false_seen_and_expiration()
        self.frame_index += 1

    def force_object_in_hand(self, hand_index, reference_tracked_object):
        """hand_index: 0 = right, 1 = left"""
        tracked_object = self.get_tracked_object_by_id(reference_tracked_object.tracker_id)
        if tracked_object is None:
            for i in range(len(self.expired_objects)):
                if self.expired_objects[i].tracker_id == reference_tracked_object.tracker_id:
                    tracked_object = self.expired_objects.pop(i)
                    self.tracked_objects.append(tracked_object)
                    break
        if tracked_object is None:
            print("Error: No expired object can be found with specified tracker id.")
            return
        
        tracked_object.last_seen_frame_index = self.frame_index
        tracked_object.frames_persistence = max(tracked_object.frames_persistence, ObjectTracker.FALSE_SEEN_FRAMES_PATIENCE)
        tracked_object.in_hand_frames_persistence = max(tracked_object.in_hand_frames_persistence, ObjectTracker.STABLE_IN_HAND_FRAMES_PATIENCE)
        tracked_object.is_visible = False
        tracked_object.is_moving = True
        if hand_index == 0:
            self.right_hand_tracked_object = tracked_object
        elif hand_index == 1:
            self.left_hand_tracked_object = tracked_object

    def get_tracked_object_by_id(self, tracker_id):
        for tracked_object in self.tracked_objects:
            if tracked_object.tracker_id == tracker_id:
                return tracked_object
        return None
    
    def _get_closest_tracked_object(self, yolo_object, already_tracked_object_ids):
        """Returns the closest tracked object to a specified YoloObject and the respective distance"""
        near_tracked_objects = []
        distances = []
        yolo_object_center = yolo_object.get_center()
        for tracked_object in self.tracked_objects:
            if tracked_object.tracker_id not in already_tracked_object_ids:
                if tracked_object.yolo_object.label_id == yolo_object.label_id:
                    distance = self._get_distance_between_object_centers(yolo_object_center, tracked_object.yolo_object.get_center())
                    near_tracked_objects.append(tracked_object)
                    distances.append(distance)

        if near_tracked_objects:
            closest_object_index = np.argmin(distances)
            return near_tracked_objects[closest_object_index], distances[closest_object_index]
        else:
            return None, None

    def _register_new_tracked_object(self, yolo_object):
        tracked_object = TrackedObject(yolo_object, self.next_object_tracker_id, self.frame_index)
        self.next_object_tracker_id += 1
        self.tracked_objects.append(tracked_object)
        return tracked_object

    def _get_distance_between_object_centers(self, object_center_1, object_center_2):
        return (object_center_1[0] - object_center_2[0])**2 + (object_center_1[1] - object_center_2[1])**2
    
    def _check_false_seen_and_expiration(self):
        # False seen logic
        false_seen_objects_indices = []
        for i, tracked_object in enumerate(self.tracked_objects):
            if not tracked_object.is_visible and tracked_object.frames_persistence < ObjectTracker.FALSE_SEEN_FRAMES_PATIENCE:
                false_seen_objects_indices.append(i)
        for i in sorted(false_seen_objects_indices, reverse=True):
            self.tracked_objects.pop(i)

        # Expiration logic
        expired_object_indices = []
        for i, tracked_object in enumerate(self.tracked_objects):
            if (self.right_hand_tracked_object is not None and self.right_hand_tracked_object.tracker_id == tracked_object.tracker_id) or \
               (self.left_hand_tracked_object is not None and self.left_hand_tracked_object.tracker_id == tracked_object.tracker_id):
                patience_coefficient = ObjectTracker.PATIENT_COEFFICIENT_NOT_SEEN_IN_HAND
            else:
                patience_coefficient = 1
            if tracked_object.last_seen_frame_index < (self.frame_index - ObjectTracker.EXPIRATION_FRAMES_PATIENCE*patience_coefficient):
                expired_object_indices.append(i)
        for i in sorted(expired_object_indices, reverse=True):
            expired_object = self.tracked_objects.pop(i)
            self.expired_objects.append(expired_object)

        tracked_object_tracker_ids = []
        for tracked_object in self.tracked_objects:
            tracked_object_tracker_ids.append(tracked_object.tracker_id)
        if self.right_hand_tracked_object is not None and self.right_hand_tracked_object.tracker_id not in tracked_object_tracker_ids:
            self.right_hand_tracked_object = None
        elif self.left_hand_tracked_object is not None and self.left_hand_tracked_object.tracker_id not in tracked_object_tracker_ids:
            self.left_hand_tracked_object = None
