## Overview

<p align="center">
  <img src="./videos/object_tracker.gif" alt="animated" />
</p>

This object tracker provides functionality to track objects detected by the [YOLO](https://docs.ultralytics.com) object detection system, combined with information provided by the [MediaPipe hand tracker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker). The main focus of the tracker is objects held in the hand, with specific logic to manage their state, even if they are hidden by the hand. The tracker maintains a list of active and expired objects, and updates their state based on visibility, movement, and whether they are held in the hand.

The tracker is inspired by [BoT-SORT](https://github.com/NirAharon/BoT-SORT) and [ByteTrack](https://github.com/ifzhang/ByteTrack), which are directly usable with YOLO with the [Ultralytics](https://docs.ultralytics.com/modes/track/) library. The structure is simplified compared to these. The distinguishing feature is the implementation of hand tracking with MediaPipe and all the hand-held object tracking logic.

## Features

- Tracks objects detected by YOLO, associating each of them with an id.
- Associates objects with detected hand positions using MediaPipe to determine which object is being held in the hand.
- Maintains persistence of tracked objects across frames.
- Manages the state of objects being held or moved by hands.
- Handles the re-identification of objects when they reappear after being hidden by hand.

## Installation

To use the `ObjectTracker`, you need to have the following dependencies installed:

- `numpy`
- YOLO object detection implementation
- MediaPipe hand tracking implementation

You can install `numpy` using pip:

```bash
pip install numpy
```

## Usage

### Initialization

Create an instance of `ObjectTracker` by providing the width and height of the image frames:

```python
from object_tracker import ObjectTracker

image_width = 640
image_height = 480
object_tracker = ObjectTracker(image_width, image_height)
```

### Register Seen Objects

Update the tracker with objects detected by YOLO and the hand midpoints detected by MediaPipe:

```python
# seen_yolo_objects: List of YOLO detected objects
# tips_midpoints: List of tuples with (x, y) coordinates of hand tip midpoints in pixels (use the HandHelper class to get it)
object_tracker.register_seen_objects(seen_yolo_objects, tips_midpoints)
```

### Increment Frame Index

Increment the frame index to keep track of the frame count and handle expiration and false seen logic:

```python
object_tracker.increment_frame_index()
```

### Get Tracked Object by ID

Retrieve a tracked object by its unique tracker ID:

```python
tracker_id = 0
tracked_object = object_tracker.get_tracked_object_by_id(tracker_id)
```

### Get Objects Held in Hands

Retrieve objects held in hands (None is returned if no object is detected in the hand):

```python
right_hand_tracked_object = object_tracker.right_hand_tracked_object
left_hand_tracked_object = object_tracker.left_hand_tracked_object
```

## Class Reference

### ObjectTracker

- `MOVING_DISTANCE_PERCENTAGE_OF_WIDTH`: Distance threshold for detecting movement, as a percentage of image width.
- `IN_HAND_DISTANCE_PERCENTAGE_OF_WIDTH`: Distance threshold for detecting objects in hand, as a percentage of image width.
- `IN_HAND_HIDDEN_DISTANCE_PERCENTAGE_OF_WIDTH`: Distance threshold for detecting hidden objects in hand, as a percentage of image width.
- `FALSE_SEEN_FRAMES_PATIENCE`: Number of frames to persist objects seen falsely.
- `EXPIRATION_FRAMES_PATIENCE`: Number of frames after which an unseen object expires.
- `PATIENT_COEFFICIENT_NOT_SEEN_IN_HAND`: Coefficient for patience when objects are not seen but are supposed in hand.
- `STABLE_IN_HAND_FRAMES_PATIENCE`: Number of frames to confirm stable objects in hand.


### TrackedObject

- `yolo_object`: The YOLO object associated with this tracked object.
- `tracker_id`: Unique identifier for the tracked object.
- `last_seen_frame_index`: Frame index when the object was last seen.
- `frames_persistence`: Number of frames the object has been persistently seen.
- `in_hand_frames_persistence`: Number of frames the object has been persistently seen in hand.
- `is_visible`: Boolean indicating if the object is currently visible.
- `is_moving`: Boolean indicating if the object is currently moving.

### YoloObject

- `label_id`: The class label ID of the detected object (defined in YOLO training).
- `conf`: The confidence score of the detected object.
- `x1, y1, x1, y2`: The coordinates of the bounding box corners.

### HandHelper

- `right_hand_landmarks`: List of landmarks for the right hand.
- `left_hand_landmarks`: List of landmarks for the left hand.
#####
- `register_hands_landmarks(self, right_hand_landmarks, left_hand_landmarks)`: Registers the landmarks for both hands.
- `get_tips_midpoints(self)`: Returns the midpoints of the thumb and index finger tips for both hands.

## Example

Here is an example demonstrating the usage of the `ObjectTracker` class within the other utility classes. The YOLO and MediaPipe implementations are not reported.

```python
from object_tracker import ObjectTracker
from hand_helper import HandHelper
from yolo_object import YoloObject

# Example image dimensions
image_width = 640
image_height = 480

# Create instances of ObjectTracker and HandHelper
object_tracker = ObjectTracker(image_width, image_height)
hand_helper = HandHelper(image_width, image_height)

# Example of camera loop
while True:
    # object_results = ...
    # mp_results = ...

	# Example YOLO detected objects
	seen_yolo_objects = []
    for object_result in objects_results.boxes:
        if object_result.conf.item() > 0.6:
            seen_yolo_objects.append(YoloObject.from_yolo_box_result(object_result))
    # Example hand landmarks
    right_hand_landmarks, left_hand_landmarks = extract_hand_landmarks(mp_results)

	# Register hand landmarks
	hand_helper.register_hands_landmarks(right_hand_landmarks, left_hand_landmarks)
	# Get tips midpoints from HandHelper
	tips_midpoints = hand_helper.get_tips_midpoints()

	# Register seen objects with the tracker
	object_tracker.register_seen_objects(seen_yolo_objects, tips_midpoints)
	# Increment the frame index
	object_tracker.increment_frame_index()

	# Get a tracked object by ID
	tracker_id = 0
	tracked_object = object_tracker.get_tracked_object_by_id(tracker_id)
	
	# Get objects held in hands
	right_hand_tracked_object = object_tracker.right_hand_tracked_object
	left_hand_tracked_object = object_tracker.left_hand_tracked_object

	# Implement your logic with tracked objects
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
