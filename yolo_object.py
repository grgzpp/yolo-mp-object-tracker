import numpy as np

class YoloObject:

    def __init__(self, label_id, conf, x1, y1, x2, y2):
        self.label_id = label_id
        self.conf = conf
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    @classmethod
    def from_yolo_box_result(cls, object_result):
        label_id = int(object_result.cls.item())
        conf = object_result.conf.item()
        x1, y1, x2, y2 = map(int, object_result.xyxy[0].tolist())
        return cls(label_id, conf, x1, y1, x2, y2)
    
    @classmethod
    def from_np_array(cls, np_array):
        label_id, x1, y1, x2, y2 = map(int, np_array[0:5])
        conf = np_array[5]
        return cls(label_id, conf, x1, y1, x2, y2)
    
    def get_np_array(self):
        return np.array((self.label_id, self.conf, self.x1, self.x2, self.y1, self.y2))
    
    def get_center(self):
        return ((self.x1 + self.x2)//2, (self.y1 + self.y2)//2)
