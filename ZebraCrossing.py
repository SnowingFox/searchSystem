#斑马线闯红灯预警
from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()

camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0
detector.loadModel()
custom_objects = detector.CustomObjects(person=True)
detector = detector.detectObjectsFromVideo(custom_objects=custom_objects,camera_input=camera,
                                output_file_path=os.path.join(execution_path, "camera_detected_video")
                                , frames_per_second=20, log_progress=True, minimum_percentage_probability=30)
for eachObject in detector:
     print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
