
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("insetos/models/yolov3_insetos_mAP-0.51397_epoch-34.pt")
detector.setJsonPath("insetos/json/insetos_yolov3_detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="insetos_teste2.jpg", output_image_path="insetos_teste2-detected-new2.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


