
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("armadilhas/insetos/models/mAP-0.29539_epoch-769.pt")
detector.setJsonPath("armadilhas/insetos/json/insetos_yolov3_detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="armadilhas/teste/insetos_teste1.jpg", output_image_path="armadilhas/novos_testes_arquivos/epoch-769/insetos_teste1-detected-1.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


