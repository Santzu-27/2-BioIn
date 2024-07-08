
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("armadilhas/insetos/models/0.36705_414-885.pt")
detector.setJsonPath("armadilhas/insetos/json/insetos_yolov3_detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="armadilhas/teste/lagarta_teste1.jpg", output_image_path="armadilhas/novos_testes_arquivos/epoch-414/lagarta_teste1-detected-name.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


