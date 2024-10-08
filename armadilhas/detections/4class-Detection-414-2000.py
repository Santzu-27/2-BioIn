
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("armadilhas/insetos/models/0.36705_epoch-414.pt")
detector.setJsonPath("armadilhas/insetos/json/insetos_yolov3_detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="armadilhas/teste/instetos_teste1.jpg", output_image_path="armadilhas/novos_testes_arquivos/epoch-414/2-insetos_Teste.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


