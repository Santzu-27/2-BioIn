
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("armadilhas/insetos/models/yolov3_classes-separadas_52p_ep-16.pt")
detector.setJsonPath("armadilhas/insetos/json/insetos_yolov3_detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="armadilhas/teste/tuta_teste2.jpg", output_image_path="armadilhas/novos_testes_arquivos/classes-separadas/lagarta_teste4.jpg-detected-1.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


