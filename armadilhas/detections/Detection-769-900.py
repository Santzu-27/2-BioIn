
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("armadilhas/insetos/models/yolov3_insetos_mAP-0.29539_epoch-769.pt")
detector.setJsonPath("armadilhas/insetos/json/insetos_yolov3_detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(
    input_image="armadilhas/novos_testes_arquivos/epoch-769/310.jpg",
    output_image_path="armadilhas/novos_testes_arquivos/epoch-769/001-310.jpg",
    display_box=True,
    extract_detected_objects=False, minimum_percentage_probability=90,
    display_percentage_probability=False, display_object_name=False,
    display_box=True,
    custom_objects=None,
    nms_treshold= 0.09,
    objectness_treshold= 0.008
)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


