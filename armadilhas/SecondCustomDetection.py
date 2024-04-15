from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("insetos/models/yolov3_insetos_mAP-0.51397_epoch-34.pt")
detector.setJsonPath("insetos/json/insetos_yolov3_detection_config.json")
detector.loadModel()
detections, extracted_objects_array = detector.detectObjectsFromImage(input_image="insetos_teste1.jpg", output_image_path="inseto_teste1-detected.jpg", extract_detected_objects=True)

for detection, object_path in zip(detections, extracted_objects_array):
    print(object_path)
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print("---------------")