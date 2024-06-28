from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="armadilhas/insetos")
trainer.setTrainConfig(object_names_array=["tuta-absoluta", "lagarta"], batch_size=4, num_experiments=850, train_from_pretrained_model="armadilhas/insetos/models/yolov3_insetos_mAP-0.20310_epoch-1.pt")
trainer.trainModel()