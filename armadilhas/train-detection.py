from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="armadilhas/insetos")
trainer.setTrainConfig(object_names_array=["tuta-absoluta", "lagarta", "abelha", "mosca"], batch_size=2, num_experiments=2000)
trainer.trainModel()