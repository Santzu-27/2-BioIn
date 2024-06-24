from imageai.Detection.Custom import DetectionModelTrainer
import torch

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="armadilhas/insetos")
trainer.setTrainConfig(object_names_array=["tuta-absoluta", "lagarta"], batch_size=4, num_experiments=600)
trainer.trainModel()