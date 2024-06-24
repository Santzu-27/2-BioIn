from imageai.Detection.Custom import DetectionModelTrainer
import torch

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="armadilhas/insetos")
trainer.setTrainConfig(object_names_array=["install"], batch_size=5, num_experiments=850)
trainer.trainModel()