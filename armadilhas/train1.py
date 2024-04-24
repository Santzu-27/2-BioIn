from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="armadilhas/insetos")
trainer.setTrainConfig(object_names_array=["insetos"], batch_size=6, num_experiments=90, train_from_pretrained_model="/home/oficinas40/2-BioIn/armadilhas/insetos/models/yolov3_insetos_mAP-0.54737_epoch-30.pt")
# In the above,when training for detecting multiple objects,
#set object_names_array=["object1", "object2", "object3",..."objectz"]
trainer.trainModel()
