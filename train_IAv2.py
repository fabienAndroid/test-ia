from ultralytics import YOLO

model = YOLO('yolov8n.yaml')  # create new model
results = model.train(data="D:/Dataset_IA/dataset_yolov8/data.yaml", epochs = 10,imgsz=[128,128], patience=5, batch=1024, save_period=1)
model.export()

# from ultralytics import YOLO
# model = YOLO('C:/Users/fabie/OneDrive/Documents/fabien/cours/Annee_3/fil_rouge/best.pt')  # load an official model (task=segment,device=0)
# model.export(format='engine',device=0)