# import sys
# sys.path.append("/mnt/ultralytics-main")
import os
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.yaml')
# # # 使用1个GPU训练模型
results = model.train(data='seaship.yaml', epochs=1000, imgsz=1024,device=0)
# model = YOLO('./runs/detect/m-full/weights/best.pt')  # 预训练的 YOLOv8n 模型
# results = model.train(resume=True,device=1)

# # Validate the model
# metrics = model.val(data='seaship.yaml',device=0,val_slice_mode = False)  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category



# # 在图片列表上运行批量推理
# source = 'K:/img/'

# # # Run inference on the source
# model.predict(source, save=True, imgsz=1024, conf=0.5)
# results = model(source, stream=True)  # generator of Results objects
# i = 0
# for result in results:
#     str_n = str(i) + '.pdf'
#     filename = os.path.join("./output/",str_n)
#     result.save(filename=filename)
#     i += 1

# # 导出模型
# model.export(format='onnx')
