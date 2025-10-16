from ultralytics import YOLO
# 加载训练好的模型，改为自己的路径
model = YOLO('runs/detect/train9/weights/best.pt')  #修改为训练好的路径
source = r'G:\ultralytics-8.3.116\ultralytics-8.3.116\VOCdevkit\val\images\a7.png' #修改为自己的图片路径及文件名
# 运行推理，并附加参数
model.predict(source, save=True)