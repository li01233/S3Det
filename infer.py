import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')

# 打开视频文件
video_path = "path/to/your/video/file.mp4"
cap = cv2.VideoCapture(video_path)

# 遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在该帧上运行YOLOv8推理
        results = model(frame)

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        # 显示带注释的帧
        cv2.imshow("YOLOv8推理", annotated_frame)

        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()