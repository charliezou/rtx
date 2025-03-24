import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# 初始化人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 初始化变量
nod_threshold = 10  # 点头阈值（可根据实际情况调整）
shake_threshold = 10  # 摇头阈值（可根据实际情况调整）
nod_count = 0
shake_count = 0
prev_nose_y = None
prev_nose_x = None

cap_count = 0

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)


    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_np = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)])

        # 提取鼻子和下巴的关键点
        nose = landmarks_np[33]  # 鼻子关键点
        jaw = landmarks_np[8]  # 下巴关键点

        # 计算点头动作
        if prev_nose_y is not None:
            nose_movement = abs(nose[1] - prev_nose_y)
            if nose_movement > nod_threshold:
                nod_count += 1
                print(f"Detected nod")
                break

        # 计算摇头动作
        if prev_nose_x is not None:
            nose_movement_x = abs(nose[0] - prev_nose_x)
            if nose_movement_x > shake_threshold:
                shake_count += 1
                print(f"Detected shake")
                break

        # 更新关键点位置
        prev_nose_y = nose[1]
        prev_nose_x = nose[0]

        print(f"nose:{nose}")

        # 绘制关键点
        cv2.circle(frame, tuple(nose), 2, (0, 255, 0), -1)
        cv2.circle(frame, tuple(jaw), 2, (0, 255, 0), -1)

    # 显示结果
    #cv2.imshow("Frame", frame)

    if nod_count > 0 or shake_count > 0:
        break

cap.release()
cv2.destroyAllWindows()