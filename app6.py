import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# 初始化人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 初始化变量
nod_threshold = 6  # 点头阈值（可根据实际情况调整）
shake_threshold = 10  # 摇头阈值（可根据实际情况调整）
nod_count = 0
shake_count = 0
prev_nose = None
prev_jaw = None

timecount = 0

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timecount += 1 
    if timecount > 2000:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)


    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_np = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)])

        # 提取鼻子和下巴的关键点
        nose = landmarks_np[33]  # 鼻子关键点
        jaw = landmarks_np[8]  # 下巴关键点

        # 显示头部姿态  
        cv2.putText(frame, f"nose: {nose}", (10, 30),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
        cv2.putText(frame, f"jaw: {jaw}", (10, 60),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        
        nose_movement_x = 0
        nose_movement_y = 0
        
        if prev_nose is not None:
            nose_movement = abs(nose - prev_nose)
            jaw_movement = abs(jaw - prev_jaw)

            # 计算摇头动作
            if (nose_movement[0] > shake_threshold) & (jaw_movement[0] > shake_threshold) & (nose_movement[1] < nod_threshold*0.5) & (jaw_movement[1] < nod_threshold*0.5):
                shake_count += 1
                cv2.putText(frame, "SHAKE DETECTED!", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 计算点头动作
            if (nose_movement[1] > nod_threshold) & (jaw_movement[1] > nod_threshold) & (nose_movement[0] < shake_threshold*0.5) & (jaw_movement[0] < shake_threshold*0.5):
                nod_count += 1
                cv2.putText(frame, "NOD DETECTED!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 更新关键点位置
        prev_nose = nose
        prev_jaw = jaw

        # 绘制关键点
        cv2.circle(frame, tuple(nose), 2, (0, 255, 0), -1)
        cv2.circle(frame, tuple(jaw), 2, (0, 255, 0), -1)

    # 显示结果
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()