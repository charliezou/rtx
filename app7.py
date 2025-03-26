import cv2
import dlib
import numpy as np
import time

# 初始化dlib的人脸检测器和面部特征点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 需要下载这个模型文件

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 3D模型参考点（基于通用人脸模型）
model_points = np.array([
    (0.0, 0.0, 0.0),             # 鼻尖
    (0.0, -330.0, -65.0),         # 下巴
    (-225.0, 170.0, -135.0),      # 左眼左角
    (225.0, 170.0, -135.0),       # 右眼右角
    (-150.0, -150.0, -125.0),     # 嘴左角
    (150.0, -150.0, -125.0)       # 嘴右角
])

# 相机内参（假设）
focal_length = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
center = [cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2]
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

# 假设没有镜头畸变
dist_coeffs = np.zeros((4, 1))

# 动作检测变量
prev_time = time.time()
prev_pitch = 0
prev_yaw = 0
nod_threshold = 10  # 点头角度阈值
shake_threshold = 10  # 摇头角度阈值
action_detected = None
action_start_time = 0

is_started = False
timecount = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    timecount+=1
    if timecount>300:
        break

    # 转换为灰度图像（dlib需要）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = detector(gray, 0)
    
    for face in faces:
        # 获取面部特征点
        landmarks = predictor(gray, face)
        
        # 获取关键点坐标（使用68点模型中的特定点）
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),     # 鼻尖
            (landmarks.part(8).x, landmarks.part(8).y),       # 下巴
            (landmarks.part(36).x, landmarks.part(36).y),     # 左眼左角
            (landmarks.part(45).x, landmarks.part(45).y),     # 右眼右角
            (landmarks.part(48).x, landmarks.part(48).y),     # 嘴左角
            (landmarks.part(54).x, landmarks.part(54).y)      # 嘴右角
        ], dtype="double")
        
        # 解决PnP问题，获取旋转向量和平移向量
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # 从旋转矩阵中提取欧拉角
        # 注意：OpenCV的坐标系与常规的有所不同
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)  
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        pitch = -pitch+180 if pitch>0  else -pitch-180
        print(f"pitch:{pitch:.1f};Yaw: {yaw:.1f};Roll: {roll:.1f}")


        
        # 计算角度变化
        pitch_change = abs(pitch - prev_pitch)
        yaw_change = abs(yaw - prev_yaw)
        
        current_time = time.time()
        time_elapsed = current_time - prev_time
        prev_time = current_time
        
        # 检测点头动作（主要pitch变化）
        if is_started and pitch_change > nod_threshold and yaw_change < shake_threshold/2:
            if action_detected != "Nodding":
                action_detected = "Nodding"
                action_start_time = current_time
                print("检测到点头动作")
        
        # 检测摇头动作（主要yaw变化）
        elif is_started and yaw_change > shake_threshold and pitch_change < nod_threshold/2:
            if action_detected != "Shaking":
                action_detected = "Shaking"
                action_start_time = current_time
                print("检测到摇头动作")
        
        # 如果超过0.5秒没有动作，重置状态
        if current_time - action_start_time > 0.5:
            action_detected = None
        
        # 更新前一帧的角度
        prev_pitch = pitch
        prev_yaw = yaw
        
        # 在图像上显示结果
        cv2.putText(image, f"Pitch: {pitch:.1f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Yaw: {yaw:.1f}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Roll: {roll:.1f}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        
        
        if action_detected:
            cv2.putText(image, f"Action: {action_detected}", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #print(f"Action: {action_detected}")
        
        # 绘制面部特征点（可选）
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        if not is_started:
            is_started = True
    
    # 显示结果
    #cv2.imshow('Head Pose Estimation', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()