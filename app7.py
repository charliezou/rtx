import cv2  
import dlib  
import numpy as np

# 初始化dlib的人脸检测器和特征点预测器  
detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3D面部基准点（参考点）  
model_points = np.array([  
    (0.0, 0.0, 0.0),             # 鼻尖  
    (0.0, -330.0, -65.0),        # 下巴  
    (-225.0, 170.0, -135.0),     # 左眼左角  
    (225.0, 170.0, -135.0),      # 右眼右角  
    (-150.0, -150.0, -125.0),    # 左嘴角  
    (150.0, -150.0, -125.0)      # 右嘴角  
])

# 对应的2D特征点索引（基于68点模型）  
index_2d = [30, 8, 36, 45, 48, 54]

# 相机参数（假设）  
focal_length = 1000  
center = (320, 240)  
camera_matrix = np.array(  
    [[focal_length, 0, center[0]],  
     [0, focal_length, center[1]],  
     [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))  # 假设无镜头畸变

# 初始化角度历史记录  
prev_pitch = 0  
prev_yaw = 0  
pitch_threshold = 15  # 角度变化阈值（单位：度）
yaw_threshold = 30  # 角度变化阈值（单位：度）

cap = cv2.VideoCapture(0)

timecount = 0

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
        shape = predictor(gray, face)  
        shape = np.array([[p.x, p.y] for p in shape.parts()])  
          
        # 获取需要的2D特征点  
        image_points = np.array([shape[i] for i in index_2d], dtype=np.float32)  
          
        # 使用solvePnP计算旋转向量和平移向量  
        (success, rotation_vector, translation_vector) = cv2.solvePnP(  
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)  
          
        # 将旋转向量转换为欧拉角  
        rmat, _ = cv2.Rodrigues(rotation_vector)  
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)  
        pitch, yaw, roll = angles[0], angles[1], angles[2]  
          
        # 显示头部姿态  
        cv2.putText(frame, f"angles: {angles}", (10, 30),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
        #cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60),   
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
          
        # 检测点头动作（Pitch角度变化）  
        if abs(pitch - prev_pitch) > pitch_threshold:  
            cv2.putText(frame, "NOD DETECTED!", (10, 120),   
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
          
        # 检测摇头动作（Yaw角度变化）  
        if abs(yaw - prev_yaw) > yaw_threshold:  
            cv2.putText(frame, "SHAKE DETECTED!", (10, 160),   
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
          
        # 更新前值  
        prev_pitch = pitch  
        prev_yaw = yaw  
      
    cv2.imshow("Head Pose Estimation", frame)  
      
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()  
cv2.destroyAllWindows()