#define PY_SSIZE_T_CLEAN
import sys
import pyaudio
from datetime import datetime
import librosa
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QPushButton, QLabel, QHBoxLayout, QListWidget, QLineEdit, QGraphicsDropShadowEffect, QScrollArea, QGridLayout)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt
from database import VoiceCardDB
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_audio()
        self.db = VoiceCardDB()
        self.current_recording_path = ''
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)

    def init_ui(self):
        self.setWindowTitle('言语康复语音助手')
        self.setGeometry(100, 100, 800, 600)

        # 主布局
        main_widget = QWidget()
        main_widget.setObjectName('MainWidget')
        # 加载样式表
        with open('styles.qss', 'r') as f:
            self.setStyleSheet(f.read())
        self.setCentralWidget(main_widget)
        
        # 添加模糊背景效果
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(25)
        main_widget.setGraphicsEffect(self.shadow)
        
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 侧边栏导航
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.addItems(['语音卡库', '语音识别', '系统设置'])
        self.sidebar.setCurrentRow(0)
        layout.addWidget(self.sidebar)

        # 语音卡管理面板
        card_panel = QVBoxLayout()
        
        # 名称输入
        self.card_name_input = QLineEdit(placeholderText='输入语音卡名称')
        card_panel.addWidget(QLabel('语音卡名称：'))
        card_panel.addWidget(self.card_name_input)
        
        # 录音控制
        self.card_record_btn = QPushButton('录制样本', self)
        self.card_record_btn.setObjectName('macOSButton')
        self.card_record_btn.clicked.connect(self.on_card_record_clicked)
        card_panel.addWidget(self.card_record_btn)
        
        # 搜索框
        self.search_field = QLineEdit(placeholderText='搜索语音卡')
        #self.search_field.textChanged.connect(self.filter_cards)
        card_panel.addWidget(self.search_field)

        # 卡片网格
        self.scroll_area = QScrollArea()
        """
        self.card_container = QWidget()
        self.card_layout = QGridLayout(self.card_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.card_container)
        card_panel.addWidget(self.scroll_area)
        """
        # 候选词列表
        self.card_list = QListWidget()
        self.card_list.itemDoubleClicked.connect(self.confirm_delete_card)
        card_panel.addWidget(self.card_list)

        # 刷新按钮
        self.refresh_btn = QPushButton('刷新列表', self)
        self.refresh_btn.clicked.connect(self.load_voice_cards)
        card_panel.addWidget(self.refresh_btn)

        # 参数显示
        self.params_display = QLabel('基频: -共振峰: -')
        card_panel.addWidget(self.params_display)
        
        layout.addLayout(card_panel, 1)

        # 左侧控制面板
        control_panel = QVBoxLayout()
        self.record_btn = QPushButton('开始录音', self)
        self.record_btn.setObjectName('macOSButton')
        self.record_btn.clicked.connect(self.on_record_button_clicked)
        control_panel.addWidget(self.record_btn)

        self.status_label = QLabel('准备就绪')
        control_panel.addWidget(self.status_label)

        # 候选词列表
        self.candidate_list = QListWidget()
        control_panel.addWidget(QLabel('候选词语：'))
        control_panel.addWidget(self.candidate_list)

        layout.addLayout(control_panel, 1)

        # 右侧可视化区域
        viz_panel = QVBoxLayout()
        self.waveform_label = QLabel('实时声波显示')
        viz_panel.addWidget(self.waveform_label)
        layout.addLayout(viz_panel, 1)

    def init_audio(self):
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.is_recording = False
        self.frames = []

    def on_record_button_clicked(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_btn.setText('停止录音')
            self.start_recording()
        else:
            self.record_btn.setText('开始录音')
            self.stop_recognition_recording()

    def on_card_record_clicked(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.card_record_btn.setText('停止录制')
            self.start_card_recording()
        else:
            self.card_record_btn.setText('录制样本')
            self.stop_card_recording()

    def start_recording(self):
        self.status_label.setText('录音中...')
        self.frames = []  # 清空之前的录音数据
        self.stream = self.audio.open(format=self.format,
                                    channels=self.channels,
                                    rate=self.rate,
                                    input=True,
                                    frames_per_buffer=self.chunk,
                                    stream_callback=self.audio_callback)
        self.stream.start_stream()

    def start_card_recording(self):
        self.status_label.setText('正在录制样本...')
        self.frames = []  # 清空之前的录音数据
        self.stream = self.audio.open(format=self.format,
                                    channels=self.channels,
                                    rate=self.rate,
                                    input=True,
                                    frames_per_buffer=self.chunk,
                                    stream_callback=self.audio_callback)
        self.stream.start_stream()

    def stop_recognition_recording(self):
        self.status_label.setText('分析中...')
        self.stream.stop_stream()
        self.stream.close()
        
        audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        self.process_recognition_audio(audio_data)
        
    def stop_card_recording(self):

        self.status_label.setText('保存中...')
        self.stream.stop_stream()
        self.stream.close()
        
        # 保存录音文件
        audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        self.current_recording_path = f'recordings/{self.card_name_input.text()}_{datetime.now().strftime("%Y%m%d%H%M%S")}.wav'
        
        # 提取声学特征
        y = audio_data.astype(np.float32) / 32768.0
        
        # 计算基频
        base_freq = librosa.yin(y, fmin=50, fmax=500)
        base_freq = np.median(base_freq[base_freq > 0])
        
        # 计算共振峰
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.rate)[0]
        formants = spectral_centroids[np.argsort(spectral_centroids)][-3:]
        
        # 显示参数
        self.params_display.setText(f'基频: {base_freq:.1f}Hz\n共振峰: {formants[0]:.1f}, {formants[1]:.1f}, {formants[2]:.1f}Hz')
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=self.rate, n_mfcc=13)
        
        # 保存到数据库
        card_id = self.db.add_voice_card(
            name=self.card_name_input.text(),
            audio_path=self.current_recording_path,
            mfcc_features=mfcc,
            base_freq=float(base_freq),
            formants=formants.tolist()
        )
        self.status_label.setText(f'语音卡保存成功！ID: {card_id}')
        self.card_name_input.clear()

        self.process_recognition_audio(audio_data)
        
    def process_recognition_audio(self, audio_data):
        # 复用特征提取逻辑
        y = audio_data.astype(np.float32) / 32768.0
        mfcc = librosa.feature.mfcc(y=y, sr=self.rate, n_mfcc=13)
        
        # 获取数据库所有语音卡
        candidates = []
        min_db = -1
        best_match = None
        
        # 计算与每个语音卡的DTW距离
        for card in self.db.get_all_voice_cards():
            similarity = self.calculate_similarity(mfcc.T, card['mfcc_features'].T)
            
            if similarity > min_db:
                min_db = similarity
                best_match = card
            candidates.append((card['name'], similarity))
        
        # 更新候选列表
        self.candidate_list.clear()
        for name, score in sorted(candidates, key=lambda x: x[1], reverse=True)[:5]:
            self.candidate_list.addItem(f"{name} ({score:.2f})")
        
        # 记录识别日志
        self.db.log_recognition(mfcc)
        
        # 显示识别结果
        if best_match and min_db < 75:
            self.status_label.setText(f'识别结果: {best_match["name"]}')
        else:
            self.status_label.setText('未找到匹配，请手动选择')

    def process_recognition_audio_2(self, audio_data):
        # 复用特征提取逻辑
        y = audio_data.astype(np.float32) / 32768.0
        mfcc = librosa.feature.mfcc(y=y, sr=self.rate, n_mfcc=13)
        
        # 获取数据库所有语音卡
        candidates = []
        min_db = -1
        best_match = None
        
        # 计算与每个语音卡的DTW距离
        for card in self.db.get_all_voice_cards():
            distance = self.dtw_distance(mfcc.T, card['mfcc_features'].T)
            
            if distance < min_db:
                min_db = distance
                best_match = card
            candidates.append((card['name'], distance))
        
        # 更新候选列表
        self.candidate_list.clear()
        for name, score in sorted(candidates, key=lambda x: x[1])[:5]:
            self.candidate_list.addItem(f"{name} ({score:.2f})")
        
        # 记录识别日志
        self.db.log_recognition(mfcc)
        
        # 显示识别结果
        if best_match and min_db < 75:
            self.status_label.setText(f'识别结果: {best_match["name"]}')
        else:
            self.status_label.setText('未找到匹配，请手动选择')

    def load_voice_cards(self):
        cards = self.db.get_all_voice_cards()
        self.card_list.clear()
        for card in cards:
            self.card_list.addItem(f"{card['id']}: {card['name']}")

    def confirm_delete_card(self):
        from PyQt5.QtWidgets import QMessageBox
        selected = self.card_list.currentItem()
        if not selected:
            return
        reply = QMessageBox.question(self, '确认删除', '确定要删除这个语音卡吗？',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            card_id = selected.text().split(':')[0]
            self.db.delete_voice_card(card_id)
            self.load_voice_cards()

    # --------------------------
    # 步骤2：DTW相似度计算
    # --------------------------
    def calculate_similarity(self, mfcc1, mfcc2):
        """
        使用DTW计算MFCC序列的相似度
        :param mfcc1: 输入语音MFCC
        :param mfcc2: 目标语音卡MFCC
        :return: 相似度得分 (0~100)
        """
        # 计算DTW最优路径和累计距离
        distance, _ = fastdtw(mfcc1, mfcc2, dist=euclidean)
        
        # 归一化处理（可根据具体需求调整）
        max_length = max(len(mfcc1), len(mfcc2))
        normalized_score = 1 / (1 + distance/max_length) * 100
        
        return round(normalized_score, 1)

    def dtw_distance(self, seq1, seq2):
        # 动态时间规整算法实现
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[1:, 0] = np.inf
        dtw_matrix[0, 1:] = np.inf
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.linalg.norm(seq1[i-1]-seq2[j-1])
                dtw_matrix[i,j] = cost + min(dtw_matrix[i-1,j],
                                            dtw_matrix[i,j-1],
                                            dtw_matrix[i-1,j-1])
        return dtw_matrix[n,m]

    def audio_callback(self, in_data, frame_count, time_info, status):
        # 缓存音频数据用于后续处理
        self.frames.append(in_data)
        
        # 实时波形绘制
        audio_array = np.frombuffer(in_data, dtype=np.int16)
        self.update_waveform(audio_array)
        
        return (in_data, pyaudio.paContinue)

    def update_waveform(self, data):
        # 使用QPixmap实现实时波形显示
        pixmap = QPixmap(600, 200)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.blue, 1))
        
        step = max(1, len(data)//600)
        for i in range(0, len(data), step):
            y = 100 + int(data[i] * 0.01)
            painter.drawPoint(i//step, y)
        
        painter.end()
        self.waveform_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())