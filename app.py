#define PY_SSIZE_T_CLEAN
import sys
import pyaudio
from datetime import datetime
import librosa
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QPushButton, QLabel, QHBoxLayout, QListWidget, QLineEdit, QGraphicsDropShadowEffect, QScrollArea, QGridLayout, QStackedLayout)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt
from database import VoiceCardDB
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
#import wave
import noisereduce as nr
from scipy.spatial.distance import euclidean
import soundfile as sf
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplWaveformWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 2), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#F0F0F0')
        self.line, = self.ax.plot([], [], 'b-', lw=1)
        self.ax.set_ylim(-32768, 32768)
        self.ax.set_xlim(0, 16000)  # 默认显示1秒音频
        self.ax.get_xaxis().set_visible(False)
        self.fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_audio()
        self.db = VoiceCardDB()
        self.current_recording_path = ''
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.load_voice_cards()
        self.update_waveform()

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
        self.sidebar.addItems(['语音卡库', '语音识别'])
        self.sidebar.setCurrentRow(0)
        layout.addWidget(self.sidebar)

        # 主内容区域改为堆叠布局
        self.stacked_layout = QStackedLayout()
        
        # 语音卡管理面板（从原布局中提取出来）
        card_panel_widget = QWidget()

        # 语音卡管理面板
        card_panel = QVBoxLayout(card_panel_widget)
        
        # 名称输入
        self.card_name_input = QLineEdit(placeholderText='输入语音卡名称')
        #card_panel.addWidget(QLabel('语音卡名称：'))
        card_panel.addWidget(self.card_name_input)
        
        # 录音控制
        self.card_record_btn = QPushButton('录制样本', self)
        self.card_record_btn.setObjectName('macOSButton')
        self.card_record_btn.clicked.connect(self.on_card_record_clicked)
        card_panel.addWidget(self.card_record_btn)

        self.card_status_label = QLabel('准备就绪')
        card_panel.addWidget(self.card_status_label)
        
        # 卡片网格
        self.scroll_area = QScrollArea()
        
        # 候选词列表
        self.card_list = QListWidget()
        self.card_list.setStyleSheet("""
            QListWidget::item {
                padding: 2px;
                margin: 1px;
            }
        """)
        self.card_list.itemClicked.connect(self.show_card_details)  # 新增点击事件
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

        self.stacked_layout.addWidget(card_panel_widget)  # 索引0
        #layout.addLayout(self.stacked_layout, 1)

        # 语音识别面板（从原布局中提取出来）
        control_panel_widget = QWidget()
        control_panel = QVBoxLayout(control_panel_widget)
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
        self.stacked_layout.addWidget(control_panel_widget)  # 索引1

        # 主布局结构调整
        main_content = QVBoxLayout()
        main_content.addLayout(self.stacked_layout, 1)  # 堆叠面板在上方
        
        # 右侧可视化区域
        viz_panel = QVBoxLayout()
        
        self.waveform_canvas = MplWaveformWidget()
        viz_panel.addWidget(self.waveform_canvas)

        #self.waveform_label = QLabel('实时声波显示')
        #viz_panel.addWidget(self.waveform_label)
        main_content.addLayout(viz_panel)  # 可视化面板在底部
        layout.addLayout(main_content, 1)

        self.sidebar.currentRowChanged.connect(self.show_panel)
        

    def show_panel(self, index):
        """切换显示对应的面板"""
        # 索引0:语音卡库，索引1:语音识别
        self.stacked_layout.setCurrentIndex(index if index in [0,1] else 0)

    def init_audio(self):
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
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
        self.card_status_label.setText('正在录制样本...')
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
        #self.update_waveform(audio_data)  # 传入完整音频数据
        
    def stop_card_recording(self):

        self.card_status_label.setText('保存中...')
        self.stream.stop_stream()
        self.stream.close()
        
        # 保存录音文件
        audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        self.current_recording_path = f'recordings/{self.card_name_input.text()}.wav'

        sf.write(self.current_recording_path, audio_data, self.rate)

        """
        # 使用wave模块保存为WAV文件
        wf = wave.open(self.current_recording_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
       """
        # 提取声学特征
        
        y = audio_data.astype(np.float32)
        
        # 计算基频
        base_freq = librosa.yin(y, fmin=50, fmax=500)
        base_freq = np.median(base_freq[base_freq > 0])
        
        # 计算共振峰
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.rate)[0]
        formants = spectral_centroids[np.argsort(spectral_centroids)][-3:]

        # 提取MFCC特征
        y_clean, mfcc, non_silent = self.get_audio_mfcc(y)
        mfcc_shape = mfcc.shape if mfcc is not None else "无数据"

        # 显示参数
        self.params_display.setText(f'基频: {base_freq:.1f}Hz\n共振峰: {formants[0]:.1f}, {formants[1]:.1f}, {formants[2]:.1f}Hz\nMFCC维度: {mfcc_shape}')     
        
        # 保存到数据库
        card_id = self.db.add_voice_card(
            name=self.card_name_input.text(),
            audio_path=self.current_recording_path,
            mfcc_features=mfcc,
            base_freq=float(base_freq),
            formants=formants.tolist()
        )
        self.card_status_label.setText(f'语音卡保存成功！ID: {card_id}')
        self.card_name_input.clear()
        self.load_voice_cards()

        self.update_waveform(y,non_silent = non_silent)  # 传入完整音频数据

        
    def process_recognition_audio(self, audio_data):
        # 复用特征提取逻辑
        y = audio_data.astype(np.float32)
        y_clean, mfcc, non_silent = self.get_audio_mfcc(y)

        self.update_waveform(y,non_silent = non_silent)  # 传入完整音频数据
        
        # 获取数据库所有语音卡
        candidates = []
        msx_sim = 0
        best_match = None
        
        # 计算与每个语音卡的DTW距离
        for card in self.db.get_all_voice_cards():
            similarity, distance = self.calculate_similarity(mfcc.T, card['mfcc_features'].T)
            
            if similarity > msx_sim:
                msx_sim = similarity
                best_match = card
            candidates.append((card['name'], similarity, distance))
        
        # 更新候选列表
        self.candidate_list.clear()
        for name, score, distance in sorted(candidates, key=lambda x: x[1], reverse=True)[:5]:
            self.candidate_list.addItem(f"{name} ({score:.2f}) {distance:.1f}")
        
        # 记录识别日志
        self.db.log_recognition(mfcc)
        
        # 显示识别结果
        if best_match and msx_sim > 0.75:
            self.status_label.setText(f'识别结果: {best_match["name"]}')
        else:
            self.status_label.setText('未找到匹配，请手动选择')

    
    def load_voice_cards(self):
        cards = self.db.get_all_voice_cards()
        self.card_list.clear()
        for card in cards:
            self.card_list.addItem(f"{card['id']}: {card['name']}")
    
    def show_card_details(self, item):
        """显示选中语音卡的详细参数"""
        card_id = item.text().split(':')[0]
        card_data = self.db.get_voice_card(card_id)
        
        if card_data:
            mfcc_shape = card_data['mfcc_features'].shape if card_data['mfcc_features'] is not None else "无数据"
            formants = card_data.get('formants', [])
            formant_text = ', '.join(f"{f:.1f}" for f in formants) if formants else '-'
            
            display_text = (
                f"基频: {card_data.get('base_freq', '-'):.1f}Hz\n"
                f"共振峰: {formant_text}Hz\n"
                f"MFCC维度: {mfcc_shape}"
            )
            self.params_display.setText(display_text)

            audio_path = card_data['audio_path']
            audio_data, _ = librosa.load(audio_path, sr=self.rate)
            y = audio_data.astype(np.float32)

            y_clean, mfcc, non_silent = self.get_audio_mfcc(y)

            self.update_waveform(y,non_silent = non_silent)  # 传入完整音频数据


    def confirm_delete_card(self):
        from PyQt5.QtWidgets import QMessageBox
        selected = self.card_list.currentItem()
        if not selected:
            return
        reply = QMessageBox.question(self, '确认删除', '确定要删除这个语音卡吗？',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            
            card_id = selected.text().split(':')[0]
            card_data = self.db.get_voice_card(card_id) 
            self.db.delete_voice_card(card_id)         
            # 删除音频文件
            if card_data and os.path.exists(card_data['audio_path']):
                try:
                    os.remove(card_data['audio_path'])
                except Exception as e:
                    QMessageBox.warning(self, '删除失败', f'文件删除失败: {str(e)}')
            self.load_voice_cards()

    # --------------------------
    # 步骤2：噪声抑制 (使用谱减法)
    # --------------------------
    def reduce_noise(self, y, sr=16000, noise_start=0, noise_end=0.3):
        """
        基于噪声样本的谱减法降噪
        :param noise_start: 噪声样本开始时间（秒）
        :param noise_end: 噪声样本结束时间（秒）
        """
        # 提取噪声样本
        noise_clip = y[int(noise_start*sr):int(noise_end*sr)]
        
        # 使用noisereduce库降噪
        y_clean = nr.reduce_noise(
            y=y, 
            y_noise=noise_clip,
            sr=sr,
            stationary=True  # 假设噪声是稳态的
        )

        # 提取噪声样本
        noise_clip = y[len(y)-int(noise_end*sr):]
        
        # 使用noisereduce库降噪
        y_clean2 = nr.reduce_noise(
            y=y_clean, 
            y_noise=noise_clip,
            sr=sr,
            stationary=True  # 假设噪声是稳态的
        )

        return y_clean2

    # --------------------------
    # 步骤3：静音检测 (基于能量阈值)
    # --------------------------
    def detect_silence(self, y, top_db=25, frame_length=2048, hop_length=512):
        """
        检测非静音区域
        :param top_db: 低于该阈值视为静音（分贝）
        :return: 有效音频段的起始和结束索引
        """
        # 计算分贝值
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # 转换为分贝
        db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # 检测非静音区域
        non_silent = librosa.effects.split(
            y,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )

        # 提取有效音频段
        valid_audio = np.concatenate([y[start:end] for start, end in non_silent])
        return valid_audio, non_silent


    def get_audio_mfcc(self, y, sr=16000):
        # 降噪处理
        y_clean = self.reduce_noise(y)
        # 静音检测
        y_clean, non_silent = self.detect_silence(y_clean)

        # 预加重处理
        pre_emphasis = 0.97
        #y = y_clean.astype(np.float32) / xd
        y_preemphasized = np.append(y_clean[0], y_clean[1:] - pre_emphasis * y_clean[:-1])

        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=y_preemphasized, sr=sr, n_mfcc=13)
        # 提取pitch特征
        pitch = librosa.yin(y=y_preemphasized, fmin=50, fmax=2000).reshape(1, -1)
        # 提取energy特征
        energy = librosa.feature.rms(y=y_preemphasized)
        # 合并特征
        combined_feat = np.concatenate([mfccs, pitch, energy], axis=0)

        # 特征归一化
        mfccs_normalized = librosa.util.normalize(mfccs, axis=1)

        return y_clean, mfccs_normalized, non_silent
    
    # --------------------------
    # 步骤2：DTW相似度计算
    # --------------------------
    def calculate_similarity(self, mfcc1, mfcc2):
        """
        使用DTW计算MFCC序列的相似度
        :param mfcc1: 输入语音MFCC
        :param mfcc2: 目标语音卡MFCC
        :return: 相似度得分 (0~1)
        """
        # 计算DTW最优路径和累计距离
        distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
        
        # 归一化处理（可根据具体需求调整）
        max_length = max(len(mfcc1), len(mfcc2))
        similarity = 1 / (1 + distance/max_length)

        #path_length = len(path)  # 实际对齐路径长度
        #avg_distance = distance / path_length  # 单位路径长度距离
        #similarity = np.exp(-avg_distance / 2.5)  # 温度系数τ=0.5
    
        return round(similarity, 4), distance

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
        #audio_array = np.frombuffer(in_data, dtype=np.int16)
        audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        self.update_waveform(audio_array)
        
        return (in_data, pyaudio.paContinue)

    def update_waveform(self, data=None, isnormalized=True, non_silent=None):
        if data is None:  # 实时模式
            self.waveform_canvas.line.set_data([], [])
            self.waveform_canvas.draw_idle()
            return
        
        # 完整波形模式
        self.waveform_canvas.ax.clear()
        if isinstance(data, np.ndarray):
            
            if isnormalized:
                data = data.astype(np.float32) / np.max(np.abs(data))
            else:
                data = data.astype(np.float32)

            
            # 创建时间轴
            duration = len(data) / self.rate
            time = np.linspace(0, duration, len(data))
            
            # 绘制波形
            self.waveform_canvas.ax.plot(time, data, 'b-', lw=0.5)
            #self.waveform_canvas.ax.set_xlabel('时间 (秒)', fontsize=9)
            self.waveform_canvas.ax.set_xlim(0, duration)
            # 新增：确保坐标轴可见
            self.waveform_canvas.ax.get_xaxis().set_visible(True)

            if non_silent is not None:
                for start, end in non_silent:
                    start_time = start / self.rate
                    end_time = end / self.rate
                    self.waveform_canvas.ax.axvspan(start_time, end_time, color="red", alpha=0.3)
        # 新增：调整布局防止标签被裁剪
        self.waveform_canvas.figure.tight_layout()
        self.waveform_canvas.draw_idle()

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())