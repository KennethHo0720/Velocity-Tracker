import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
import threading
import queue
import time

# --- 頁面配置 ---
st.set_page_config(page_title="Barbell Tracker Pro V2", layout="wide") 

# 自定義 CSS 以優化手機顯示
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 8px; }
    /* Mobile Touch Fix: Prevent scrolling when touching canvas */
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        touch-action: none; 
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏋️ 杠鈴速度分析 V2 (Web)")
st.caption("移植自 Desktop Pro 版 | 支援 Reps 偵測與降幅分析")
st.markdown("---")

# ---------------------------------------------------------
#  KALMAN FILTER (NEW ADDITION - Ported from Desktop)
# ---------------------------------------------------------
class SimpleKalmanFilter:
    def __init__(self, initial_value, initial_velocity=0.0, process_noise=0.005, measurement_noise=5.0):
        # 狀態向量 [位置, 速度]
        self.x = np.array([[initial_value], [initial_velocity]])
        
        # 狀態共變異數矩陣 P (初始不確定性)
        self.P = np.eye(2) * 1.0
        
        # 狀態轉移矩陣 F (假設恆定速度模型: pos = pos + vel*dt)
        # dt 會在 update 時動態應用，這裡先設基礎結構
        self.F = np.eye(2)
        
        # 測量矩陣 H (我們只測量位置)
        self.H = np.array([[1.0, 0.0]])
        
        # 測量雜訊 R (感測器/追蹤誤差)
        self.R = np.array([[measurement_noise]])
        
        # 過程雜訊 Q (系統內部變化，如運動員突然發力)
        self.Q_base = process_noise

    def process(self, measurement, dt):
        if dt <= 0: return self.x[0, 0], self.x[1, 0]

        # 1. 更新狀態轉移矩陣 F 和過程雜訊 Q
        self.F[0, 1] = dt
        
        # Q 矩陣構建 (Discrete White Noise Acceleration Model)
        # 允許速度隨時間變化 (加速度)
        q_pos = (dt**4)/4
        q_pos_vel = (dt**3)/2
        q_vel = (dt**2)
        
        Q = np.array([
            [q_pos, q_pos_vel],
            [q_pos_vel, q_vel]
        ]) * self.Q_base

        # 2. 預測 (Predict)
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + Q

        # 3. 更新 (Update)
        z = np.array([[measurement]])
        y = z - np.dot(self.H, self.x) # Residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman Gain
        
        self.x = self.x + np.dot(K, y)
        I = np.eye(2)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        # 返回平滑後的位置和速度
        return self.x[0, 0], self.x[1, 0]

def apply_kalman_filter(data, R=0.1, Q=100.0):
    # Backward compatibility wrapper if needed, or just placeholder
    # In the new logic we won't use this, but keeping it just in case of reference error until fully replaced.
    # Actually, we will remove usages.
    pass

class ThreadedVideoReader:
    def __init__(self, path, start_frame, end_frame, scale_factor, rotation_code=None):
        self.path = path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.scale_factor = scale_factor
        self.rotation_code = rotation_code
        self.queue = queue.Queue(maxsize=1024) # Buffer size increased for performance
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
    
    def start(self):
        self.thread.start()
        return self

    def update(self):
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        current_idx = self.start_frame

        while current_idx < self.end_frame:
            if self.stopped:
                break
            
            if not self.queue.full():
                ret, frame = cap.read()
                if not ret:
                    self.stop()
                    break
                
                # Rotation
                if self.rotation_code is not None:
                    frame = cv2.rotate(frame, self.rotation_code)

                # Pre-process in thread
                frame_small = cv2.resize(frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)
                
                self.queue.put((ret, frame_small, current_idx))
                current_idx += 1
            else:
                time.sleep(0.01) # Wait a bit if queue is full

        cap.release()
        self.stopped = True

    def read(self):
        # Return next frame in the queue. 
        # returns (ret, frame, idx) or None if empty/finished
        try:
            return self.queue.get(timeout=1)
        except queue.Empty:
            return None

    def more(self):
        return not self.stopped or not self.queue.empty()

    def stop(self):
        self.stopped = True
        # Drain queue to allow thread to exit if blocked
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


# --- 1. 上傳與初始化 ---
st.header("1. 上傳影片")
uploaded_file = st.file_uploader("選擇影片文件 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Check if a new file is uploaded
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.last_uploaded = uploaded_file.name
        # Clear specific session state keys to force reset
        keys_to_reset = ["initial_drawing", "stroke_color"]
        for k in keys_to_reset:
            if k in st.session_state:
                del st.session_state[k]

    # 保存臨時文件
    # Use delete=False to keep file for OpenCV processing
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close() # Close the file so OpenCV can open it safely on Windows
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if np.isnan(fps) or fps < 1: fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    # --- Global Settings (Sidebar) ---
    st.sidebar.header("設定 (Settings)")
    
    # 1. Mobile Optimization
    is_mobile = st.sidebar.checkbox("📱 手機模式 (Mobile View)", value=True, help="開啟以獲得最佳手機體驗")
    
    # 2. Rotation (Auto Portrait)
    # Get Video Dimensions
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    st.sidebar.subheader("影片旋轉 (Rotation)")
    auto_portrait = st.sidebar.checkbox("🔄 自動轉正 (Auto Portrait)", value=True, help="若影片為橫向 (寬 > 高)，自動旋轉 90 度")
    
    rotation_code = None
    
    if auto_portrait:
        if v_width > v_height:
            rotation_code = cv2.ROTATE_90_CLOCKWISE
            st.sidebar.info(f"已自動旋轉 90 度\n(原始: {v_width}x{v_height})")
    else:
        # Manual Rotation
        rotate_option = st.sidebar.selectbox(
            "手動調整 (Manual)",
            options=[0, 90, 180, 270],
            index=0,
            help="若自動轉正不正確，請關閉自動模式並手動選擇"
        )
        
        if rotate_option == 90:
            rotation_code = cv2.ROTATE_90_CLOCKWISE
        elif rotate_option == 180:
            rotation_code = cv2.ROTATE_180
        elif rotate_option == 270:
            rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            
    # 3. Performance / Speed Mode
    st.sidebar.subheader("效能 (Performance)")
    perf_mode = st.sidebar.radio(
        "處理模式 (Processing Mode)",
        ("Balanced (Default)", "High Accuracy", "Turbo Speed"),
        index=0,
        help="High Accuracy: 逐幀處理\nBalanced: 每 2 幀處理一次 (推薦)\nTurbo: 每 3 幀處理一次 (最快，適合長影片)"
    )
    
    frame_skip = 1 # Default Balanced
    if perf_mode == "High Accuracy":
        frame_skip = 0
    elif perf_mode == "Turbo Speed":
        frame_skip = 2

    # --- 2. 剪輯 (Trim) ---
    st.header("2. 設定分析範圍")
    st.info("💡 拖曳滑桿來選擇起始與結束點 (即時預覽)")
    
    # Trim Sliders with Preview
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        start_t = st.slider("開始時間 (s)", 0.0, duration, 0.0, step=0.1)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
        ret_s, frame_s = cap.read()
        if ret_s:
            if rotation_code is not None:
                frame_s = cv2.rotate(frame_s, rotation_code)
            st.image(frame_s, channels="BGR", caption=f"Start: {start_t}s", width=300)
            
    with col_t2:
        end_t = st.slider("結束時間 (s)", 0.0, duration, duration, step=0.1)
        cap.set(cv2.CAP_PROP_POS_MSEC, end_t * 1000)
        ret_e, frame_e = cap.read()
        if ret_e:
            if rotation_code is not None:
                frame_e = cv2.rotate(frame_e, rotation_code)
            st.image(frame_e, channels="BGR", caption=f"End: {end_t}s", width=300)

    if start_t >= end_t:
        st.error("結束時間必須大於開始時間")
        st.stop()
        
    # --- Advanced Settings (Analysis Thresholds) ---
    with st.expander("⚙️ 進階設定 (Analysis Settings)"):
        st.caption("若無法偵測到較慢的次數 (Grinders)，請嘗試降低速度門檻")
        min_velo_threshold = st.slider("最小速度門檻 (Min Velocity, m/s)", 0.05, 1.0, 0.20, step=0.05)
        kalman_r = st.slider("濾波強度 (Kalman R)", 0.01, 1.0, 0.1, step=0.01, help="數值越大，平滑效果越強，但延遲越高")
        min_rom_threshold = st.slider("最小行程 (Min ROM, m)", 0.05, 0.80, 0.15, step=0.05, help="過濾掉行程過短的誤判 (例如臥推建議 0.15, 深蹲 0.30)")

    # 讀取分析起始幀 (用於畫框)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
    ret, first_frame = cap.read()
    
    if ret:
        if rotation_code is not None:
            first_frame = cv2.rotate(first_frame, rotation_code)

        h_orig, w_orig = first_frame.shape[:2]
        
        # --- 3. 校准與追蹤設定 (Canvas) ---
        st.header("3. 校準與目標設定")
        
        # --- Mobile Optimization (Moved to Global Settings) ---
        # is_mobile definition moved to top
        
        if is_mobile:
            max_canvas_width = 300 # Reduced to 300 to fit smaller screens (iPhone SE is 320px)
            max_canvas_height = 500
        else:
            max_canvas_width = 800
            max_canvas_height = 600

        # Instructions in expander
        with st.expander("👉 繪圖操作說明 (按此展開)", expanded=not is_mobile):
             st.info("👇 操作說明")
             st.markdown("請在下方圖片上依序畫框 (紅 -> 綠)：")
             st.markdown("1. **紅色框**: 校準槓片")
             st.markdown("2. **綠色框**: 追蹤目標")
             if is_mobile:
                 st.markdown("---")
                 st.warning("📱 **手機操作提示**:")
                 st.markdown("- **單指 (One Finger)**: 畫框 (請用力按壓並拖曳)")
                 st.markdown("- **雙指 (Two Fingers)**: 捲動頁面")
                 st.markdown("- 若無法畫圖，請確保網頁沒有放大縮小")

        from streamlit_drawable_canvas import st_canvas
        from PIL import Image

        # 縮放圖片以適應畫布
        h_orig, w_orig = first_frame.shape[:2]
        
        # 計算縮放比例
        scale_w = max_canvas_width / w_orig
        scale_h = max_canvas_height / h_orig
        canvas_scale = min(1.0, scale_w, scale_h)
        
        display_w = int(w_orig * canvas_scale)
        display_h = int(h_orig * canvas_scale)
        
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb).resize((display_w, display_h))
        
        if "stroke_color" not in st.session_state:
            st.session_state.stroke_color = "#FF0000"

        # --- Inline Canvas (Pre-defined Boxes for Mobile Ease) ---
        st.markdown("##### 步驟 3.1: 調整框的位置")
        st.info("👆 直接拖曳框到正確位置。 **紅色=槓片** (校準用), **綠色=追蹤目標**")
        
        # Initial Drawing Objects (Fabric.js JSON format)
        if "initial_drawing" not in st.session_state:
            # Default positions: consistent logic regardless of image size, but use absolute pixels
            # Plate (Red) top-leftish, Target (Green) centerish
            st.session_state.initial_drawing = {
                "version": "4.4.0",
                "objects": [
                    {
                        "type": "rect",
                        "left": int(display_w * 0.1),
                        "top": int(display_h * 0.8),
                        "width": 100,
                        "height": 100,
                        "fill": "rgba(255, 0, 0, 0.2)",
                        "stroke": "#FF0000",
                        "strokeWidth": 3
                    },
                    {
                        "type": "rect",
                        "left": int(display_w * 0.4),
                        "top": int(display_h * 0.3),
                        "width": 100,
                        "height": 100,
                        "fill": "rgba(0, 255, 0, 0.2)",
                        "stroke": "#00FF00",
                        "strokeWidth": 3
                    }
                ]
            }

        # Canvas
        # Use a dynamic key based on filename to force full component remount when video changes
        # This prevents the "Missing file" error caused by the canvas trying to load the old background image
        canvas_key = f"canvas_{st.session_state.last_uploaded}"
        
        c_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)",
            stroke_width=3,
            background_image=frame_pil,
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="transform", # Only allow moving/resizing
            initial_drawing=st.session_state.initial_drawing,
            key=canvas_key,
            display_toolbar=False, # Hide toolbar
        )
        
        # Process Canvas Result & Color Logic
        plate_rect = None
        target_rect = None
        
        if c_result.json_data is not None:
            objects = c_result.json_data["objects"]
            
            # Identify by color
            for obj in objects:
                color = obj.get("stroke", "").upper()
                left = int(obj["left"] / canvas_scale)
                top = int(obj["top"] / canvas_scale)
                w = int(obj["width"] * obj.get("scaleX", 1) / canvas_scale)
                h = int(obj["height"] * obj.get("scaleY", 1) / canvas_scale)
                
                rect = (left, top, w, h)
                
                if color == "#FF0000":
                    plate_rect = rect
                elif color == "#00FF00":
                    target_rect = rect

            if plate_rect and target_rect:
                st.success(f"✅ 設定完成! 槓片: {plate_rect}, 目標: {target_rect}")
            else:
                 # Should theoretically not happen unless they delete it (which is hard without toolbar)
                 st.warning("⚠️ 檢測不到框，請重新整理網頁")

        # --- 4. 執行分析 ---
        st.markdown("###")
        btn_disabled = (plate_rect is None or target_rect is None)
        
        if st.button("🚀 開始智能分析 (Start Analysis)", type="primary", disabled=btn_disabled):
            if btn_disabled:
                st.error("請先完成校準物與目標的框選！")
                st.stop()
            
            # --- 初始化數據 ---
            st.write("正在處理影像... (這可能需要幾秒鐘)")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 使用我們從 Canvas 拿到的座標，而不是 Sliders
            (plate_x, plate_y, plate_w, plate_h) = plate_rect
            # Plate Size 用寬度或高度的平均，或原本邏輯
            # 在原 logic 中 plate_s 只有一個維度，這裡我們取最大邊作為直徑估計
            plate_s = max(plate_w, plate_h) 
            
            (bar_x, bar_y, bar_w, bar_h) = target_rect

            # 1. 設置 Tracker
            # 2. 優化: 計算縮放比例 (Process Scale) 以加速處理
            target_width = 640  # 限制處理寬度為 640px，大幅提升網頁端速度
            process_scale = target_width / float(w_orig)
            
            # 依比例縮小 ROI
            roi_track_small = (int(bar_x * process_scale),
                               int(bar_y * process_scale),
                               int(bar_w * process_scale),
                               int(bar_h * process_scale))
            
            # 依比例縮小第一幀並初始化
            first_frame_small = cv2.resize(first_frame, (0,0), fx=process_scale, fy=process_scale)
            
            # --- 增強追蹤: 灰階 + CLAHE (參考 Local Logic) ---
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            first_frame_gray = cv2.cvtColor(first_frame_small, cv2.COLOR_BGR2GRAY)
            first_frame_enhanced = clahe.apply(first_frame_gray)
            
            tracker = cv2.TrackerCSRT_create()
            tracker.init(first_frame_enhanced, roi_track_small)
            
            # 計算真實世界比例尺 (Meters per Pixel)
            # 假設標準片直徑 0.45 米 (45cm)
            meters_per_pixel = 0.45 / float(plate_s) 
            
            positions = []
            times = []
            
            start_frame = int(start_t * fps)
            curr_frame_idx = start_frame
            end_frame_idx = int(end_t * fps)
            total_frames = end_frame_idx - start_frame
            
            # --- 分析迴圈 ---
            # --- 分析迴圈 (Optimized with Threading) ---
            # Start the threaded video reader
            video_reader = ThreadedVideoReader(video_path, start_frame, end_frame_idx, process_scale, rotation_code)
            video_reader.start()
            
            while video_reader.more():
                item = video_reader.read()
                if item is None:
                    break
                    
                (ret, frame_small, idx) = item
                if not ret:
                    break
                
                # Performance Optimization: Frame Skipping
                current_process_idx = idx - start_frame
                
                # Default assume success for skipped frames (we will interpolate later)
                # But for tracking, we only update tracker on specific frames
                
                if current_process_idx == 0 or current_process_idx % (frame_skip + 1) == 0:
                    # --- 增強追蹤: 灰階 + CLAHE ---
                    frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                    frame_enhanced = clahe.apply(frame_gray)
                    
                    success, box = tracker.update(frame_enhanced)
                    
                    if success:
                        (x, y, bw, bh) = [int(v) for v in box]
                        cy_small = y + bh/2
                        cy_original = cy_small / process_scale
                        
                        positions.append(cy_original)
                        times.append(idx / fps)
                
                # 更新進度條 (每 10 幀更新一次以節省資源)
                if total_frames > 0 and idx % 10 == 0:
                    prog = (idx - start_frame) / total_frames
                    progress_bar.progress(min(prog, 1.0))
            
            progress_bar.progress(1.0)
            video_reader.stop()
            
            # --- 5. 數據後處理 (Data Post-Processing - SYNCED WITH DESKTOP) ---
            if len(positions) > 10:
                # === REPLACED: Updated with Physics-Based Kalman Filter ===
                
                # 1. 數據預處理 (像素 -> 公尺)
                # 原始 positions (cy_original) 是向下增加，所以我們取負號讓「向上」為正
                raw_y_meters = [(-y * meters_per_pixel) for y in positions]
                
                # 2. 初始化卡爾曼濾波器
                # 參數直接對齊 Desktop 版: Process Noise=150.0, Meas Noise=0.1
                kf = SimpleKalmanFilter(
                    initial_value=raw_y_meters[0], 
                    process_noise=150.0, 
                    measurement_noise=0.1 
                )
                
                kalman_pos = []
                kalman_vel = []
                t_clean = []
                
                for i in range(len(raw_y_meters)):
                    current_time = times[i]
                    meas = raw_y_meters[i]
                    
                    if i == 0:
                        dt = 0
                    else:
                        dt = current_time - times[i-1]
                    
                    # 過濾掉異常的時間差 (例如掉幀)
                    if dt > 1.0: dt = 1.0/fps
                    
                    k_pos, k_vel = kf.process(meas, dt)
                    
                    kalman_pos.append(k_pos)
                    kalman_vel.append(k_vel)
                    t_clean.append(current_time)

                # 轉換為 numpy array 以便後續處理
                # 位置我們用相對位移 (從 0 開始)
                y_smooth = np.array(kalman_pos)
                y_smooth = y_smooth - y_smooth[0] 
                
                # 卡爾曼直接給出速度，無需再微分
                velocity_smooth = np.array(kalman_vel)
                time_array = np.array(t_clean)
                
                # 3. REP DETECTION LOGIC (State Machine & VBT Metrics)
                
                # 先計算加速度 (Acceleration) 用於 MPV
                # a = dv/dt
                acceleration = np.gradient(velocity_smooth, time_array)
                
                reps = []
                
                # --- 參數設定 (Config) ---
                MIN_VEL_THRESH = 0.05       
                MIN_DUR_SEC = 0.05          
                MIN_DUR_FRAMES = int(MIN_DUR_SEC * fps) 
                
                # [FIX 1] 將 0.15 改為 0.05 (5公分)
                MIN_ROM_METERS = 0.05       
                
                GRAVITY = 9.81              

                in_concentric = False
                start_idx = 0
                
                for i in range(len(velocity_smooth) - MIN_DUR_FRAMES):
                    v = velocity_smooth[i]
                    
                    if not in_concentric:
                        # --- START TRIGGER ---
                        if v > MIN_VEL_THRESH:
                            # 預讀接下來幾幀
                            future_window = velocity_smooth[i : i + MIN_DUR_FRAMES]
                            
                            # [FIX 2] 將 np.min 改為 np.mean (容許微小抖動)
                            if np.mean(future_window) > MIN_VEL_THRESH:
                                in_concentric = True
                                start_idx = i
                                
                    else:
                        # --- END TRIGGER ---
                        # 條件：速度 < 0 (開始下放) 或 數據結束
                        if v < 0 or i == len(velocity_smooth) - MIN_DUR_FRAMES - 1:
                            in_concentric = False
                            end_idx = i
                            
                            # --- 數據切片 (Slicing) ---
                            vel_slice = velocity_smooth[start_idx : end_idx]
                            acc_slice = acceleration[start_idx : end_idx]
                            pos_slice = y_smooth[start_idx : end_idx]
                            t_slice = time_array[start_idx : end_idx]
                            
                            # --- VALIDATION (過濾無效次數) ---
                            
                            # 1. ROM 檢查 (位移量)
                            rep_rom = pos_slice[-1] - pos_slice[0]
                            if rep_rom < MIN_ROM_METERS:
                                continue # 跳過這次誤判
                                
                            # --- METRICS CALCULATION (計算 MV & MPV) ---
                            
                            # A. Mean Velocity (MV) - 整個向心階段
                            mv = np.mean(vel_slice)
                            
                            # B. Mean Propulsive Velocity (MPV)
                            # 條件：加速度 >= -9.81 m/s^2 (代表運動員正在施力對抗重力)
                            propulsive_mask = acc_slice >= -GRAVITY
                            if np.any(propulsive_mask):
                                mpv = np.mean(vel_slice[propulsive_mask])
                            else:
                                mpv = mv # 異常保護
                            
                            peak_v = np.max(vel_slice)
                            peak_local_idx = np.argmax(vel_slice)
                            peak_time = t_slice[peak_local_idx]
                            
                            reps.append({
                                'start_idx': start_idx,
                                'end_idx': end_idx,
                                'mv': mv,
                                'mpv': mpv,
                                'peak_v': peak_v,
                                'peak_t': peak_time,
                                'rom': rep_rom
                            })

                num_reps = len(reps)
                
                # --- RESULTS AGGREGATION ---
                mv_list = [r['mv'] for r in reps]
                mpv_list = [r['mpv'] for r in reps]
                
                # 計算各自的「整組平均值」 (Set Average)
                avg_mv = float(np.mean(mv_list)) if mv_list else 0.0
                avg_mpv = float(np.mean(mpv_list)) if mpv_list else 0.0 # Renamed to match downstream
                
                # 計算最佳值與疲勞 (以 MPV 為基準)
                best_mpv = max(mpv_list) if mpv_list else 0.0
                worst_mpv = min(mpv_list) if mpv_list else 0.0
                
                loss_pct = 0.0
                slowest_rep_num = 0
                
                if mpv_list:
                    # 找出最慢的一下 (基於 MPV)
                    slowest_idx = mpv_list.index(worst_mpv)
                    # 計算疲勞流失率
                    loss_pct = ((best_mpv - worst_mpv) / best_mpv) * 100
                
                # For compatibility with plotting code below
                height_smooth = y_smooth 
                
                # Drop calculations - keeping original UI variable name 'biggest_drop_pct' for display logic below
                biggest_drop_pct = loss_pct
                drop_reps_indices = (-1, -1) # Disable the old pair-wise drop logic for now or mapped to fatigue
                
                # --- 6. 結果展示 ---
                st.success(f"分析完成！偵測到 {num_reps} 組動作 (Reps)")
                
                # 統計數據卡片 (Result Metrics - Desktop Style)
                c1, c2, c3 = st.columns(3)
                c1.metric("Best MPV", f"{best_mpv:.2f} m/s", delta=f"Avg: {avg_mpv:.2f}")
                c2.metric("Set Avg MV", f"{avg_mv:.2f} m/s")
                
                loss_label = "Fatigue (Loss%)"
                if fastest_rep_idx := locals().get('slowest_rep', None): # Not defined here actually
                    pass 
                
                c3.metric(loss_label, f"{loss_pct:.1f}%", delta_color="inverse" if loss_pct > 10 else "normal")

                # --- 繪圖 (Matplotlib) ---
                # --- 繪圖 (Matplotlib) - PRO Style ---
                # 使用 Dark Background 風格
                plt.style.use('dark_background')
                
                # Create Figure with specific bg color to match Streamlit dark theme approximation
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('#0e1117') # Match Streamlit dark bg
                ax.set_facecolor('#0e1117')
                
                # 1. 繪製「背景」底層曲線 (Raw Velocity)
                ax.plot(time_array, velocity_smooth, color='#444444', linewidth=1.0, alpha=0.6, label='Raw Velocity')
                
                # 2. 繪製「有效向心階段」
                max_v_limit = 0
                
                for i, r in enumerate(reps):
                    rep_num = i + 1
                    t_s_idx = r['start_idx']
                    t_e_idx = r['end_idx']
                    
                    t_segment = time_array[t_s_idx : t_e_idx+1]
                    v_segment = velocity_smooth[t_s_idx : t_e_idx+1]
                    
                    if len(t_segment) > 0:
                        # Plot Rep Segment
                        ax.plot(t_segment, v_segment, color='#00e5ff', linewidth=2.5)
                        # Fill Area
                        ax.fill_between(t_segment, v_segment, 0, color='#00e5ff', alpha=0.2)
                        
                        # Annotations
                        display_val = r['mpv'] # Priority: MPV
                        peak_time = r['peak_t']
                        peak_val = r['peak_v']
                        
                        label_text = f"#{rep_num}\n{display_val:.2f}"
                        
                        ax.annotate(
                            label_text, 
                            xy=(peak_time, peak_val), 
                            xytext=(0, 20), 
                            textcoords='offset points', 
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold', color='#ffffff',
                            bbox=dict(boxstyle="round,pad=0.3", fc="#262730", ec="#00e5ff", alpha=0.9)
                        )
                        
                        max_v_limit = max(max_v_limit, peak_val)
                        
                # 3. 輔助線與網格
                ax.axhline(0, color='#666666', linewidth=1, linestyle='--')
                
                # Axis Styling
                ax.set_xlabel('Time (s)', color='#fafafa', fontsize=10)
                ax.set_ylabel('Velocity (m/s)', color='#00e5ff', fontweight='bold', fontsize=10)
                
                ax.tick_params(axis='x', colors='#fafafa')
                ax.tick_params(axis='y', colors='#fafafa')
                
                # Hide Spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#444444')
                ax.spines['left'].set_color('#444444')
                
                ax.grid(True, which='major', axis='y', alpha=0.1, color="#ffffff", linestyle=':')
                
                if max_v_limit > 0:
                    ax.set_ylim(top=max_v_limit * 1.3) # Leave space for annotations
                
                st.pyplot(fig)
                
                # --- 下載數據 ---
                # Build detailed rep table
                rep_data = []
                for i, r in enumerate(reps):
                    rep_data.append({
                        "Rep": i+1,
                        "Mean Velocity (m/s)": round(r['mv'], 3),
                        "Mean Propulsive V (m/s)": round(r['mpv'], 3),
                        "Peak Velocity (m/s)": round(r['peak_v'], 3),
                        "ROM (m)": round(r['rom'], 3),
                        "Duration (s)": round(time_array[r['end_idx']] - time_array[r['start_idx']], 3)
                    })
                
                st.write("### 詳細數據 (Detailed Data)")
                st.dataframe(pd.DataFrame(rep_data))
                
                df = pd.DataFrame({
                    "Time": time_array, 
                    "Velocity": velocity_smooth, 
                    "Acceleration": acceleration,
                    "Height": height_smooth
                })
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 下載 Raw CSV 數據", csv, "barbell_analysis_raw.csv", "text/csv")
                
            else:
                st.error("❌ 追蹤失敗或數據太短。請嘗試：\n1. 調整綠色追蹤框的位置\n2. 確保影片光線充足且背景單純")

    else:
        st.error("無法讀取影片幀，請檢查影片格式。")