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

# --- 輔助函數: 平滑處理 ---
def smooth_data(data, window_size):
    if len(data) < window_size: return data
    window = np.hanning(window_size)
    window = window / window.sum()
    return np.convolve(data, window, mode='same')

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
    # 保存臨時文件
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
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
        smooth_window = st.slider("平滑係數 (Smoothing Window)", 3, 21, 9, step=2)

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
        c_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)",
            stroke_width=3,
            background_image=frame_pil,
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="transform", # Only allow moving/resizing
            initial_drawing=st.session_state.initial_drawing,
            key="main_canvas_transform",
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
            tracker = cv2.TrackerCSRT_create()
            tracker.init(first_frame_small, roi_track_small)
            
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
                    success, box = tracker.update(frame_small)
                    
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
            
            # --- 5. 數據後處理 (Data Post-Processing) ---
            if len(positions) > 5:
                # Interpolation for Performance Mode
                if frame_skip > 0 and len(positions) > 1:
                    full_times = np.linspace(times[0], times[-1], int((times[-1]-times[0])*fps))
                    full_positions = np.interp(full_times, times, positions)
                    
                    time_array = full_times
                    pos_array = full_positions
                else:
                    pos_array = np.array(positions)
                    time_array = np.array(times)
                
                # A. 像素轉位移 (Y軸向下為正，需反轉)
                # 假設起始位置為 0，向上移動為正
                height_pixels = -(pos_array - pos_array[0])
                height_m = height_pixels * meters_per_pixel
                height_smooth = smooth_data(height_m, 15) # 平滑位移
                
                # B. 計算速度 (Gradient)
                velocity = np.gradient(height_smooth, time_array)
                velocity_smooth = smooth_data(velocity, smooth_window) # 使用自定義平滑係數
                
                # C. 尋找 Reps (Peak Detection) - 移植自 Desktop 版
                candidate_peaks = []
                # 閾值：速度必須大於 min_velo_threshold 且是局部最大值
                for i in range(1, len(velocity_smooth)-1):
                    if velocity_smooth[i] > velocity_smooth[i-1] and velocity_smooth[i] > velocity_smooth[i+1]:
                        if velocity_smooth[i] > min_velo_threshold:
                            candidate_peaks.append({'v': velocity_smooth[i], 't': time_array[i], 'idx': i})
                
                # D. 合併接近的 Peaks (Merge Reps)
                reps = []
                merge_window = 1.5  # 秒
                
                for peak in candidate_peaks:
                    if not reps:
                        reps.append(peak)
                    else:
                        last_rep = reps[-1]
                        if (peak['t'] - last_rep['t']) < merge_window:
                            # 如果時間太近，保留速度較大的那個
                            if peak['v'] > last_rep['v']:
                                reps[-1] = peak
                        else:
                            reps.append(peak)
                            
                peak_vs = [r['v'] for r in reps]
                num_reps = len(reps)
                
                # E. 計算進階統計 (Biggest Drop, etc.)
                avg_v = np.mean(peak_vs) if peak_vs else 0
                min_v = np.min(peak_vs) if peak_vs else 0
                max_v = np.max(peak_vs) if peak_vs else 0
                
                biggest_drop_pct = 0
                drop_reps_indices = (-1, -1) # (index of rep A, index of rep B)
                
                # Logic synchronized with Desktop App: Find Biggest Absolute Drop first
                if num_reps > 1:
                    max_drop_val = 0
                    v_start_for_pct = 0
                    
                    for i in range(num_reps - 1):
                        drop = peak_vs[i] - peak_vs[i+1]
                        if drop > max_drop_val:
                            max_drop_val = drop
                            v_start_for_pct = peak_vs[i]
                            drop_reps_indices = (i, i+1) # 0-based index
                            
                    # Calculate Percentage for the biggest absolute drop
                    if max_drop_val > 0 and v_start_for_pct > 0:
                        biggest_drop_pct = (max_drop_val / v_start_for_pct) * 100
                
                # --- 6. 結果展示 ---
                st.success(f"分析完成！偵測到 {num_reps} 組動作 (Reps)")
                
                # 統計數據卡片
                c1, c2, c3 = st.columns(3)
                c1.metric("平均峰值速度", f"{avg_v:.2f} m/s", delta=f"Max: {max_v:.2f}")
                c2.metric("最慢一下 (Slowest)", f"{min_v:.2f} m/s")
                
                drop_str = f"{biggest_drop_pct:.1f}%"
                drop_label = "最大降幅 (Drop)"
                if drop_reps_indices[0] != -1:
                    drop_label += f" (R{drop_reps_indices[0]+1} -> R{drop_reps_indices[1]+1})"
                c3.metric(drop_label, drop_str, delta_color="inverse" if biggest_drop_pct > 10 else "normal")

                # --- 繪圖 (Matplotlib) ---
                fig, ax = plt.subplots(figsize=(10, 5))
                # 繪製速度曲線
                ax.plot(time_array, velocity_smooth, color='#1f77b4', linewidth=2, label='Velocity', alpha=0.8)
                ax.axhline(0, color='black', alpha=0.3, linewidth=1)
                
                # 標記 Reps
                for i, r in enumerate(reps):
                    rep_num = i + 1
                    # 預設顏色
                    color = 'red'
                    size = 50
                    
                    # 如果是最大降幅涉及的那兩下，改為紫色
                    if drop_reps_indices[0] != -1:
                        if i == drop_reps_indices[0] or i == drop_reps_indices[1]:
                            color = 'purple'
                            size = 80
                    
                    ax.scatter(r['t'], r['v'], color=color, s=size, zorder=5)
                    ax.annotate(f"{r['v']:.2f}\n(R{rep_num})", 
                                (r['t'], r['v']), 
                                xytext=(0, 15), 
                                textcoords='offset points', 
                                ha='center', 
                                fontsize=9, 
                                fontweight='bold',
                                color='#333')
                
                ax.set_title(f"Velocity Profile ({num_reps} Reps)", fontsize=12)
                ax.set_ylabel("Speed (m/s)")
                ax.set_xlabel("Time (s)")
                ax.grid(True, alpha=0.3, linestyle='--')
                
                st.pyplot(fig)
                
                # --- 下載數據 ---
                df = pd.DataFrame({
                    "Time": time_array, 
                    "Velocity": velocity_smooth, 
                    "Height": height_smooth
                })
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 下載詳細 CSV 數據", csv, "barbell_analysis.csv", "text/csv")
                
            else:
                st.error("❌ 追蹤失敗或數據太短。請嘗試：\n1. 調整綠色追蹤框的位置\n2. 確保影片光線充足且背景單純")

    else:
        st.error("無法讀取影片幀，請檢查影片格式。")