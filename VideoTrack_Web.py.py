import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd

# --- 頁面配置 ---
st.set_page_config(page_title="Barbell Tracker Pro V2", layout="centered") 

# 自定義 CSS 以優化手機顯示
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 8px; }
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
    
    # --- 2. 剪輯 (Trim) ---
    st.header("2. 設定分析範圍")
    st.info("💡 手機端請輸入數字來精確調整時間")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        start_t = st.number_input("開始時間 (s)", 0.0, duration, 0.0, step=0.5)
    with col_t2:
        end_t = st.number_input("結束時間 (s)", 0.0, duration, duration, step=0.5)
    
    if start_t >= end_t:
        st.error("結束時間必須大於開始時間")
        st.stop()

    # 讀取預覽幀
    start_frame = int(start_t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = cap.read()
    
    if ret:
        h_orig, w_orig = first_frame.shape[:2]
        
        # --- 3. 校准與追蹤設定 (Sliders) ---
        st.header("3. 校準與目標設定")
        st.warning("⚠️ 請務必確認紅框包住槓片、綠框包住槓鈴末端")

        # 使用 Expander 節省空間
        with st.expander("🛠️ 點擊展開調整位置 (校準/追蹤)", expanded=True):
            st.subheader("🔴 校準物 (Plate 45lb/20kg)")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                plate_x = st.slider("Plate X", 0, w_orig, int(w_orig*0.2), key="px")
                plate_y = st.slider("Plate Y", 0, h_orig, int(h_orig*0.5), key="py")
            with col_p2:
                # 預設給大一點的範圍，方便手機調整
                plate_s = st.slider("Plate Size", 10, 400, int(w_orig*0.15), key="ps")
            
            st.markdown("---")
            st.subheader("🟢 追蹤目標 (Bar End)")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                bar_x = st.slider("Bar X", 0, w_orig, int(w_orig*0.5), key="bx")
                bar_y = st.slider("Bar Y", 0, h_orig, int(h_orig*0.5), key="by")
            with col_b2:
                bar_w = st.slider("Bar Width", 10, 200, 60, key="bw")
                bar_h = st.slider("Bar Height", 10, 200, 60, key="bh")

        # --- 繪製預覽圖 ---
        # 為了手機顯示，這裡我們縮小顯示用的圖片，但不影響原始座標
        display_frame = first_frame.copy()
        cv2.rectangle(display_frame, (plate_x, plate_y), (plate_x + plate_s, plate_y + plate_s), (0, 0, 255), 4)
        cv2.putText(display_frame, "Plate", (plate_x, plate_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), 4)
        cv2.putText(display_frame, "Target", (bar_x, bar_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        
        st.image(display_frame, channels="BGR", caption="設定預覽 (請確保框選準確)", use_container_width=True)
        
        # --- 4. 執行分析 ---
        st.markdown("###")
        if st.button("🚀 開始智能分析 (Start Analysis)", type="primary"):
            
            # --- 初始化數據 ---
            st.write("正在處理影像... (這可能需要幾秒鐘)")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
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
            # 注意：這裡要用縮放後的像素大小來計算，或者用原始像素
            # 為了簡單，我們用原始像素計算比例，最後將追蹤到的像素還原回原始尺寸
            meters_per_pixel = 0.45 / float(plate_s) 
            
            positions = []
            times = []
            
            curr_frame_idx = start_frame
            end_frame_idx = int(end_t * fps)
            total_frames = end_frame_idx - start_frame
            
            # --- 分析迴圈 ---
            while curr_frame_idx < end_frame_idx:
                ret, frame = cap.read()
                if not ret: break
                
                # 縮小畫面進行追蹤 (加速關鍵)
                frame_small = cv2.resize(frame, (0,0), fx=process_scale, fy=process_scale)
                
                success, box = tracker.update(frame_small)
                if success:
                    (x, y, bw, bh) = [int(v) for v in box]
                    cy_small = y + bh/2
                    # 還原回原始尺寸的 Y 座標
                    cy_original = cy_small / process_scale
                    
                    positions.append(cy_original)
                    times.append(curr_frame_idx / fps)
                
                curr_frame_idx += 1
                
                # 更新進度條 (每 10 幀更新一次以節省資源)
                if total_frames > 0 and curr_frame_idx % 10 == 0:
                    prog = (curr_frame_idx - start_frame) / total_frames
                    progress_bar.progress(min(prog, 1.0))
            
            progress_bar.progress(1.0)
            cap.release()
            
            # --- 5. 數據後處理 (Data Post-Processing) ---
            if len(positions) > 10:
                pos_array = np.array(positions)
                time_array = np.array(times)
                
                # A. 像素轉位移 (Y軸向下為正，需反轉)
                # 假設起始位置為 0，向上移動為正
                height_pixels = -(pos_array - pos_array[0])
                height_m = height_pixels * meters_per_pixel
                height_smooth = smooth_data(height_m, 15) # 平滑位移
                
                # B. 計算速度 (Gradient)
                velocity = np.gradient(height_smooth, time_array)
                velocity_smooth = smooth_data(velocity, 9) # 平滑速度
                
                # C. 尋找 Reps (Peak Detection) - 移植自 Desktop 版
                candidate_peaks = []
                # 閾值：速度必須大於 0.3 m/s 且是局部最大值
                for i in range(1, len(velocity_smooth)-1):
                    if velocity_smooth[i] > velocity_smooth[i-1] and velocity_smooth[i] > velocity_smooth[i+1]:
                        if velocity_smooth[i] > 0.3:
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