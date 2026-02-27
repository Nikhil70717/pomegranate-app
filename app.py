import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PomegranateAI - Precision Agriculture", page_icon="🌳", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #4CAF50; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌳 PomegranateAI: Orchard Health Analytics")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    merge_dist = st.slider("Merge Distance (px)", 50, 500, 200, 50)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("📂 Upload Orchard Map", type=['jpg', 'png', 'jpeg', 'tif'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    large_img = cv2.imdecode(file_bytes, 1)
    h_img, w_img, _ = large_img.shape
    
    # --- ⚠️ THE MEMORY OPTIMIZER ⚠️ ---
    # Streamlit free tier has 1GB limit. Resize if image is massive.
    MAX_DIM = 6000
    GSD = 0.45 
    
    if max(h_img, w_img) > MAX_DIM:
        st.warning("⚠️ Image is very large! Optimizing memory and adjusting scale automatically...")
        scale = MAX_DIM / max(h_img, w_img)
        large_img = cv2.resize(large_img, (0,0), fx=scale, fy=scale)
        h_img, w_img, _ = large_img.shape
        GSD = 0.45 / scale  # Fixes the math so cm measurements stay correct!
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Input Map Preview")
        img_rgb = cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((1000, 1000)) 
        st.image(pil_img, caption=f"Processing Resolution: {w_img}x{h_img} px", use_container_width=True)

    with col2:
        st.subheader("🚀 Analysis")
        if st.button("START ANALYSIS"):
            with st.spinner('AI is scanning your orchard...'):
                
                # Dynamic Patch Sizes to save RAM
                PATCH_W, PATCH_H = min(2000, w_img), min(2000, h_img)
                OVERLAP = int(PATCH_W * 0.15) 
                
                # --- SLIDING WINDOW ---
                all_global_boxes = []
                step_x = max(1, PATCH_W - OVERLAP)
                step_y = max(1, PATCH_H - OVERLAP)
                
                progress_bar = st.progress(0)
                total_steps = ((h_img // step_y) + 1) * ((w_img // step_x) + 1)
                current_step = 0

                for y in range(0, h_img, step_y):
                    for x in range(0, w_img, step_x):
                        current_step += 1
                        progress_bar.progress(min(current_step / total_steps, 0.95))
                        
                        x_end = min(x + PATCH_W, w_img)
                        y_end = min(y + PATCH_H, h_img)
                        patch = large_img[y:y_end, x:x_end]
                        
                        if patch.shape[0] < 100 or patch.shape[1] < 100: continue
                        
                        results = model.predict(patch, imgsz=640, conf=conf_thresh, verbose=False)
                        for box in results[0].boxes:
                            lx1, ly1, lx2, ly2 = box.xyxy[0].cpu().numpy()
                            all_global_boxes.append([lx1 + x, ly1 + y, lx2 + x, ly2 + y])

                # --- MERGING ---
                final_boxes = []
                if len(all_global_boxes) > 0:
                    boxes = np.array(all_global_boxes)
                    keep_mask = np.ones(len(boxes), dtype=bool)
                    for i in range(len(boxes)):
                        if not keep_mask[i]: continue
                        sx1, sy1, sx2, sy2 = boxes[i]
                        for j in range(i + 1, len(boxes)):
                            if not keep_mask[j]: continue
                            ox1, oy1, ox2, oy2 = boxes[j]
                            c1, c2 = [(sx1+sx2)/2, (sy1+sy2)/2], [(ox1+ox2)/2, (oy1+oy2)/2]
                            dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                            if dist < merge_dist:
                                sx1, sy1 = min(sx1, ox1), min(sy1, oy1)
                                sx2, sy2 = max(sx2, ox2), max(sy2, oy2)
                                keep_mask[j] = False
                        final_boxes.append([sx1, sy1, sx2, sy2])

                # --- DRAWING & STATS ---
                stats_data = []
                THICKNESS = max(1, int(w_img / 1000))
                FONT_SCALE = max(0.4, w_img / 4000)

                for box in final_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    diameter_cm = max(x2-x1, y2-y1) * GSD
                    
                    if diameter_cm > 180: cat, color = "Best (>1.8m)", (0, 255, 0)
                    elif diameter_cm > 130: cat, color = "Good (1.3-1.8m)", (0, 255, 255)
                    elif diameter_cm > 80: cat, color = "Average (0.8-1.3m)", (0, 165, 255)
                    else: cat, color = "Small (<0.8m)", (0, 0, 255)
                    
                    stats_data.append(cat)
                    cv2.rectangle(large_img, (x1, y1), (x2, y2), color, THICKNESS)
                    cv2.putText(large_img, f"{int(diameter_cm)}cm", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, THICKNESS)

                progress_bar.progress(1.0)
                
                # --- RESULTS DISPLAY ---
                st.divider()
                st.header("📊 Orchard Report")
                
                df = pd.DataFrame(stats_data, columns=['Category'])
                summary = df['Category'].value_counts().reset_index()
                summary.columns = ['Tree Class', 'Count']
                order_map = {"Best (>1.8m)": 0, "Good (1.3-1.8m)": 1, "Average (0.8-1.3m)": 2, "Small (<0.8m)": 3}
                summary['Order'] = summary['Tree Class'].map(order_map)
                summary = summary.sort_values('Order').dropna()

                plot_colors = [{"Best (>1.8m)": '#00FF00', "Good (1.3-1.8m)": '#FFFF00', "Average (0.8-1.3m)": '#FFA500', "Small (<0.8m)": '#FF0000'}.get(x, '#808080') for x in summary['Tree Class']]
                
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1: st.metric("Total Trees", len(final_boxes))
                with c2: 
                    best = summary[summary['Tree Class'] == "Best (>1.8m)"]['Count'].sum() if "Best (>1.8m)" in summary['Tree Class'].values else 0
                    st.metric("Best Quality", int(best))
                with c3:
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.barh(summary['Tree Class'], summary['Count'], color=plot_colors)
                    st.pyplot(fig)

                st.subheader("🗺️ Classified Map")
                st.image(cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB), use_container_width=True)
