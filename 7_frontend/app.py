import streamlit as st 
import cv2 
import pandas as pd
import numpy as np
import mediapipe as mp
import joblib
import sys


#import func from previous phases
sys.path.append("../5_predict_video")
from detector import predict_strokes_in_vid

st.set_page_config(page_title="SpinTracker", layout="wide")
st.title("SpinTracker")
st.write("Detect Table Tennis Strokes using AI")

#stores model in cache so it doesnt load every user interaction
@st.cache_resource
def load_model():
    return joblib.load("../4_training_model/stroke_classifer_model.pkl")

model = load_model()

print("App Loaded")

st.sidebar.title("Choose Mode")
mode = st.sidebar.radio("Select mode:", ["Upload Video", "Live Webcam"])

if mode == "Upload Video":
    st.subheader("Upload Video Mode")
    st.write("Upload an MP4 video to detect strokes")
    
    uploaded_file = st.file_uploader("Choose a video file", type =["mp4"])
    
    if uploaded_file is not None:
        st.write(f"File uploaded: {uploaded_file.name}")
        
        if st.button("Process Video"):
            
            #creates a temp vid path for function
            temp_vid_path = f"temp_{uploaded_file.name}"
            with open(temp_vid_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            output_video = f"../5_predict_video/outputs/videos/{uploaded_file.name.replace('.mp4', '_label.mp4')}"
            output_csv = f"../5_predict_video/outputs/csv/{uploaded_file.name.replace('.mp4', '_detections.csv')}"
                    
            #Calling phase 5 function 
            with st.spinner("Processing video... might take a few mins"):
                predict_strokes_in_vid(temp_vid_path, output_video, output_csv)
            
            st.session_state['output_video'] = output_video
            st.session_state['output_csv'] = output_csv
            
            #Summary Charts
        if 'output_csv' in st.session_state:
            output_csv = st.session_state['output_csv']
            output_video = st.session_state['output_video']

            st.success("Video processed successfully")
            st.markdown("<h2 style = 'text-align: center;'> Video Summary</h2>", unsafe_allow_html=True)
            df = pd.read_csv(output_csv)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Stroke Frame Counts**")
                stroke_counts = df['stroke'].value_counts()
                st.bar_chart(stroke_counts)
            with col2:
                st.write("**Confidence Distribution**")
                st.bar_chart(df['confidence'].value_counts().sort_index())
                
            #displays strokes in table
            st.dataframe(df) 
            st.subheader("Download Results")
            col_csv, col_vid = st.columns(2)
            
            with col_csv:
                with open(output_csv, 'rb') as f:
                    st.download_button (
                        label = "Download Detection CSV",
                        data = f.read(),
                        file_name= output_csv.split('/')[-1],
                        mime= "text/csv"
                    )
            
            with col_vid:
                with open(output_video, 'rb') as f:
                    st.download_button (
                        label = "Download Labeled Video",
                        data = f.read(),
                        file_name= output_video.split('/')[-1],
                        mime= "video/mp4"
                    )   
    else:
        st.info("Upload a video to start")
    
elif mode == "Live Webcam":
    st.subheader("Live Webcam Mode")
    st.write("Detect strokes in real time")