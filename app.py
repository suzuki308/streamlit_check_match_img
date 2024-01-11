import gc

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np

from sift_match import SIFTMatch

st.title("My first Streamlit app")
st.write("Hello, world")


def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    sift_match = SIFTMatch()
    result = sift_match.matching(img)
    if result:
        black = np.zeros(img.shape, np.uint8)
        cv2.putText(black, "Matching", (int(img.shape[1]/2-50), int(img.shape[0]/2-20)), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))
        img_result = cv2.addWeighted(img, 0.5, black, 0.5, 1)
    else:
        img_result = img

    gc.collect()

    return av.VideoFrame.from_ndarray(img_result, format="bgr24")


webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
