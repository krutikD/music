import base64
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('image_bg.jpg')

st.markdown("<h2 style='text-align: center; color: white;'><b> -- Emotion based Music Recommender -- </b></h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'><b>''Hey there!👋''</b></h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>''Welcome to a Smart Music recommender system that suggests you songs based on your emotion detected through your facial features''</b></h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: purple;'><b>''Let's get you started!''</b></h5>", unsafe_allow_html=True)


if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		##############################
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []

		if res.face_landmarks:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
				for i in res.left_hand_landmarks.landmark:
					lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			if res.right_hand_landmarks:
				for i in res.right_hand_landmarks.landmark:
					lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			lst = np.array(lst).reshape(1,-1)

			pred = label[np.argmax(model.predict(lst))]

			print(pred)
			cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

			np.save("emotion.npy", np.array([pred]))

		##############################

		return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Enter your preferred Language")


if lang and st.session_state["run"] != "false":
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me songs")

if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
		st.session_state["run"] = "true"
	else:
		webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"
