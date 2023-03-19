import numpy as np
import os, cv2
from tensorflow import keras
from keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

model = load_model('model', compile=False)
model.compile()

st.title('Letter recognition')\

SIZE = 192
mode = st.checkbox("Draw or delete", True)

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width = 20,
    stroke_color = '#ffffff',
    background_color = '#000000',
    width = SIZE,
    height = SIZE,
    drawing_mode = 'freedraw' if mode else "transform",
    key = "canvas"
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28,28))
    rescaled = cv2.resize(img, (SIZE,SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('model input')
    st.image(rescaled)

if st.button('predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1,28,28))
    st.write(f'result: {np.argmax(val[0])}')
    st.bar_chart(val[0])