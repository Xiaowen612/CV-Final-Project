"""Run: streamlit run app.py"""
from pathlib import Path
import tempfile
import streamlit as st
from experiment.inference import infer

st.set_page_config(page_title='GenImage Detector', layout='centered')
st.title('Real / AI-Generated Image Detector')
default_checkpoint = 'runs_pilot/effort_aug/seed_42/best.pt'
if not Path(default_checkpoint).exists():
    default_checkpoint = 'runs/effort_aug/seed_42/best.pt'
checkpoint = st.text_input('Checkpoint', default_checkpoint)
upload = st.file_uploader('Choose a JPEG or PNG image', type=['jpg', 'jpeg', 'png'])
if upload is not None:
    st.image(upload)
    if st.button('Detect'):
        if not Path(checkpoint).is_file():
            st.error('Checkpoint not found. Complete training first.')
        else:
            with tempfile.TemporaryDirectory() as folder:
                image = Path(folder) / 'input.png'
                image.write_bytes(upload.getvalue())
                result = infer(checkpoint, image)
            st.metric('Prediction', result['label'])
            st.metric('AI score', f"{result['ai_probability']:.1%}")
            st.caption('This score is not proof of image provenance and has not been probability-calibrated.')
