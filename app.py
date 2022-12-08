from fastai.vision.widgets import *
from fastai.vision.all import *
from PIL import Image
from pathlib import Path

import streamlit as st

st.markdown("<h1 style='text-align: center; color: grey;'>De quelle race est votre toutou? 🐶</h1>", unsafe_allow_html=True)

image= Image.open('malamute.png')
st.image(image, use_column_width= True)
class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(Path()/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Détecter'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            if pred =='error':
                st.write("Oouups!! Ceci n'a pas l'air d'un chien :(")
            else:
                st.write(f'Votre chien ressemble à un {pred.capitalize()} avec une probabilité de {100*(probs[pred_idx]):.02f} %')
        else: 
            st.write(f'Appuyez sur le bouton pour Détecter') 

if __name__=='__main__':

    file_name='dogbreeds_b2_87.pkl'

    predictor = Predict(file_name)