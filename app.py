import streamlit as  st
import joblib
from PIL import Image
import numpy as np
# 0->carcinoma
# 1->keratosis
# 2->acne
# 3->eczema
# 4->rosacea
# 5->milia

model=joblib.load('models/model2.joblib')

st.title('Skin Condition Classifier')
st.write('Upload an image of skin to get the desired prediction')

uploaded_file=st.file_uploader('Upload an Image',type=['jpg','png'])

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption='image successfully uploaded..',use_column_width=True)

    img=image.resize((224,224))
    img=np.array(img)
    img=np.expand_dims(img,axis=0)

    prediction=model.predict(img).argmax()

    predictions={       0:'Carcinoma', 
                        1:'Keratosis',
                        2:'Acne',
                        3:'Eczema',
                        4:'Rosacea',
                        5:'Milia'       }
    
    st.write(f"Predicted condition :{predictions[prediction]}")
