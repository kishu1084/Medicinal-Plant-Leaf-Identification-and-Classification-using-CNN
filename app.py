import os
from tensorflow.keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

# Set page title and header
st.set_page_config(page_title="Leaf Classification", layout="centered")

# Title and subheader
st.title('🌿 Leaf Classification CNN Model')
st.subheader("Upload or Capture Leaf Image for Classification")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f4f8;
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            color: #333;
        }
        .footer {
            font-size: 12px;
            text-align: center;
            padding: 10px;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #e8e8e8;
            color: #333;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 5px;
            height: 50px;
            width: 100%;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .image {
            margin: 20px auto;
            display: block;
        }
        .card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
            text-align: left;
        }
        .medicinal-uses {
            font-size: 14px;
            line-height: 1.5;
        }
        .classification-result {
            font-weight: bold;
            font-size: 18px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Define plant names and their medicinal uses
flower_names = ['Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala']

medicinal_uses = {
    'Turmeric': [
        'Anti-inflammatory and antioxidant properties.',
        'Aids digestion and alleviates symptoms of indigestion.',
        'May reduce symptoms of arthritis and joint pain.',
        'Improves liver function and detoxification.',
        'Enhances skin health and reduces acne.',
        'Protective effects against certain cancers.'
    ],
    'ashoka': [
        'Used for gynecological disorders and uterine health.',
        'Relieves menstrual pain and regulates the menstrual cycle.',
        'Assists in treating hormonal imbalances.',
        'Reduces stress and anxiety.',
        'Improves fertility in women.',
        'Benefits skin conditions like eczema and acne.'
    ],
    'camphor': [
        'Used for cough, congestion, and pain relief.',
        'Relieves muscle pain and stiffness when applied topically.',
        'Effective for reducing itching and irritation.',
        'Has antibacterial properties for minor cuts and infections.',
        'Used in aromatherapy to reduce anxiety.',
        'Common in ointments and balms for headache relief.'
    ],
    'kamakasturi': [
        'Treats respiratory issues and skin ailments.',
        'Alleviates symptoms of bronchitis and asthma.',
        'May have antifungal properties for skin infections.',
        'Soothes eczema and psoriasis.',
        'Reduces inflammation and swelling.',
        'Used in aromatherapy for calming effects.'
    ],
    'kepala': [
        'Antiseptic and anti-inflammatory properties.',
        'Aids in wound healing for cuts and burns.',
        'Helps with respiratory issues, including colds.',
        'Assists in relieving digestive disorders and gas.',
        'Promotes scalp health and reduces dandruff.',
        'Supports overall immune function.'
    ]
}

# Load the pre-trained model
model = load_model('Leaf_Recog_Model.keras')

# Function to classify images
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_plant = flower_names[np.argmax(result)]
    score = np.max(result) * 100

    # Getting medicinal use of the classified plant
    medicinal_use = "\n".join(medicinal_uses[predicted_plant])

    outcome = f'🌿 The image belongs to **{predicted_plant}** with a score of **{score:.2f}%**.'
    outcome += f'\n\n💊 **Medicinal Uses**:\n<div class="medicinal-uses">{medicinal_use}</div>'
    
    return outcome

# Create an "uploads" directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Feature for uploading an image
st.markdown("### Upload a Leaf Image:")
uploaded_file = st.file_uploader('Choose a file', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')

if uploaded_file is not None:
    # Save and classify the uploaded image
    uploaded_file_path = os.path.join('uploads', uploaded_file.name)
    with open(uploaded_file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Uploaded Image", width=300, use_column_width='always', output_format='auto')
    result = classify_images(uploaded_file_path)
    st.markdown(f"<div class='card'>{result}</div>", unsafe_allow_html=True)

# Camera start/stop logic
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False

# Start camera button
if st.button("Start Camera"):
    st.session_state['camera_active'] = True

# Stop camera button


# Show camera input only if camera is active
if st.session_state['camera_active']:
    st.markdown("### Capture a Leaf Image:")
    camera_image = st.camera_input('Capture a photo of the leaf')

    if camera_image is not None:
        # Save and classify the captured image
        camera_image_path = os.path.join('uploads', 'camera_image.jpg')
        with open(camera_image_path, 'wb') as f:
            f.write(camera_image.getbuffer())

        st.image(camera_image, caption="Captured Image", width=300, use_column_width='always', output_format='auto')
        result = classify_images(camera_image_path)
        st.markdown(f"<div class='card'>{result}</div>", unsafe_allow_html=True)

# Adding a footer for a better user experience
st.markdown("""
    <div class="footer">
        Developed by Krishna & Team | Leaf Classification App 🌿
    </div>
""", unsafe_allow_html=True)
