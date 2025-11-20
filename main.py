import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Signature Forgery Detection",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Sidebar Background */
    [data-testid="stSidebar"] { background-color: #1e1e1e; }
    [data-testid="stSidebar"] > div:first-child { background-color: #1e1e1e; }
    
    /* Radio Buttons (Make them look like toggle tabs) */
    div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 10px;
        width: 100%;
    }
    div[data-testid="stRadio"] > div {
        background-color: #262730;
        padding: 10px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sumSquared = tf.keras.backend.sum(tf.keras.backend.square(featsA - featsB), axis=1, keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sumSquared, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred + 1e-7, 0)))

@st.cache_resource
def load_signature_model():
    custom_objects = {'euclidean_distance': euclidean_distance, 'contrastive_loss': contrastive_loss}
    try:
        return tf.keras.models.load_model('best_siamese_model.h5', custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None

model = load_signature_model()

# --- SIDEBAR CONTENT (Restored) ---
if model:
    st.sidebar.success("✅ Model loaded successfully!")

st.sidebar.markdown("---")
st.sidebar.header("About the Project")
st.sidebar.info(
    "This web application is a deployment of the research project 'Distinguishing Genuine and Forged Signatures Using Siamese Networks'.\n\n"
    "**Authors:**\n"
    "- Khinje Louis P. Curugan\n"
    "- Rui Manuel A. Palabon\n"
    "- Aj Ian L. Resurreccion\n\n"
    "**Institution:**\n"
    "University of Southeastern Philippines"
)

# --- ROBUST PREPROCESSING ---
def preprocess_image(image_input):
    IMG_SIZE = (150, 150)
    
    # 1. Convert PIL to Numpy if needed
    if isinstance(image_input, Image.Image):
        image_input = np.array(image_input.convert('RGB'))
    
    img = image_input.astype('uint8')

    # 2. Handle Transparency (RGBA -> RGB White Background)
    if len(img.shape) == 3 and img.shape[2] == 4:
        background = np.ones_like(img[:, :, :3]) * 255
        alpha = img[:, :, 3] / 255.0
        for c in range(3):
            background[:, :, c] = alpha * img[:, :, c] + (1 - alpha) * 255
        img = background.astype('uint8')
    
    # 3. Convert to Grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 4. Smart Inversion
    if np.mean(img) > 127:
        img = 255 - img
        
    # 5. Thresholding
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Smart Crop
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2*pad)
        h = min(img.shape[0] - y, h + 2*pad)
        img = img[y:y+h, x:x+w]

    # 7. Resize with Padding
    old_size = img.shape[:2]
    ratio = float(IMG_SIZE[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    
    img = cv2.resize(img, (new_size[1], new_size[0]))
    
    delta_w = IMG_SIZE[1] - new_size[1]
    delta_h = IMG_SIZE[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # 8. Normalize
    img_final = img.astype('float32') / 255.0
    img_final = np.expand_dims(img_final, axis=-1)
    img_final = np.expand_dims(img_final, axis=0)
    
    return img_final, img

# --- MAIN UI LAYOUT ---
st.title("✒️ Signature Forgery Detection")
st.markdown("Using a Siamese Neural Network to distinguish between genuine and forged signatures.")

# --- INSTRUCTIONS (Restored) ---
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    ### Instructions:
    1.  **Select Input Method:** Choose between **Upload Image** or **Draw Signature** for both Reference and Test sections.
    2.  **Provide Signatures:** Upload a clear image or draw the signature on the canvas.
    3.  **Verify:** Once both signatures are provided, click the **"Verify Signatures"** button.
    """)
st.markdown("---")

col1, col2 = st.columns(2)

def render_input_column(col, title, key_prefix):
    """
    Renders input column using Radio Buttons to enforce Single Mode (Draw OR Upload).
    """
    final_image = None
    
    with col:
        st.header(title)
        
        # MODE SELECTOR
        mode = st.radio(
            "Input Method:", 
            ["Upload Image", "Draw Signature"], 
            key=f"{key_prefix}_mode",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.info(f"Mode: **{mode}**")
        
        if mode == "Upload Image":
            uploaded_file = st.file_uploader("Choose file", type=['png', 'jpg', 'jpeg'], key=f"{key_prefix}_up")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=250)
                final_image = image
        
        else: # Draw Mode
            canvas_result = st_canvas(
                stroke_width=3, 
                stroke_color="black", 
                background_color="white",
                height=300, 
                width=500, 
                key=f"{key_prefix}_canvas",
                display_toolbar=True
            )
            
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                final_image = canvas_result.image_data

    return final_image, mode

# --- GET INPUTS ---
ref_image, ref_mode = render_input_column(col1, "Reference (Genuine)", "ref")
test_image, test_mode = render_input_column(col2, "Test (To Verify)", "test")

st.markdown("---")

# --- VERIFICATION LOGIC ---
if st.button("🔍 Verify Signatures", type="primary"):
    
    if ref_image is None:
        st.error(f"❌ Reference Missing! Please provide a signature in '{ref_mode}' mode.")
    elif test_image is None:
        st.error(f"❌ Test Missing! Please provide a signature in '{test_mode}' mode.")
    elif model is None:
        st.error("❌ Model could not be loaded.")
    else:
        try:
            # Preprocess
            ref_processed, ref_viz = preprocess_image(ref_image)
            test_processed, test_viz = preprocess_image(test_image)
            
            st.write("### 👁️ What the Model Sees")
            c1, c2 = st.columns(2)
            c1.image(ref_viz, caption=f"Ref ({ref_mode})", width=200, clamp=True, channels='GRAY')
            c2.image(test_viz, caption=f"Test ({test_mode})", width=200, clamp=True, channels='GRAY')
            
            # Predict
            distance = model.predict([ref_processed, test_processed], verbose=0)[0][0]
            THRESHOLD = 0.45 
            
            st.write("### 📊 Results")
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                if distance < THRESHOLD:
                    st.success("✅ **MATCH CONFIRMED: GENUINE**")
                    st.write("The signatures are statistically identical.")
                else:
                    st.error("🚫 **MISMATCH DETECTED: FORGED**")
                    st.write("The signatures differ significantly.")
            
            with res_col2:
                st.metric("Dissimilarity Score", f"{distance:.4f}")
                st.caption(f"Threshold: {THRESHOLD}")
                
        except Exception as e:
            st.error(f"Processing Error: {e}")