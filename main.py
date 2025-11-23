import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import base64

def get_image_base64(image_path):
    """Convert local image to base64 string for HTML embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Signature Forgery Detection",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide GitHub icon and deploy button
# Hide GitHub icon, deploy button, and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;}
    .viewerBadge_link__1S137 {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    .stDeployButton {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
# Create header with About button
header_col1, header_col2 = st.columns([6, 1])

with header_col1:
    st.title("✒️ Signature Forgery Detection")
    st.markdown("Using a Siamese Neural Network to distinguish between genuine and forged signatures.")

with header_col2:
    st.write("")  # Spacer for alignment
    if st.button("ℹ️ About", key="about_button", use_container_width=True):
        st.session_state.show_about = True

# About Dialog/Modal
if st.session_state.get('show_about', False):
    with st.container():
        st.markdown("---")
        
        # Header with close button
        about_header_col1, about_header_col2 = st.columns([6, 1])
        with about_header_col1:
            st.markdown("### 📋 About This Project")
        with about_header_col2:
            if st.button("✖", key="close_about"):
                st.session_state.show_about = False
                st.rerun()
        
        # Welcome Section
        st.info("""
        **Welcome!**  
        This is a deployed web app for a research project about signature forgery detection using Siamese Neural Networks.
        """)
        
        st.markdown("---")
        
        # Authors Section
        st.markdown("### 👥 Research Authors")
        
        author_col1, author_col2, author_col3 = st.columns(3)
        
        # Convert images to base64 for HTML embedding
        author1_b64 = get_image_base64("assets/BSCS3_Khin.jpg")
        author2_b64 = get_image_base64("assets/BSCS2_Rui.jpg")
        author3_b64 = get_image_base64("assets/BSCS3_Ian.jpeg")

        with author_col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #2ECC71; border-radius: 10px; color: white;">
                <h4 style="margin-bottom: 5px;">Khinje Louis P. Curugan</h4>
                <p style="margin-bottom: 15px;"><strong>BSCS Student</strong></p>
                <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                    <img src="data:image/jpeg;base64,{author1_b64}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; border: 3px solid white;">
                </div>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
                <p style="font-size: 15px; margin: 0;">College of Information and Computing<br>
                BS Computer Science - Major in Data Science<br>
                CS 3110 Modelling and Simulation [BSCS 3, AY 2025-2026]</p>
            </div>
            """, unsafe_allow_html=True)
        
        with author_col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #9B59B6; border-radius: 10px; color: white;">
                <h4 style="margin-bottom: 5px;">Rui Manuel A. Palabon</h4>
                <p style="margin-bottom: 15px;"><strong>BSCS Student</strong></p>
                <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                    <img src="data:image/jpeg;base64,{author2_b64}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; border: 3px solid white;">
                </div>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
                <p style="font-size: 15px; margin: 0;">College of Information and Computing<br>
                BS Computer Science - Major in Data Science<br>
                CS 3110 Modelling and Simulation [BSCS 3, AY 2025-2026]</p>
            </div>
            """, unsafe_allow_html=True)
        
        with author_col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #5D6D7E; border-radius: 10px; color: white;">
                <h4 style="margin-bottom: 5px;">Aj Ian L. Resurreccion</h4>
                <p style="margin-bottom: 15px;"><strong>BSCS Student</strong></p>
                <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                    <img src="data:image/jpeg;base64,{author3_b64}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; border: 3px solid white;">
                </div>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
                <p style="font-size: 15px; margin: 0;">College of Information and Computing<br>
                BS Computer Science - Major in Data Science<br>
                CS 3110 Modelling and Simulation [BSCS 3, AY 2025-2026]</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Datasets Section
        st.markdown("### 📊 Datasets Used")
        
        dataset_col1, dataset_col2 = st.columns(2)
        
        with dataset_col1:
            st.markdown("#### 1. GPDS 1-150 Dataset")
            st.write("**About:** Signature dataset from Kaggle")
            st.write("**Link to Dataset:**")
            st.link_button(
                "🔗 View on Kaggle",
                "https://www.kaggle.com/datasets/adeelajmal/gpds-1150/data",
                use_container_width=True
            )
            st.caption("Source: Kaggle - GPDS 1-150")
        
        with dataset_col2:
            st.markdown("#### 2. CEDAR Signature Database")
            st.write("**About:** Consists of signatures from 55 writers with 24 original signatures and 24 skilled forgeries each")
            st.write("**Links:**")
            st.link_button(
                "🔗 Official Website",
                "https://cedar.buffalo.edu/signature/",
                use_container_width=True
            )
            st.link_button(
                "⬇️ Download Dataset",
                "https://github.com/nikostsagk/signature-verification/releases/download/cedar/cedar_dataset.zip",
                use_container_width=True
            )
            st.caption("Source: CEDAR, University at Buffalo")
        
        st.markdown("---")
        
        # Institution
        st.markdown("### 🏛️ Institution")
        st.info("**University of Southeastern Philippines**")
        
        st.markdown("---")

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
                stroke_width=4, 
                stroke_color="black", 
                background_color="white",
                height=300, 
                width=700, 
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
            c1.image(ref_viz, caption=f"Ref ({ref_mode})", width=250, clamp=True, channels='GRAY')
            c2.image(test_viz, caption=f"Test ({test_mode})", width=250, clamp=True, channels='GRAY')
            
            # Predict
            distance = model.predict([ref_processed, test_processed], verbose=0)[0][0]
            THRESHOLD = 0.45 
            
            st.write("### 📊 Verification Results") #
            res_col1, res_col2 = st.columns([2, 1])
                                                
            # Calculate confidence score (inverse of normalized distance)
            max_distance = 2.0  # Theoretical maximum for normalized euclidean distance
            confidence_percentage = max(0, min(100, (1 - distance / max_distance) * 100))
            is_genuine = distance < THRESHOLD
                        
            # Main Result Display
            if is_genuine:
                st.success("✅ **MATCH CONFIRMED: GENUINE SIGNATURE**")
            else:
                st.error("🚫 **MISMATCH DETECTED: POTENTIAL FORGERY**")
                        
            st.markdown("---")
                        
            # Statistics in organized columns
            stat_col1, stat_col2, stat_col3 = st.columns(3)
                        
            with stat_col1:
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence_percentage:.1f}%",
                    delta="High Confidence" if confidence_percentage > 70 else "Low Confidence",
                    delta_color="normal" if confidence_percentage > 70 else "inverse",
                    help="Model's confidence in the verification result. Low % = Different (Forged), High % = Identical (Genuine)."
                )
                
            with stat_col2:
                st.metric(
                    label="Dissimilarity Distance",
                    value=f"{distance:.4f}",
                    delta="Below Threshold" if is_genuine else "Above Threshold",
                    delta_color="normal" if is_genuine else "inverse",
                    help="Lower values indicate more similar signatures"
                )
                
            with stat_col3:
                st.metric(
                    label="Decision Threshold",
                    value=f"{THRESHOLD}",
                    help="Signatures with distance below this value are considered genuine"
                )

            st.markdown("---")
                        
            # Detailed Analysis Section
            st.write("### 📈 Detailed Analysis")
                        
            analysis_col1, analysis_col2 = st.columns(2)
                        
            with analysis_col1:
                st.markdown("**Statistical Metrics:**")
                
                # Calculate similarity percentage (inverse of distance)
                similarity_percentage = max(0, (1 - distance) * 100)
                
                # Determine verification status
                if is_genuine:
                    status_color = "🟢"
                    status_text = "GENUINE"
                    interpretation = "The Siamese Network detected high feature similarity between the signatures."
                else:
                    status_color = "🔴"
                    status_text = "FORGED"
                    interpretation = "The Siamese Network detected significant feature differences between the signatures."
                
                st.markdown(f"- **Verification Status:** {status_color} {status_text}")
                st.markdown(f"- **Similarity Score:** {similarity_percentage:.2f}%")
                st.markdown(f"- **Confidence Level:** {confidence_percentage:.1f}%")
                st.markdown(f"- **Distance from Threshold:** {abs(distance - THRESHOLD):.4f}")
                
            with analysis_col2:
                st.markdown("**Interpretation:**")
                st.info(interpretation)
                
                # Risk Assessment
                if is_genuine:
                    if confidence_percentage > 85:
                        risk_level = "Very Low Risk"
                        risk_color = "🟢"
                    elif confidence_percentage > 70:
                        risk_level = "Low Risk"
                        risk_color = "🟡"
                    else:
                        risk_level = "Moderate Risk"
                        risk_color = "🟠"
                else:
                    if confidence_percentage < 30:
                        risk_level = "High Forgery Risk"
                        risk_color = "🔴"
                    elif confidence_percentage < 50:
                        risk_level = "Moderate Forgery Risk"
                        risk_color = "🟠"
                    else:
                        risk_level = "Low Forgery Risk"
                        risk_color = "🟡"
                
                st.markdown(f"**Risk Assessment:** {risk_color} {risk_level}")
                
                # Recommendation
                if is_genuine and confidence_percentage > 75:
                    st.success("✓ Signature verification passed with high confidence.")
                elif is_genuine and confidence_percentage <= 75:
                    st.warning("⚠ Signature appears genuine but confidence is moderate. Manual review recommended.")
                elif not is_genuine and confidence_percentage < 40:
                    st.error("✗ Strong indication of forgery. Reject signature.")
                else:
                    st.warning("⚠ Possible forgery detected. Manual verification strongly recommended.")

            st.markdown("---")
                        
        except Exception as e:
            st.error(f"❌ Processing Error: {e}")
            st.info("Please ensure both signatures are clearly drawn or uploaded.")
                
        except Exception as e:
            st.error(f"Processing Error: {e}")

# version 3.2           