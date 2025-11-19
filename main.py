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
    layout="wide"
)

# --- CSS TO HIDE DEFAULT STREAMLIT ELEMENTS ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- MODEL AND HELPER FUNCTIONS ---
def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sumSquared = tf.keras.backend.sum(tf.keras.backend.square(featsA - featsB), axis=1, keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sumSquared, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred + 1e-7, 0))
    loss = y_true * square_pred + (1 - y_true) * margin_square
    return tf.reduce_mean(loss)

def contrastive_accuracy(y_true, y_pred, threshold=0.5):
    y_true = tf.cast(y_true, tf.float32)
    predictions = tf.cast(y_pred < threshold, tf.float32)
    correct = tf.equal(predictions, y_true)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

@st.cache_resource
def load_signature_model():
    custom_objects = {
        'euclidean_distance': euclidean_distance,
        'contrastive_loss': contrastive_loss,
        'contrastive_accuracy': contrastive_accuracy
    }
    try:
        # FIX FOR TERMINAL WARNING:
        # 1. Load with compile=False to stop the "metrics not built" warning during load.
        # 2. Manually compile immediately after.
        model = tf.keras.models.load_model('best_siamese_model.h5', custom_objects=custom_objects, compile=False)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[contrastive_accuracy])
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, is_canvas=False):
    """
    Preprocess the signature image to the required format.
    Ensures output is White Ink on Black Background.
    """
    IMG_SIZE = (150, 150) 
    
    # 1. Convert to numpy array and Grayscale
    if is_canvas:
        img = image.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        img = np.array(image.convert('L'))

    # 2. Invert colors if necessary (Ensure White Ink on Black Background)
    if np.mean(img) > 127:
        img = 255 - img

    # 3. Resize
    img = cv2.resize(img, IMG_SIZE)

    # 4. Binarize (Make lines distinct)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 5. Normalize to 0-1
    img = img.astype('float32') / 255.0
    
    # 6. Reshape for Model
    img_final = np.expand_dims(img, axis=-1)
    img_final = np.expand_dims(img_final, axis=0)
    
    return img_final, img

# --- LOAD MODEL ---
model = load_signature_model()

# --- SIDEBAR CONFIGURATION ---
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

# --- SESSION STATE INITIALIZATION ---
if 'ref_drawing_state' not in st.session_state:
    st.session_state.ref_drawing_state = None
if 'ref_image_data' not in st.session_state:
    st.session_state.ref_image_data = None
if 'test_drawing_state' not in st.session_state:
    st.session_state.test_drawing_state = None
if 'test_image_data' not in st.session_state:
    st.session_state.test_image_data = None
if 'ref_uploaded_file' not in st.session_state:
    st.session_state.ref_uploaded_file = None
if 'test_uploaded_file' not in st.session_state:
    st.session_state.test_uploaded_file = None

# State for Clearing Canvases (Fixes the blinking issue)
if 'ref_canvas_key' not in st.session_state:
    st.session_state.ref_canvas_key = 0
if 'test_canvas_key' not in st.session_state:
    st.session_state.test_canvas_key = 0

# --- HELPER: DRAWING TOOLS ---
def render_drawing_tools(key_prefix):
    c1, c2 = st.columns([3, 1])
    with c1:
        stroke_width = st.slider("Stroke Width", 1, 10, 3, key=f"{key_prefix}_width")
    with c2:
        stroke_color = st.color_picker("Color", "#000000", key=f"{key_prefix}_color")
    return stroke_width, stroke_color

# --- MAIN LAYOUT ---
st.title("✒️ Signature Forgery Detection")
st.markdown("Using a Siamese Neural Network to distinguish between genuine and forged signatures.")

OPTIMAL_THRESHOLD = 0.4500

# --- SPLIT VIEW (Removed Expanded View Logic) ---
col1, col2 = st.columns(2)

# --- REFERENCE SIGNATURE COLUMN ---
with col1:
    st.header("Reference (Genuine)")
    ref_tabs = st.tabs(["✏️ Draw", "📤 Upload"])
    
    with ref_tabs[1]:
        ref_file = st.file_uploader("Upload Reference", type=['jpg', 'png', 'jpeg'], key="ref_u")
        if ref_file:
            st.session_state.ref_uploaded_file = ref_file
            st.image(Image.open(ref_file), use_container_width=True)

    with ref_tabs[0]:
        # Toolbar
        stroke_width, stroke_color = render_drawing_tools("ref_small")
        
        # Canvas with Dynamic Key for Clearing
        ref_canvas = st_canvas(
            stroke_width=stroke_width, stroke_color=stroke_color, background_color="#FFFFFF",
            height=400, width=800, 
            drawing_mode="freedraw", 
            key=f"ref_c_{st.session_state.ref_canvas_key}", # Dynamic key
            initial_drawing=st.session_state.ref_drawing_state, 
            display_toolbar=False
        )
        
        if ref_canvas.json_data is not None:
            st.session_state.ref_drawing_state = ref_canvas.json_data
        if ref_canvas.image_data is not None:
            st.session_state.ref_image_data = ref_canvas.image_data.copy()
        
        # Clear Button (Now takes full width)
        if st.button("🗑️ Clear Canvas", key="clr_ref", use_container_width=True):
            st.session_state.ref_canvas_key += 1 # Force Reset
            st.session_state.ref_drawing_state = None
            st.session_state.ref_image_data = None
            st.rerun()

# --- TEST SIGNATURE COLUMN ---
with col2:
    st.header("Test (To Verify)")
    test_tabs = st.tabs(["✏️ Draw", "📤 Upload"])

    with test_tabs[1]:
        test_file = st.file_uploader("Upload Test", type=['jpg', 'png', 'jpeg'], key="test_u")
        if test_file:
            st.session_state.test_uploaded_file = test_file
            st.image(Image.open(test_file), use_container_width=True)
        
    with test_tabs[0]:
        # Toolbar
        stroke_width, stroke_color = render_drawing_tools("test_small")

        # Canvas with Dynamic Key for Clearing
        test_canvas = st_canvas(
            stroke_width=stroke_width, stroke_color=stroke_color, background_color="#FFFFFF",
            height=400, width=800, 
            drawing_mode="freedraw", 
            key=f"test_c_{st.session_state.test_canvas_key}", # Dynamic key
            initial_drawing=st.session_state.test_drawing_state, 
            display_toolbar=False
        )
        
        if test_canvas.json_data is not None:
            st.session_state.test_drawing_state = test_canvas.json_data
        if test_canvas.image_data is not None:
            st.session_state.test_image_data = test_canvas.image_data.copy()

        # Clear Button (Now takes full width)
        if st.button("🗑️ Clear Canvas", key="clr_test", use_container_width=True):
            st.session_state.test_canvas_key += 1 # Force Reset
            st.session_state.test_drawing_state = None
            st.session_state.test_image_data = None
            st.rerun()

st.markdown("---")

# --- TIP SECTION ---
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    ### Instructions:
    1.  **Provide Signatures:** For both the 'Reference' and 'Test' sections, you can either **draw** the signature or **upload** an image.
    2.  **Drawing:** Use the controls to change stroke width or color.
    3.  **Verify:** Once both signatures are provided, click the "Verify Signatures" button.
    """)

st.markdown("---")

# --- VERIFICATION LOGIC ---
if st.button("🔍 Verify Signatures", type="primary", use_container_width=True):
    # 1. Prepare Reference Input
    ref_sig, ref_is_canvas = None, False
    if st.session_state.ref_image_data is not None and np.sum(st.session_state.ref_image_data) > 0:
        ref_sig = st.session_state.ref_image_data
        ref_is_canvas = True
    elif st.session_state.ref_uploaded_file:
        ref_sig = Image.open(st.session_state.ref_uploaded_file)

    # 2. Prepare Test Input
    test_sig, test_is_canvas = None, False
    if st.session_state.test_image_data is not None and np.sum(st.session_state.test_image_data) > 0:
        test_sig = st.session_state.test_image_data
        test_is_canvas = True
    elif st.session_state.test_uploaded_file:
        test_sig = Image.open(st.session_state.test_uploaded_file)

    # 3. Validation Check
    if ref_sig is None or test_sig is None:
        st.error("⚠️ **Action Required:** Please provide BOTH a Reference Signature and a Test Signature to proceed.")
    elif not model:
        st.error("❌ **System Error:** Model is not loaded. Check file path.")
    else:
        # 4. Processing & Prediction
        try:
            ref_processed, ref_viz = preprocess_image(ref_sig, is_canvas=ref_is_canvas)
            test_processed, test_viz = preprocess_image(test_sig, is_canvas=test_is_canvas)

            distance = model.predict([ref_processed, test_processed], verbose=0)[0][0]
            is_match = distance < OPTIMAL_THRESHOLD

            # --- RESULTS ---
            st.markdown("### 📊 Results")
            
            with st.expander("👁️ View What the Model Sees (Debugging)"):
                d_col1, d_col2 = st.columns(2)
                d_col1.image(ref_viz, caption="Processed Reference", clamp=True, width=150)
                d_col2.image(test_viz, caption="Processed Test", clamp=True, width=150)

            r_col1, r_col2 = st.columns([2, 1])
            
            with r_col1:
                if is_match:
                    st.success("✅ **GENUINE SIGNATURE**")
                    st.write("The signatures are statistically similar.")
                else:
                    st.error("❌ **FORGED SIGNATURE**")
                    st.write("The signatures are statistically different.")
                
                # --- CONFIDENCE SCORE ---
                confidence = max(0, 100 * (1 - (distance / (OPTIMAL_THRESHOLD * 1.5))))
                st.progress(int(confidence))
                st.markdown(f"**Match Confidence:** `{confidence:.2f}%`")
            
            with r_col2:
                st.metric("Dissimilarity Score", f"{distance:.4f}")
                st.caption(f"Threshold: {OPTIMAL_THRESHOLD}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# version 2.3