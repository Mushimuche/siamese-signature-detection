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
    """Load the pre-trained Siamese network model."""
    custom_objects = {
        'euclidean_distance': euclidean_distance,
        'contrastive_loss': contrastive_loss,
        'contrastive_accuracy': contrastive_accuracy
    }
    try:
        model = tf.keras.models.load_model('best_siamese_model.h5', custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the 'best_siamese_model.h5' file is in the same directory as this script.")
        return None

def preprocess_image(image, is_canvas=False):
    """Preprocess the signature image to the required format."""
    IMG_SIZE = (150, 150)
    
    if is_canvas:
        img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    else:
        img = np.array(image.convert('L'))

    if np.mean(img) < 127:
        img = 255 - img

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# --- APPLICATION UI ---
st.title("✒️ Signature Forgery Detection")
st.markdown("Using a Siamese Neural Network to distinguish between genuine and forged signatures.")

model = load_signature_model()

if model:
    st.sidebar.success("Model loaded successfully!")

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

OPTIMAL_THRESHOLD = 0.4500

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

col1, col2 = st.columns(2)

# --- REFERENCE SIGNATURE COLUMN ---
with col1:
    st.header("Reference Signature")
    st.markdown("Provide the **genuine** signature here.")
    
    ref_tabs = st.tabs(["✏️ Draw", "📤 Upload"])
    
    with ref_tabs[1]:
        ref_uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=['jpg', 'png'], key="ref_uploader")
        if ref_uploaded_file:
            st.session_state.ref_uploaded_file = ref_uploaded_file
            st.image(Image.open(ref_uploaded_file), caption="Uploaded Reference Signature", use_container_width=True)

    with ref_tabs[0]:
        st.markdown("**Draw your signature below:**")
        ref_canvas_result = st_canvas(
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=200,
            width=400,
            drawing_mode="freedraw",
            key="ref_canvas",
            initial_drawing=st.session_state.ref_drawing_state,
            # FIX: Hide the built-in toolbar to avoid confusion
            display_toolbar=False 
        )
        
        if ref_canvas_result.json_data is not None:
            st.session_state.ref_drawing_state = ref_canvas_result.json_data
        if ref_canvas_result.image_data is not None:
            st.session_state.ref_image_data = ref_canvas_result.image_data.copy()
        
        # FIX: Re-introduce the reliable button to clear the canvas state
        if st.button("Clear Reference Canvas", key="clear_ref"):
            st.session_state.ref_drawing_state = None
            st.session_state.ref_image_data = None
            st.rerun()

# --- TEST SIGNATURE COLUMN ---
with col2:
    st.header("Test Signature")
    st.markdown("Provide the signature to be **verified**.")

    test_tabs = st.tabs(["✏️ Draw", "📤 Upload"])

    with test_tabs[1]:
        test_uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=['jpg', 'png'], key="test_uploader")
        if test_uploaded_file:
            st.session_state.test_uploaded_file = test_uploaded_file
            st.image(Image.open(test_uploaded_file), caption="Uploaded Test Signature", use_container_width=True)
        
    with test_tabs[0]:
        st.markdown("**Draw your signature below:**")
        test_canvas_result = st_canvas(
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=200,
            width=400,
            drawing_mode="freedraw",
            key="test_canvas",
            initial_drawing=st.session_state.test_drawing_state,
            # FIX: Hide the built-in toolbar to avoid confusion
            display_toolbar=False
        )
        
        if test_canvas_result.json_data is not None:
            st.session_state.test_drawing_state = test_canvas_result.json_data
        if test_canvas_result.image_data is not None:
            st.session_state.test_image_data = test_canvas_result.image_data.copy()

        # FIX: Re-introduce the reliable button to clear the canvas state
        if st.button("Clear Test Canvas", key="clear_test"):
            st.session_state.test_drawing_state = None
            st.session_state.test_image_data = None
            st.rerun()

st.markdown("---")

with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    ### Instructions:
    1.  **Provide Signatures:** For both the 'Reference' and 'Test' sections, you can either **draw** the signature or **upload** an image.
    2.  **Verify:** Once both signatures are provided, click the "Verify Signatures" button.
    """)

st.markdown("---")

# --- VERIFICATION LOGIC ---
if st.button("🔍 Verify Signatures", type="primary", use_container_width=True):
    ref_sig, test_sig = None, None
    ref_is_canvas, test_is_canvas = False, False

    if st.session_state.ref_image_data is not None and np.sum(st.session_state.ref_image_data) > 0:
        ref_sig = st.session_state.ref_image_data
        ref_is_canvas = True
    elif st.session_state.ref_uploaded_file is not None:
        ref_sig = Image.open(st.session_state.ref_uploaded_file)

    if st.session_state.test_image_data is not None and np.sum(st.session_state.test_image_data) > 0:
        test_sig = st.session_state.test_image_data
        test_is_canvas = True
    elif st.session_state.test_uploaded_file is not None:
        test_sig = Image.open(st.session_state.test_uploaded_file)

    if ref_sig is None or test_sig is None:
        st.error("⚠️ Please provide both a reference and a test signature.")
    elif not model:
        st.error("❌ Model not loaded.")
    else:
        with st.spinner("🔄 Analyzing signatures..."):
            try:
                ref_processed = preprocess_image(ref_sig, is_canvas=ref_is_canvas)
                test_processed = preprocess_image(test_sig, is_canvas=test_is_canvas)

                distance = model.predict([ref_processed, test_processed], verbose=0)[0][0]
                is_match = distance < OPTIMAL_THRESHOLD

                st.markdown("---")
                st.subheader("📊 Verification Result")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if is_match:
                        st.success("✅ **Signatures Match (Genuine)**")
                    else:
                        st.error("❌ **Signatures Do Not Match (Possible Forgery)**")
                
                with col_res2:
                    st.metric(
                        label="Calculated Distance", 
                        value=f"{distance:.4f}",
                        help=f"A lower distance means the signatures are more similar. The threshold is {OPTIMAL_THRESHOLD}.",
                        delta=f"{'Lower' if is_match else 'Higher'} than threshold",
                        delta_color="inverse" if is_match else "normal"
                    )
                
                confidence = max(0, 100 * (1 - (distance / (OPTIMAL_THRESHOLD * 1.5))))
                st.progress(int(confidence))
                st.markdown(f"**Match Confidence:** `{confidence:.2f}%`")
                
                st.info(f"💡 The model calculated a distance of **{distance:.4f}**. Distances below **{OPTIMAL_THRESHOLD}** are considered a match.")
                
            except Exception as e:
                st.error(f"❌ An error occurred during verification: {str(e)}")

# version 1.7