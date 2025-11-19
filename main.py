import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Signature Forgery Detection",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TO HIDE DEFAULT STREAMLIT ELEMENTS ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            [data-testid="stSidebar"] { background-color: #1e1e1e; }
            [data-testid="stSidebar"] > div:first-child { background-color: #1e1e1e; }
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
        model = tf.keras.models.load_model('best_siamese_model.h5', custom_objects=custom_objects, compile=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[contrastive_accuracy])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- PREPROCESSING LOGIC ---

def preprocess_canvas_image(image):
    """
    Simple preprocessing for digital drawings (Canvas).
    (Unchanged as requested)
    """
    IMG_SIZE = (150, 150)
    
    if image.shape[-1] == 4:
        bg = np.ones_like(image[:, :, :3]) * 255
        alpha = image[:, :, 3] / 255.0
        for c in range(3):
            bg[:, :, c] = alpha * image[:, :, c] + (1 - alpha) * 255
        img = cv2.cvtColor(bg.astype('uint8'), cv2.COLOR_RGB2GRAY)
    else:
        img = image.astype('uint8')
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if np.mean(img) > 127:
        img = 255 - img

    img = cv2.resize(img, IMG_SIZE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    img_final = img.astype('float32') / 255.0
    img_final = np.expand_dims(img_final, axis=-1)
    img_final = np.expand_dims(img_final, axis=0)
    
    return img_final, img

def preprocess_uploaded_image(image):
    """
    Robust preprocessing for Uploaded Images using Morphological Operations + Otsu.
    """
    IMG_SIZE = (150, 150)
    
    # 1. Convert to Numpy Grayscale
    if isinstance(image, Image.Image):
        img = np.array(image.convert('L'))
    else:
        img = image.astype('uint8')

    # 2. Gaussian Blur (Remove noise/texture)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Otsu's Binarization + Inversion
    # We use binary_inv + Otsu to handle black ink on white paper automatically
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Morphological Closing (Connect broken lines for better bounding box)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5. Smart Cropping (Find Bounding Box of Ink)
    coords = cv2.findNonZero(closed)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Ensure crop isn't tiny noise
        if w > 15 and h > 15:
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(binary.shape[1] - x, w + 2*pad)
            h = min(binary.shape[0] - y, h + 2*pad)
            
            # Crop the BINARY image (sharp edges), not the closed one
            binary = binary[y:y+h, x:x+w]

    # 6. Resize
    if binary.size == 0:
        binary = np.zeros(IMG_SIZE, dtype=np.uint8)
    else:
        binary = cv2.resize(binary, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 7. Normalize
    img_final = binary.astype('float32') / 255.0
    img_final = np.expand_dims(img_final, axis=-1)
    img_final = np.expand_dims(img_final, axis=0)
    
    return img_final, binary

def debug_image_info(image, label="Image"):
    """Helper to debug what we're actually receiving"""
    if image is None:
        st.warning(f"{label}: None")
        return
    
    if isinstance(image, np.ndarray):
        st.info(f"{label}: NumPy array, shape={image.shape}, dtype={image.dtype}")
    elif isinstance(image, Image.Image):
        st.info(f"{label}: PIL Image, size={image.size}, mode={image.mode}")
    else:
        st.warning(f"{label}: Unknown type: {type(image)}")

# --- LOAD MODEL ---
model = load_signature_model()

# --- SIDEBAR ---
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

# --- SESSION STATE ---
if 'ref_image' not in st.session_state:
    st.session_state.ref_image = None
if 'test_image' not in st.session_state:
    st.session_state.test_image = None
if 'ref_drawing_state' not in st.session_state:
    st.session_state.ref_drawing_state = None
if 'test_drawing_state' not in st.session_state:
    st.session_state.test_drawing_state = None
if 'ref_uploaded_file' not in st.session_state:
    st.session_state.ref_uploaded_file = None
if 'test_uploaded_file' not in st.session_state:
    st.session_state.test_uploaded_file = None
if 'ref_canvas_key' not in st.session_state:
    st.session_state.ref_canvas_key = 0
if 'test_canvas_key' not in st.session_state:
    st.session_state.test_canvas_key = 0

def render_drawing_tools(key_prefix):
    c1, c2 = st.columns([3, 1])
    with c1:
        stroke_width = st.slider("Stroke Width", 1, 10, 4, key=f"{key_prefix}_width")
    with c2:
        stroke_color = st.color_picker("Color", "#000000", key=f"{key_prefix}_color")
    return stroke_width, stroke_color

# --- MAIN LAYOUT ---
st.title("✒️ Signature Forgery Detection")
st.markdown("Using a Siamese Neural Network to distinguish between genuine and forged signatures.")

OPTIMAL_THRESHOLD = 0.4500

col1, col2 = st.columns(2)

# --- REFERENCE COLUMN ---
with col1:
    st.header("Reference (Genuine)")
    ref_tabs = st.tabs(["✏️ Draw", "📤 Upload"])
    
    with ref_tabs[1]:
        ref_file = st.file_uploader("Upload Reference", type=['jpg', 'png', 'jpeg'], key="ref_u")
        if ref_file is not None:
            try:
                # Reset file pointer to beginning
                ref_file.seek(0)
                # Read image into memory
                uploaded_image = Image.open(ref_file)
                # Convert to RGB if necessary (handles various formats)
                if uploaded_image.mode not in ('RGB', 'L'):
                    uploaded_image = uploaded_image.convert('RGB')
                
                # Store uploaded file info
                st.session_state.ref_uploaded_file = ref_file
                st.session_state.ref_image = uploaded_image.copy()  # Store a copy
                
                # Clear canvas state when uploading
                st.session_state.ref_drawing_state = None
                
                # Display the uploaded image
                st.image(uploaded_image, caption="Uploaded Reference Signature")
                st.success("✅ Reference image uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading reference image: {e}")

    with ref_tabs[0]:
        stroke_width, stroke_color = render_drawing_tools("ref_small")
        ref_canvas = st_canvas(
            stroke_width=stroke_width, stroke_color=stroke_color, background_color="#FFFFFF",
            height=400, width=800, drawing_mode="freedraw", 
            key=f"ref_c_{st.session_state.ref_canvas_key}", 
            initial_drawing=st.session_state.ref_drawing_state, 
            display_toolbar=False
        )
        
        if ref_canvas.json_data is not None:
            st.session_state.ref_drawing_state = ref_canvas.json_data
        if ref_canvas.image_data is not None:
            # Only update if there's actual drawing (not just empty canvas)
            if np.sum(ref_canvas.image_data) > 0:
                st.session_state.ref_image = ref_canvas.image_data.copy()
                st.session_state.ref_uploaded_file = None  # Clear uploaded file when drawing
        
        if st.button("🗑️ Clear Canvas", key="clr_ref"):
            st.session_state.ref_canvas_key += 1
            st.session_state.ref_drawing_state = None
            st.session_state.ref_image = None
            st.rerun()

# --- TEST COLUMN ---
with col2:
    st.header("Test (To Verify)")
    test_tabs = st.tabs(["✏️ Draw", "📤 Upload"])

    with test_tabs[1]:
        test_file = st.file_uploader("Upload Test", type=['jpg', 'png', 'jpeg'], key="test_u")
        if test_file is not None:
            try:
                # Reset file pointer to beginning
                test_file.seek(0)
                # Read image into memory
                uploaded_image = Image.open(test_file)
                # Convert to RGB if necessary (handles various formats)
                if uploaded_image.mode not in ('RGB', 'L'):
                    uploaded_image = uploaded_image.convert('RGB')
                
                # Store uploaded file info
                st.session_state.test_uploaded_file = test_file
                st.session_state.test_image = uploaded_image.copy()  # Store a copy
                
                # Clear canvas state when uploading
                st.session_state.test_drawing_state = None
                
                # Display the uploaded image
                st.image(uploaded_image, caption="Uploaded Test Signature")
                st.success("✅ Test image uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading test image: {e}")
        
    with test_tabs[0]:
        stroke_width, stroke_color = render_drawing_tools("test_small")
        test_canvas = st_canvas(
            stroke_width=stroke_width, stroke_color=stroke_color, background_color="#FFFFFF",
            height=400, width=800, drawing_mode="freedraw", 
            key=f"test_c_{st.session_state.test_canvas_key}", 
            initial_drawing=st.session_state.test_drawing_state, 
            display_toolbar=False
        )
        
        if test_canvas.json_data is not None:
            st.session_state.test_drawing_state = test_canvas.json_data
        if test_canvas.image_data is not None:
            # Only update if there's actual drawing (not just empty canvas)
            if np.sum(test_canvas.image_data) > 0:
                st.session_state.test_image = test_canvas.image_data.copy()
                st.session_state.test_uploaded_file = None  # Clear uploaded file when drawing

        if st.button("🗑️ Clear Canvas", key="clr_test"):
            st.session_state.test_canvas_key += 1
            st.session_state.test_drawing_state = None
            st.session_state.test_image = None
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
if st.button("🔍 Verify Signatures", type="primary"):
    ref_sig = st.session_state.ref_image
    test_sig = st.session_state.test_image
    
    # DEBUG: Show what we received
    with st.expander("🔧 Debug Information", expanded=False):
        debug_image_info(ref_sig, "Reference Image")
        debug_image_info(test_sig, "Test Image")
    
    if ref_sig is None or test_sig is None:
        st.error("⚠️ **Action Required:** Please provide BOTH a Reference Signature and a Test Signature.")
    elif not model:
        st.error("❌ **System Error:** Model is not loaded.")
    else:
        try:
            # Determine Source (Canvas = Numpy Array, Upload = PIL Image)
            ref_is_canvas = isinstance(ref_sig, np.ndarray)
            test_is_canvas = isinstance(test_sig, np.ndarray)

            st.info(f"Reference is from: {'Canvas (Drawing)' if ref_is_canvas else 'Uploaded Image'}")
            st.info(f"Test is from: {'Canvas (Drawing)' if test_is_canvas else 'Uploaded Image'}")

            # --- SELECT PREPROCESSING METHOD ---
            if ref_is_canvas:
                ref_processed, ref_viz = preprocess_canvas_image(ref_sig)
            else:
                ref_processed, ref_viz = preprocess_uploaded_image(ref_sig)

            if test_is_canvas:
                test_processed, test_viz = preprocess_canvas_image(test_sig)
            else:
                test_processed, test_viz = preprocess_uploaded_image(test_sig)

            # Show processed images BEFORE prediction
            st.markdown("### 👁️ Processed Images (What the Model Sees)")
            proc_col1, proc_col2 = st.columns(2)
            with proc_col1:
                st.image(ref_viz, caption="Processed Reference", clamp=True, width=250)
                
            with proc_col2:
                st.image(test_viz, caption="Processed Test", clamp=True, width=250)
            st.caption("These images show how the app 'sees' your signature after preprocessing. If these images are black or empty, the cropping failed. Try drawing thicker lines or uploading a clearer image.")    

            # Predict
            distance = model.predict([ref_processed, test_processed], verbose=0)[0][0]
            is_match = distance < OPTIMAL_THRESHOLD

            # --- RESULTS ---
            st.markdown("### 📊 Results")

            r_col1, r_col2 = st.columns([2, 1])
            
            with r_col1:
                if is_match:
                    st.success("✅ **GENUINE SIGNATURE**")
                    st.write("The signatures are statistically similar.")
                else:
                    st.error("❌ **FORGED SIGNATURE**")
                    st.write("The signatures are statistically different.")
                
                confidence = max(0, 100 * (1 - (distance / (OPTIMAL_THRESHOLD * 1.5))))
                st.progress(int(confidence))
                st.markdown(f"**Match Confidence:** `{confidence:.2f}%`")
            
            with r_col2:
                st.metric("Dissimilarity Score", f"{distance:.4f}")
                st.caption(f"Threshold: {OPTIMAL_THRESHOLD}")

        except Exception as e:
            import traceback
            st.error(f"An unexpected error occurred: {e}")
            st.code(traceback.format_exc())

# version 3.0