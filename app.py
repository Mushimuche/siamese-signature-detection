import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

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
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the signature image to the required format."""
    IMG_SIZE = (150, 150)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img = image

    # Invert if needed
    if np.mean(img) < 127:
        img = 255 - img

    # Threshold
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Load model at startup
model = load_signature_model()
OPTIMAL_THRESHOLD = 0.4500

def verify_signatures(ref_image, test_image):
    """Main verification function."""
    if ref_image is None or test_image is None:
        return "⚠️ Please provide both reference and test signatures.", None, None
    
    if model is None:
        return "❌ Model not loaded. Please check if 'best_siamese_model.h5' exists.", None, None
    
    try:
        # Preprocess images
        ref_processed = preprocess_image(ref_image)
        test_processed = preprocess_image(test_image)
        
        # Calculate distance
        distance = model.predict([ref_processed, test_processed], verbose=0)[0][0]
        is_match = distance < OPTIMAL_THRESHOLD
        
        # Create result message
        if is_match:
            result = f"✅ **Signatures Match (Genuine)**\n\n"
        else:
            result = f"❌ **Signatures Do Not Match (Possible Forgery)**\n\n"
        
        result += f"**Distance:** {distance:.4f}\n"
        result += f"**Threshold:** {OPTIMAL_THRESHOLD}\n\n"
        
        confidence = max(0, 100 * (1 - (distance / (OPTIMAL_THRESHOLD * 1.5))))
        result += f"**Match Confidence:** {confidence:.2f}%\n\n"
        result += f"💡 Distances below {OPTIMAL_THRESHOLD} are considered a match."
        
        # Create visualization
        match_color = "green" if is_match else "red"
        
        return result, distance, confidence
        
    except Exception as e:
        return f"❌ An error occurred: {str(e)}", None, None

# --- GRADIO INTERFACE ---
with gr.Blocks(title="Signature Forgery Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ✒️ Signature Forgery Detection
        ### Using a Siamese Neural Network to distinguish between genuine and forged signatures
        
        **Authors:** Khinje Louis P. Curugan, Rui Manuel A. Palabon, Aj Ian L. Resurreccion  
        **Institution:** University of Southeastern Philippines
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Reference Signature (Genuine)")
            ref_input = gr.Image(
                label="Upload or Draw Reference Signature",
                type="numpy",
                tool="sketch",
                height=300
            )
        
        with gr.Column():
            gr.Markdown("### 🔍 Test Signature (To Verify)")
            test_input = gr.Image(
                label="Upload or Draw Test Signature",
                type="numpy",
                tool="sketch",
                height=300
            )
    
    verify_btn = gr.Button("🔍 Verify Signatures", variant="primary", size="lg")
    
    gr.Markdown("---")
    
    with gr.Row():
        result_output = gr.Markdown(label="Result")
    
    with gr.Row():
        distance_output = gr.Number(label="Calculated Distance", precision=4)
        confidence_output = gr.Number(label="Match Confidence (%)", precision=2)
    
    gr.Markdown(
        """
        ---
        ### ℹ️ How to use:
        1. **Provide Reference Signature:** Upload an image or draw the genuine signature
        2. **Provide Test Signature:** Upload an image or draw the signature to verify
        3. **Click "Verify Signatures"** to see the result
        
        **Note:** You can either upload images or use the drawing tool on each image box.
        """
    )
    
    # Set up the verification action
    verify_btn.click(
        fn=verify_signatures,
        inputs=[ref_input, test_input],
        outputs=[result_output, distance_output, confidence_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()