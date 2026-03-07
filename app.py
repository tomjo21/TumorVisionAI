from dotenv import load_dotenv
import os
load_dotenv()

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import tensorflow as tf
import os
import io
import base64
from werkzeug.utils import secure_filename
import traceback
from utils import load_models, preprocess_image_2d, preprocess_volume_3d, calculate_tumor_metrics, generate_slice_visualization, predict_2d_model, predict_3d_model

app = Flask(__name__)
# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['GOOGLE_API_KEY'] = os.getenv('GOOGLE_PLACES_API_KEY')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models on startup
print("Loading models...")
models = load_models()
print("Models loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/segment')
def segment():
    return render_template('segment.html')

@app.route('/api/predict_2d', methods=['POST'])
def predict_2d():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process and predict (predict function handles preprocessing)
            prediction = predict_2d_model(models['2d'], filepath)
            print(f"DEBUG: 2D Prediction Output: {prediction}, Shape: {prediction.shape}, Type: {type(prediction)}")
            
            # Multi-class handling
            # Model returns raw logits (no softmax activation in last layer)
            logits = prediction[0]
            probs = tf.nn.softmax(logits).numpy().tolist()
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            
            # Standard Brain Tumor Labels (Alphabetical/Common):
            # 0: Glioma, 1: Meningioma, 2: No Tumor, 3: Pituitary
            
            LABELS = ['Pituitary', 'No Tumor', 'Meningioma', 'Glioma']
            label = LABELS[class_idx] if class_idx < len(LABELS) else f"Class {class_idx}"
            
            print(f"DEBUG: Parsed Class: {class_idx}, Label: {label}, Conf: {confidence}")
            
            result = {
                'prediction': probs,
                'class_index': class_idx,
                'confidence': confidence,
                'label': label
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/predict_3d', methods=['POST'])
def predict_3d():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 1. Predict
        segmentation = predict_3d_model(models['3d'], filepath)
        
        # 2. Extract voxel spacing from original file
        try:
            import nibabel as nib 
            img_obj = nib.load(filepath)
            dx, dy, dz = img_obj.header.get_zooms()
            voxel_spacing = (dx, dy, dz)
        except:
            voxel_spacing = (1.0, 1.0, 1.0)

        # 3. Metrics
        metrics = calculate_tumor_metrics(segmentation, voxel_spacing=voxel_spacing)
        
        # 4. Visualization
        processed_volume = preprocess_volume_3d(filepath)
        # Adapt for 3-channel models
        if models['3d'].input_shape[-1] == 3 and processed_volume.shape[-1] == 1:
            processed_volume = np.repeat(processed_volume, 3, axis=-1)
            
        slice_img_base64, max_slice = generate_slice_visualization(processed_volume, segmentation)
        
        return jsonify({
            'metrics': metrics,
            'image': slice_img_base64,
            'slice_index': int(max_slice),
            'message': 'Segmentation complete'
        })
    except Exception as e:
        print(f"3D Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/mesh', methods=['POST'])
def api_mesh():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File required'}), 400
            
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Custom Prediction Pipeline for Mesh Generation (need access to volume)
        from utils import preprocess_volume_3d
        
        model = models['3d']
        input_shape = model.input_shape
        target_shape = None
        if len(input_shape) == 5:
            target_shape = input_shape[1:4]
            
        # 1. Preprocess
        processed_volume = preprocess_volume_3d(filepath, target_shape=target_shape)
        
        # 2. Channel Adapt
        if len(input_shape) == 5:
             if input_shape[-1] == 3 and processed_volume.shape[-1] == 1:
                  processed_volume = np.repeat(processed_volume, 3, axis=-1)
                  
        # 3. Predict
        segmentation = model.predict(processed_volume)
        
        # Get spacing for scaling (convert numpy floats to native python floats)
        try:
            import nibabel as nib
            img_obj = nib.load(filepath)
            spacing = [float(s) for s in img_obj.header.get_zooms()]
        except:
            spacing = [1.0, 1.0, 1.0]
            
        print("DEBUG: Generating Mesh...")
        # segmentation shape: (1, 128, 128, 128, 4)
        # Argmax to get 3D mask
        mask_3d = np.argmax(segmentation[0], axis=-1) # (128, 128, 128)
        
        from utils_mesh import generate_tumor_mesh_obj
        # Remove shell generation to prevent OOM/timeouts
        obj_content = generate_tumor_mesh_obj(mask_3d)
        
        if obj_content is None:
            print("DEBUG: Mesh Generation returned None")
            return jsonify({'error': 'No tumor detected for 3D reconstruction'}), 404
            
        print(f"DEBUG: Mesh Generated, size {len(obj_content)} bytes")
        
        return jsonify({
            'obj': obj_content,
            'spacing': spacing
        })
        
    except Exception as e:
        print(f"Mesh API Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



@app.route('/api/report_2d', methods=['POST'])
def api_report_2d():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File required'}), 400
            
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        prediction = predict_2d_model(models['2d'], filepath)
        
        # Consistent Softmax Handling (Matches api/predict_2d)
        logits = prediction[0]
        probs = tf.nn.softmax(logits).numpy()
        class_idx = int(np.argmax(probs))
        confidence_val = float(probs[class_idx])
        confidence_str = f"{confidence_val * 100:.2f}%"
        
        # Labels MUST match api/predict_2d order!
        # Standard: 0: Pituitary, 1: No Tumor, 2: Meningioma, 3: Glioma
        LABELS = ['Pituitary', 'No Tumor', 'Meningioma', 'Glioma']
        pred_label = LABELS[class_idx] if class_idx < len(LABELS) else f"Class {class_idx}"
        
        # Generate PDF
        from utils_report import generate_pdf_report_2d
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Report_2D_{filename}.pdf")
        generate_pdf_report_2d(pred_label, confidence_str, filepath, filename, pdf_path)
        
        return send_file(pdf_path, as_attachment=True)
        
    except Exception as e:
        print(f"Report 2D Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/report_3d', methods=['POST'])
def api_report_3d():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File required'}), 400
            
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Generating 3D Report for {filename}...")
        
        # Predict
        segmentation = predict_3d_model(models['3d'], filepath)
        
        # Get Spacing
        try:
             import nibabel as nib
             dx, dy, dz = nib.load(filepath).header.get_zooms()
             spacing = (dx, dy, dz)
        except:
             spacing = (1.0, 1.0, 1.0)
             
        metrics = calculate_tumor_metrics(segmentation, voxel_spacing=spacing)
        
        # Generate Visualization Image
        try:
            from utils import generate_slice_visualization
            processed_vol = preprocess_volume_3d(filepath)
            
            # Channel handling
            if models['3d'].input_shape[-1] == 3 and processed_vol.shape[-1] == 1:
                processed_vol = np.repeat(processed_vol, 3, axis=-1)
                
            b64_str, _ = generate_slice_visualization(processed_vol, segmentation)
            header, encoded = b64_str.split(',', 1)
            data = base64.b64decode(encoded)
            
            temp_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_report_img_3d.png')
            with open(temp_img_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(f"Visualization Generation Failed: {e}")
            traceback.print_exc()
            temp_img_path = None # Report will handle None image gracefully
            
        # Generate PDF
        try:
            from utils_report import generate_pdf_report_3d
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Report_3D_{filename}.pdf")
            generate_pdf_report_3d(metrics, temp_img_path, filename, pdf_path)
            
            print(f"PDF Generated at {pdf_path}")
            return send_file(pdf_path, as_attachment=True)
        except Exception as pdf_err:
            print(f"PDF Generation Failed: {pdf_err}")
            traceback.print_exc()
            return jsonify({'error': 'PDF Generation failed after analysis'}), 500
        
    except Exception as e:
        print(f"Report 3D Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/api/comparison', methods=['POST'])
def api_comparison():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No selected files'}), 400
            
        results = []
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 1. Predict
            segmentation = predict_3d_model(models['3d'], filepath)
            
            # 2. Get Spacing
            try:
                import nibabel as nib
                dx, dy, dz = nib.load(filepath).header.get_zooms()
                spacing = (dx, dy, dz)
            except:
                spacing = (1.0, 1.0, 1.0)
                
            # 3. Calculate Metrics
            metrics = calculate_tumor_metrics(segmentation, voxel_spacing=spacing)
            
            results.append({
                'filename': filename,
                'total_volume_cm3': metrics.get('total_volume_cm3', 0),
                'total_volume_mm3': metrics.get('total_volume_mm3', 0),
                'metrics': metrics
            })
            
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Comparison API Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/olmo', methods=['POST'])
def api_chat_olmo():
    import random
    try:
        data = request.json
        message = data.get('message', '').lower().strip('?!., ')

        # -----------------------------------------------
        # PATTERN TABLE: list of ([trigger_phrases], [responses])
        # The FIRST matching pattern wins.
        # -----------------------------------------------
        patterns = [

            # ── Greetings ──────────────────────────────────────────────
            (['hello', 'hi there', 'hey', 'howdy', 'greetings'], [
                "Hello! I'm Olmo, your TumorVision assistant. How can I help you today?",
                "Hey there! Ready to dive into some brain imaging? Ask me anything.",
                "Hi! Whether it's MRI types, tumor classification, or finding specialists — I'm here."
            ]),
            (['how are you', 'how r u', "how's it going", 'you okay'], [
                "Running at full capacity! Ask me anything about your scans or results.",
                "All systems green! What can I help you with today?",
                "I'm doing great — analysing data and ready to help. What's on your mind?"
            ]),
            (['who are you', 'what are you', 'tell me about yourself', 'your name'], [
                "I'm Olmo — TumorVision's built-in AI assistant. I can explain tumor types, scan results, colors in the 3D model, and more.",
                "Name's Olmo. I'm here to help you understand your MRI results and navigate TumorVision.",
            ]),
            (['what can you do', 'your capabilities', 'how can you help', 'what do you know'], [
                "I can help with:\n• Explaining tumor types (Glioma, Meningioma, Pituitary)\n• 3D color meanings\n• MRI sequences & terms\n• Classification & segmentation\n• Treatment options\n• Finding nearby specialists\n\nJust ask!",
            ]),
            (['thank', 'thanks', 'appreciate', 'helpful'], [
                "You're very welcome! Stay curious and take care.",
                "Happy to help! Let me know if there's anything else.",
                "Always here for you. Feel free to ask more anytime!"
            ]),
            (['bye', 'goodbye', 'see you', 'take care'], [
                "Goodbye! Take care of yourself, and come back anytime.",
                "See you! I'll be right here whenever you need me.",
            ]),
            (['joke', 'funny', 'make me laugh', 'humor'], [
                "Why did the MRI scanner win an award? Because it was outstanding in its field — and very attractive!",
                "How does a neurosurgeon greet someone? With a lobe of enthusiasm!",
                "What did the brain tumor say to the surgeon? 'I think you're going to have a lot on your plate today.'",
            ]),

            # ── Tumor Types ────────────────────────────────────────────
            (['glioma', 'glioblastoma', 'gbm', 'glial'], [
                "Gliomas arise from glial cells — the support cells of the brain. They range from Grade I (slow, benign) to Grade IV Glioblastoma (GBM), which is the most aggressive primary brain tumor. Treatment typically involves surgery, radiation, and temozolomide chemotherapy.",
                "Glioma is the most common malignant brain tumor. Grade IV GBM has median survival of ~15 months with treatment. Our AI classifies these with 98.5% validation accuracy.",
            ]),
            (['meningioma', 'meninges', 'meningeal'], [
                "Meningiomas grow from the meninges — the protective membranes around the brain. About 90% are benign and slow-growing. However, they can compress brain tissue and cause serious symptoms depending on their size and location.",
                "Meningioma is actually the most common intracranial tumor (~37%). Most are benign and treated with observation, surgery, or radiosurgery (Gamma Knife).",
            ]),
            (['pituitary', 'adenoma', 'master gland', 'hormonal tumor'], [
                "Pituitary tumors (adenomas) develop in the pituitary gland — your body's 'master gland'. They're usually benign but can disrupt hormonal balance, compress the optic chiasm (causing vision issues), and cause systemic effects like Cushing's disease or acromegaly.",
                "Pituitary adenomas are mostly non-cancerous. Treatment often involves medications like dopamine agonists, or transsphenoidal surgery (through the nasal passage).",
            ]),
            (['tumor', 'tumour', 'brain cancer', 'intracranial', 'neoplasm'], [
                "Brain tumors can be primary (starting in the brain) or secondary (metastases from other cancers). The three types TumorVision classifies are Glioma, Meningioma, and Pituitary Adenoma. Ask me about any specific one!",
                "Not all brain tumors are cancerous. Meningiomas and pituitary adenomas are usually benign, while gliomas can range from benign to highly malignant.",
            ]),
            (['no tumor', 'normal', 'healthy', 'clear scan'], [
                "When the model predicts 'No Tumor', it means no recognizable tumor pattern was found in the MRI slice. This is a good sign, but always confirm with a radiologist for a clinical diagnosis.",
            ]),

            # ── Symptoms ───────────────────────────────────────────────
            (['symptom', 'signs', 'headache', 'seizure', 'vision', 'nausea', 'vomit'], [
                "Common brain tumor symptoms include:\n• Persistent headaches (especially in the morning)\n• Seizures\n• Blurred or double vision\n• Nausea/vomiting\n• Personality or memory changes\n• Weakness on one side of the body\n\nThese depend heavily on the tumor's location.",
                "Symptoms vary by tumor location. Frontal lobe tumors affect personality; parietal lobe affects sensation; occipital affects vision. Always see a neurologist if you're experiencing persistent symptoms.",
            ]),

            # ── Treatment ──────────────────────────────────────────────
            (['treatment', 'therapy', 'surgery', 'radiation', 'chemo', 'chemotherapy', 'gamma knife', 'resection'], [
                "Treatment options depend on tumor type, grade, and location:\n• **Surgery**: Primary approach to remove or debulk the tumor\n• **Radiation**: Targeted radiation (or Gamma Knife radiosurgery)\n• **Chemotherapy**: Often Temozolomide for gliomas\n• **Watch & Wait**: For small, benign tumors like low-grade meningiomas",
                "Gamma Knife radiosurgery beams hundreds of thin radiation rays precisely at the tumor — no surgical cut needed. It's often used for small, deep, or recurrent tumors.",
            ]),

            # ── MRI & Imaging ──────────────────────────────────────────
            (['mri', 'magnetic resonance', 'scan', 'imaging', 'what is mri'], [
                "MRI (Magnetic Resonance Imaging) uses powerful magnets and radio waves to generate detailed images of soft tissue like the brain — without any ionizing radiation. It's the gold standard for brain tumor detection.",
                "MRI is preferred over CT for brain imaging because it offers superior soft tissue contrast. For tumors, doctors often request contrast-enhanced MRI (with gadolinium) to highlight active tumor regions.",
            ]),
            (['t1', 't2', 'flair', 'sequence', 'contrast', 'gadolinium', 'mri type'], [
                "Common MRI sequences used in brain tumor imaging:\n• **T1**: Good for anatomy. Tumors appear dark.\n• **T1+Gd**: After contrast injection — active tumor lights up brightly.\n• **T2**: Fluid appears bright; shows edema well.\n• **FLAIR**: Suppresses CSF; great for detecting infiltration near ventricles.",
            ]),
            (['nifti', 'nii', 'volume', '3d file', 'dicom'], [
                "NIfTI (.nii or .nii.gz) is a medical imaging format that stores 3D volumetric brain data. Our segmentation model takes NIfTI files and generates a full tumor map across all brain slices.",
                "Upload a NIfTI (.nii.gz) file in the Segmentation section. The 3D U-Net will segment tumor regions and generate a rotatable 3D model.",
            ]),

            # ── TumorVision System ─────────────────────────────────────
            (['classification', 'classify', '2d model', 'detect', 'prediction'], [
                "Classification uses our 2D Ensemble model (ResNet + EfficientNet) to predict whether an MRI slice shows Glioma, Meningioma, Pituitary Tumor, or No Tumor. It also generates a Grad-CAM heatmap to show which regions the AI focused on.",
                "The 2D classification model achieves 98.5% validation accuracy. Upload any standard MRI JPG/PNG in the Classification section to get an instant prediction.",
            ]),
            (['segmentation', 'segment', '3d model', 'u-net', '3d scan', 'voxel', 'unet'], [
                "Segmentation is the 3D part of TumorVision. A 3D U-Net processes the full volumetric scan to classify each voxel as tumor, edema, necrotic core, or healthy tissue. The result is a color-coded rotatable 3D model.",
                "The 3D segmentation model achieves a Dice Score of 0.89. Upload a NIfTI file (from an MRI scanner) to see the full tumor map.",
            ]),
            (['comparison', 'longitudinal', 'progression', 'regression', 'growth', 'shrink', 'visit 1', 'visit 2'], [
                "The Comparison tool compares tumor volumes from two different time-point scans. It calculates whether the tumor has progressed (grown) or regressed (shrunk) — extremely useful for monitoring treatment effectiveness.",
                "In the Comparison section, upload two NIfTI files from different dates. TumorVision will calculate exact volume changes in cm³ and show whether the treatment is working.",
            ]),
            (['specialist', 'doctor', 'neurologist', 'oncologist', 'find doctor', 'near me'], [
                "The Specialists section uses your location to find the nearest neurologists and oncologists. Just allow location access or enter a city/area in the search box.",
                "In the Specialists tab, you can search for neurologists and oncologists near you. Results show distance, phone number, ratings, and a direct link to Google Maps.",
            ]),
            (['report', 'pdf', 'download', 'medical report'], [
                "After any classification or segmentation result, you can download a medical PDF report. It includes the AI prediction, confidence score, tumor metrics, and a scan visualization.",
            ]),
            (['confidence', 'accuracy', 'reliable', 'trust', 'how accurate'], [
                "Our 2D classification model was trained on 7,000+ annotated MRI slices and achieves 98.5% validation accuracy. For 3D segmentation, we measure Dice Score (0.89), which is the clinical standard for overlap quality.",
                "The confidence percentage shown is the model's softmax probability for its top prediction. A result above 90% usually indicates a very confident classification — but always validate with a radiologist.",
            ]),

            # ── 3D Colors ──────────────────────────────────────────────
            (['color', 'colour', '3d color', '3d colour', 'what do colors mean', 'color mean'], [
                "In the 3D tumor model:\n• 🔴 **Red** — Enhancing Tumor (most active, blood-hungry zones)\n• 🟡 **Yellow** — Peritumoral Edema (swelling around the tumor)\n• 🔵 **Cyan/Blue** — Necrotic Core (dead tissue at the center)\n\nThese map to the BraTS segmentation standard.",
            ]),
            (['red', 'enhancing', 'active tumor'], [
                "Red in the 3D model represents the Enhancing Tumor region — the most metabolically active part that absorbs contrast agent. Surgeons target this region first.",
            ]),
            (['yellow', 'edema', 'swelling'], [
                "Yellow is Peritumoral Edema — fluid accumulation around the tumor causing brain pressure. It's often what causes headaches and neurological symptoms in patients.",
            ]),
            (['cyan', 'blue', 'necrotic', 'dead tissue'], [
                "Cyan/Blue is the Necrotic Core — dead tissue at the tumor center. This happens when the tumor grows faster than its blood supply. It's common in high-grade gliomas.",
            ]),

            # ── General Medical ────────────────────────────────────────
            (['dice score', 'dice', 'metric', 'iou', 'overlap'], [
                "Dice Score measures how well a predicted segmentation mask overlaps with the ground truth. A score of 1.0 is perfect; our 3D model achieves 0.89 which is clinically competitive.",
            ]),
            (['grad-cam', 'gradcam', 'heatmap', 'explainability', 'xai'], [
                "Grad-CAM (Gradient-weighted Class Activation Mapping) generates a heatmap showing which parts of the MRI the AI paid most attention to. This makes the AI decision transparent and helps clinicians verify the result.",
            ]),
            (['biopsy', 'pathology', 'tissue sample', 'grade'], [
                "A biopsy is the most definitive way to grade a tumor. AI imaging helps narrow down the diagnosis, but a tissue biopsy is required to confirm the grade (I–IV) and molecular markers (like IDH mutation in gliomas).",
            ]),
            (['idh', 'mutation', 'molecular marker', 'genetic'], [
                "IDH (Isocitrate Dehydrogenase) mutation status is a critical prognostic marker for gliomas. IDH-mutant gliomas generally have better outcomes than IDH-wildtype ones. MRI can suggest but not confirm IDH status — biopsy is needed.",
            ]),
            (['survival', 'prognosis', 'life expectancy', 'outcome'], [
                "Prognosis depends on tumor type, grade, location, and patient age:\n• Grade I Glioma: Can be cured with surgery\n• Grade IV GBM: ~15 months median with treatment\n• Meningioma (benign): Excellent with surgery\n• Pituitary Adenoma: Very good with medication or surgery",
            ]),

            # ── Chatbot Meta ───────────────────────────────────────────
            (['help', 'what to ask', 'guide me'], [
                "Try asking me:\n• 'What is Glioma?'\n• 'What do the 3D colors mean?'\n• 'How does segmentation work?'\n• 'What are brain tumor symptoms?'\n• 'How accurate is the model?'\n• 'Tell me a joke' 😄",
            ]),
        ]

        # ── Matching Logic ──────────────────────────────────────────────
        response = ""
        for triggers, replies in patterns:
            if any(trigger in message for trigger in triggers):
                response = random.choice(replies)
                break

        # ── Fallback ────────────────────────────────────────────────────
        if not response:
            if len(message) < 3:
                response = "I'm listening! Try asking about tumor types, MRI sequences, or the 3D colors."
            else:
                fallbacks = [
                    "That's a great question! I'm focused on brain imaging and TumorVision. Try asking about Glioma, MRI sequences, or the 3D segmentation model.",
                    "Hmm, I'm not sure I caught that. You can ask me about classification results, tumor types, or finding a specialist near you.",
                    "I'm best at brain tumor topics! Ask me something like 'What is Meningioma?' or 'How does comparison work?'"
                ]
                response = random.choice(fallbacks)

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/doctors')
def doctors_page():
    return render_template('doctors.html')

@app.route('/api/nearby_doctors')
def api_nearby_doctors():
    try:
        lat = request.args.get('lat')
        lng = request.args.get('lng')
        query = request.args.get('query')
        radius = 20000

        api_key = app.config.get('GOOGLE_API_KEY')
        if not api_key or api_key == 'YOUR_API_KEY_HERE':
            return jsonify({'error': 'Google API Key not configured. Please add it to your environment variables or Space Secrets.'}), 500

        import requests as req_lib
        import concurrent.futures

        places_url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.googleMapsUri,places.types,places.nationalPhoneNumber,places.location,places.websiteUri"
        }

        user_lat = float(lat) if lat else None
        user_lng = float(lng) if lng else None

        location_bias = None
        if not query and user_lat is not None and user_lng is not None:
            location_bias = {
                "circle": {
                    "center": {"latitude": user_lat, "longitude": user_lng},
                    "radius": float(radius)
                }
            }

        # Run separate queries to ensure coverage of all specialist types
        # Two focused queries only — avoid generic hospital queries that return unrelated doctors
        search_queries = [
            ("neurologist", "neuro"),
            ("oncologist cancer specialist", "onco")
        ]
        if query:
            search_queries = [(f"{term} near {query}", tag) for term, tag in search_queries]

        def search_places(term_tag):
            term, tag = term_tag
            payload = {
                "textQuery": term,
                "maxResultCount": 10,
                "rankPreference": "DISTANCE"
            }
            if location_bias:
                payload["locationBias"] = location_bias
            try:
                resp = req_lib.post(places_url, json=payload, headers=headers, timeout=8)
                return [(p, tag) for p in resp.json().get('places', [])]
            except Exception:
                return []

        # Parallel requests
        all_places = []  # list of (place_dict, source_tag)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(search_places, qt): qt for qt in search_queries}
            for future in concurrent.futures.as_completed(futures):
                all_places.extend(future.result())

        # Deduplicate by name+address
        seen = set()
        candidates = []
        for place, source_tag in all_places:
            key = place.get('displayName', {}).get('text', '') + place.get('formattedAddress', '')
            if key in seen:
                continue
            seen.add(key)

            loc = place.get('location', {})
            p_lat = loc.get('latitude')
            p_lng = loc.get('longitude')
            distance = 99999
            if p_lat and p_lng and user_lat is not None and user_lng is not None:
                distance = ((p_lat - user_lat)**2 + (p_lng - user_lng)**2)**0.5

            candidates.append({
                'name': place.get('displayName', {}).get('text', 'N/A'),
                'address': place.get('formattedAddress', 'N/A'),
                'phone': place.get('nationalPhoneNumber', 'No phone listed'),
                'rating': place.get('rating', 'N/A'),
                'link': place.get('googleMapsUri', '#'),
                'website': place.get('websiteUri', '#') if place.get('websiteUri') else '#',
                'types': place.get('types', []),
                'source': source_tag,
                'distance': distance
            })

        # Strict relevance filter:
        # Accept only results with neuro/oncology keywords in name,
        # OR places explicitly typed as neurologist/oncologist/hospital by Google.
        NEURO_ONCO_KEYWORDS = [
            'neuro', 'oncol', 'cancer', 'tumor', 'tumour', 'brain',
            'spine', 'spinal', 'neurosurg', 'chemo', 'hematol', 'onco',
            'neuroscience', 'radiosurg', 'gamma knife'
        ]
        BLOCKED_KEYWORDS = [
            'naturopath', 'ayurved', 'homeo', 'dental', 'dentist',
            'optom', 'ophthal', 'physiother', 'dermatol', 'gynae',
            'gynecol', 'pediatric', 'veterinar', 'ent clinic',
            'urology', 'gastroenter', 'pulmonol', 'cardiolog',
            'general physician', 'general practice'
        ]
        ACCEPTED_TYPES = {'neurologist', 'oncologist'}

        def is_relevant(doc):
            name_lower = doc['name'].lower()
            if any(kw in name_lower for kw in BLOCKED_KEYWORDS):
                return False
            # Must POSITIVELY match: keyword in name OR explicit Google type
            if any(kw in name_lower for kw in NEURO_ONCO_KEYWORDS):
                return True
            if any(t in ACCEPTED_TYPES for t in doc['types']):
                return True
            return False

        results = [d for d in candidates if is_relevant(d)]

        # Sort by distance and return top 10
        results.sort(key=lambda x: x['distance'])
        results = results[:10]

        return jsonify({'doctors': results})

    except Exception as e:
        print(f"Nearby Doctors Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=7860)
