import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import requests
import json

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret-key')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
OLLAMA_URL = 'http://localhost:11434/api/generate'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image to improve OCR accuracy"""
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError("Could not load image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold for better text detection
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    
    # Apply Otsu threshold as backup
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine both thresholding methods
    combined = cv2.bitwise_and(adaptive_thresh, otsu_thresh)
    
    # Morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Save processed image with proper extension handling
    name, ext = os.path.splitext(image_path)
    processed_path = f"{name}_processed{ext}"
    cv2.imwrite(processed_path, processed)
    
    return processed_path

def extract_text_ocr(image_path):
    """Extract text from image using OCR with multiple configurations"""
    try:
        # Try multiple OCR configurations for better results
        configs = [
            '--oem 3 --psm 6',  # Default
            '--oem 3 --psm 4',  # Single column  
            '--oem 3 --psm 11', # Sparse text
            '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()-: '
        ]
        
        best_text = ""
        
        # Test original image first
        for config in configs:
            try:
                text = pytesseract.image_to_string(Image.open(image_path), config=config)
                text = text.strip()
                if len(text) > len(best_text):
                    best_text = text
            except:
                continue
        
        # Also try preprocessed image
        try:
            processed_path = preprocess_image(image_path)
            if processed_path:
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(Image.open(processed_path), config=config)
                        text = text.strip()
                        if len(text) > len(best_text):
                            best_text = text
                    except:
                        continue
                
                # Clean up processed image
                if os.path.exists(processed_path):
                    os.remove(processed_path)
        except:
            pass
        
        return best_text
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""

def analyze_with_ollama(text):
    """Analyze extracted text using Ollama Gemma3 model"""
    try:
        prompt = """
        You are a medical prescription analyzer. Analyze this OCR-extracted text from a prescription image and extract key information.

        IMPORTANT: This text may contain OCR errors, misspellings, and poor formatting. Look for patterns and context clues.

        Text to analyze:
        """ + text + """

        Extract and return ONLY a valid JSON object with these exact fields:
        - patient_name: Extract patient's name (look for Patient, Name, Mr, Mrs, Ms followed by a name)
        - doctor_name: Extract doctor's name (look for Dr, Doctor, medical credentials like MD, MBBS)
        - medications: Array of objects with name, dosage, frequency (look for drug names followed by mg/ml/tablet, then dosing instructions like twice daily, morning, evening)
        - instructions: General instructions or notes (look for Instructions, Note, Sig, special directions)
        - date: Prescription date (look for date patterns like DD/MM/YYYY, MM-DD-YYYY)
        - pharmacy: Pharmacy name/address if mentioned

        MEDICATION DETECTION TIPS:
        - Look for common drug suffixes: -in, -ol, -ide, -ine, -one, -ate
        - Look for dosage patterns: number + mg/ml/tablet/capsule
        - Look for frequency words: daily, twice, morning, evening, before/after meals
        - Common medications: Paracetamol, Ibuprofen, Amoxicillin, Metformin, etc.

        Return ONLY a valid JSON object with patient_name, doctor_name, medications array, instructions, date, and pharmacy fields.
        """
        
        payload = {
            "model": "gemma3:27b-it-qat",
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            raw_response = result.get('response', '')
            
            # Try to parse the response as JSON
            try:
                # Clean up response (remove any markdown formatting)
                cleaned_response = raw_response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                parsed_json = json.loads(cleaned_response)
                return json.dumps(parsed_json, indent=2)
            except json.JSONDecodeError:
                # Return structured error if JSON parsing fails
                return json.dumps({
                    "error": "Could not parse structured data",
                    "raw_analysis": raw_response,
                    "patient_name": None,
                    "doctor_name": None,
                    "medications": [],
                    "instructions": raw_response,
                    "date": None,
                    "pharmacy": None
                }, indent=2)
        else:
            return json.dumps({"error": "Could not connect to Ollama model"}, indent=2)
            
    except requests.exceptions.RequestException:
        return json.dumps({"error": "Ollama service not available. Please ensure Ollama is running locally."}, indent=2)
    except Exception as e:
        return json.dumps({"error": "Error analyzing text: " + str(e)}, indent=2)

@app.route('/')
def index():
    """Main page with file upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and file.filename and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text using OCR
            extracted_text = extract_text_ocr(filepath)
            
            if not extracted_text:
                flash('No text found in the image. Please try a clearer image.')
                return redirect(url_for('index'))
            
            # Analyze with Ollama
            analysis_result = analyze_with_ollama(extracted_text)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('results.html', 
                                 extracted_text=extracted_text, 
                                 analysis=analysis_result)
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload an image file.')
    return redirect(url_for('index'))

@app.route('/api/scan', methods=['POST'])
def api_scan():
    """API endpoint for prescription scanning"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Extract text
        extracted_text = extract_text_ocr(filepath)
        
        # Analyze with Ollama
        analysis_result = analyze_with_ollama(extracted_text)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'extracted_text': extracted_text,
            'analysis': analysis_result,
            'status': 'success'
        })
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'prescription-scanner'})

if __name__ == '__main__':
    # Use environment variable to control debug mode, default to False for production
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)