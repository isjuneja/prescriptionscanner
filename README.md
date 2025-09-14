# 🏥 Offline Prescription Scanner

An offline prescription scanner built with Flask, OCR (Tesseract), and Ollama (Gemma3) that can extract and analyze prescription information without requiring internet connectivity.

## 🚀 Features

- **Offline Operation**: Works completely offline once set up
- **OCR Text Extraction**: Uses Tesseract to extract text from prescription images
- **AI Analysis**: Leverages Ollama with Gemma3 model for intelligent prescription parsing
- **Web Interface**: User-friendly web interface for uploading and viewing results
- **Image Preprocessing**: Automatic image enhancement for better OCR accuracy
- **Multiple Formats**: Supports PNG, JPG, JPEG, GIF, BMP, TIFF image formats
- **API Endpoint**: RESTful API for programmatic access

## 📋 Prerequisites

### System Requirements
- Ubuntu 20.04+ (or compatible Linux distribution)
- Python 3.8+
- 4GB+ RAM (recommended for AI model)
- 2GB+ free disk space

### Required System Packages
- Tesseract OCR
- ImageMagick
- Python development headers

## 🛠️ Installation

### 1. Install System Dependencies

```bash
# Update package list
sudo apt update

# Install Tesseract OCR and ImageMagick
sudo apt install tesseract-ocr tesseract-ocr-eng imagemagick

# Install Python development packages
sudo apt install python3-dev python3-pip
```

### 2. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Pull Gemma3 model (this will download ~5GB)
ollama pull gemma2
```

### 3. Install Python Dependencies

```bash
# Install required Python packages
pip install flask pillow opencv-python pytesseract werkzeug requests
```

### 4. Verify Setup

Run the setup verification script:

```bash
python ollama_setup.py
```

This script will:
- Check if Ollama is running
- Verify Gemma3 model is available
- Test the model functionality
- Provide troubleshooting guidance

## 🏃 Running the Application

### 1. Start Ollama (if not already running)

```bash
ollama serve
```

### 2. Start the Flask Application

```bash
python app.py
```

### 3. Access the Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage

### Web Interface

1. **Upload Image**: Click the upload area and select a prescription image
2. **Scan**: Click "Scan Prescription" to process the image
3. **View Results**: See extracted text and AI analysis
4. **Copy Results**: Use the copy button to save results to clipboard

### API Usage

#### Upload and Scan Prescription

```bash
curl -X POST \
  -F "file=@prescription.jpg" \
  http://localhost:5000/api/scan
```

Response:
```json
{
  "extracted_text": "Dr. John Smith\nPatient: Jane Doe\nMetformin 500mg\nTake twice daily with meals",
  "analysis": "{\"patient_name\": \"Jane Doe\", \"doctor_name\": \"Dr. John Smith\", \"medications\": [...]}",
  "status": "success"
}
```

#### Health Check

```bash
curl http://localhost:5000/health
```

## 🔧 Configuration

### Environment Variables

- `SESSION_SECRET`: Flask session secret key (optional, defaults to dev key)

### Ollama Configuration

The application uses Ollama running on `http://localhost:11434` by default. You can modify the `OLLAMA_URL` in `app.py` if needed.

### OCR Configuration

Tesseract OCR is configured with optimal settings for prescription text recognition. You can modify OCR parameters in the `extract_text_ocr()` function.

## 📁 Project Structure

```
prescription-scanner/
├── app.py                 # Main Flask application
├── ollama_setup.py        # Setup verification script
├── templates/             # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Upload page
│   └── results.html      # Results page
├── uploads/              # Temporary file storage
└── README.md             # This file
```

## 🛠️ Troubleshooting

### Ollama Issues

- **Service not running**: Run `ollama serve` in a terminal
- **Model not found**: Pull the model with `ollama pull gemma2`
- **Connection refused**: Check if Ollama is running on port 11434

### OCR Issues

- **Poor text extraction**: Ensure image is clear, well-lit, and high resolution
- **Tesseract not found**: Install with `sudo apt install tesseract-ocr`
- **Language pack missing**: Install with `sudo apt install tesseract-ocr-eng`

### Flask Issues

- **Port 5000 in use**: Change port in `app.py` or kill existing process
- **Template not found**: Ensure `templates/` directory exists with HTML files
- **Upload fails**: Check file permissions on `uploads/` directory

## 🔒 Privacy & Security

- **Local Processing**: All data processing happens locally on your machine
- **No Internet Required**: Once set up, works completely offline
- **Temporary Storage**: Uploaded images are automatically deleted after processing
- **No Data Logging**: No prescription data is logged or stored permanently

## ⚡ Performance Tips

1. **Image Quality**: Use high-resolution, well-lit images for better OCR results
2. **Image Format**: PNG and TIFF formats often work better than JPEG
3. **Preprocessing**: The application automatically enhances images for OCR
4. **Memory**: Ensure sufficient RAM for AI model operation (4GB+ recommended)

## 🤝 Contributing

This is a standalone offline application. Feel free to modify and enhance based on your needs.

## ⚠️ Disclaimer

This tool is for educational and assistive purposes only. Always verify prescription information with healthcare professionals. Do not rely solely on automated extraction for medical decisions.

## 📄 License

This project is provided as-is for educational purposes. Please ensure compliance with local healthcare data regulations.