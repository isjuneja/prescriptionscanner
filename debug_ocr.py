#!/usr/bin/env python3
"""
OCR Debug Script
Tests OCR extraction on a specific prescription image to debug issues.
"""

import os
import cv2
import numpy as np
from PIL import Image
import pytesseract

def preprocess_image_debug(image_path):
    """Preprocess image with debugging output"""
    print(f"Processing image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
    
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    print(f"✅ Image loaded successfully: {img.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"✅ Converted to grayscale: {gray.shape}")
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("✅ Applied Gaussian blur")
    
    # Apply multiple thresholding techniques for better text detection
    # Method 1: Otsu threshold
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("✅ Applied Otsu threshold")
    
    # Method 2: Adaptive threshold
    thresh_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    print("✅ Applied adaptive threshold")
    
    # Combine both methods by taking intersection (AND operation)
    combined_thresh = cv2.bitwise_and(thresh_otsu, thresh_adaptive)
    print("✅ Combined thresholds")
    
    # Morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
    print("✅ Applied morphological operations")
    
    # Save processed image for inspection
    root, ext = os.path.splitext(image_path)
    processed_path = f"{root}_processed{ext}"
    cv2.imwrite(processed_path, processed)
    print(f"✅ Saved processed image: {processed_path}")
    
    return processed_path

def extract_text_debug(image_path):
    """Extract text with debugging output"""
    print(f"\n🔍 Starting OCR extraction on: {image_path}")
    
    try:
        # Preprocess image
        processed_path = preprocess_image_debug(image_path)
        
        if not processed_path:
            return ""
        
        # Test different OCR configurations
        configs = [
            '--oem 3 --psm 6',  # Default
            '--oem 3 --psm 4',  # Single column
            '--oem 3 --psm 8',  # Single word
            '--oem 3 --psm 11', # Sparse text
            '--oem 3 --psm 12', # Sparse text with OSD
        ]
        
        best_text = ""
        best_config = ""
        
        for config in configs:
            try:
                print(f"Testing config: {config}")
                text = pytesseract.image_to_string(Image.open(processed_path), config=config)
                text = text.strip()
                
                print(f"  Characters extracted: {len(text)}")
                print(f"  Lines: {len(text.splitlines())}")
                
                if len(text) > len(best_text):
                    best_text = text
                    best_config = config
                    
            except Exception as e:
                print(f"  Error with config {config}: {e}")
        
        print(f"\n✅ Best config: {best_config}")
        print(f"✅ Best result: {len(best_text)} characters")
        
        # Try original image too
        print(f"\n🔍 Testing OCR on original image...")
        try:
            original_text = pytesseract.image_to_string(Image.open(image_path), config='--oem 3 --psm 6')
            original_text = original_text.strip()
            print(f"Original image result: {len(original_text)} characters")
            
            if len(original_text) > len(best_text):
                best_text = original_text
                print("✅ Original image worked better!")
        except Exception as e:
            print(f"Error with original image: {e}")
        
        # Clean up processed image
        if processed_path and os.path.exists(processed_path):
            os.remove(processed_path)
        
        return best_text
        
    except Exception as e:
        print(f"❌ OCR Error: {str(e)}")
        return ""

def main():
    image_path = "attached_assets/pres5_1757838184063.jpeg"
    
    print("🏥 Prescription OCR Debug Tool")
    print("=" * 50)
    
    # Extract text
    extracted_text = extract_text_debug(image_path)
    
    print("\n" + "=" * 50)
    print("📝 EXTRACTED TEXT:")
    print("=" * 50)
    
    if extracted_text:
        print(extracted_text)
        print(f"\n📊 STATISTICS:")
        print(f"Total characters: {len(extracted_text)}")
        print(f"Total lines: {len(extracted_text.splitlines())}")
        print(f"Words: {len(extracted_text.split())}")
        
        # Check for common prescription terms
        prescription_keywords = [
            'mg', 'ml', 'tablet', 'capsule', 'dose', 'daily', 'twice', 'morning', 
            'evening', 'before', 'after', 'meal', 'doctor', 'patient', 'pharmacy',
            'prescription', 'rx', 'sig:', 'qty', 'refill'
        ]
        
        found_keywords = []
        text_lower = extracted_text.lower()
        for keyword in prescription_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        print(f"Prescription keywords found: {found_keywords}")
        
        if not found_keywords:
            print("⚠️  No prescription keywords detected - this might not be a prescription image")
        
    else:
        print("❌ No text extracted!")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check if the image is clear and well-lit")
        print("2. Ensure text is not too small or blurry")
        print("3. Try a different image format (PNG works better than JPEG)")
        print("4. Make sure the prescription is not handwritten")

if __name__ == "__main__":
    main()