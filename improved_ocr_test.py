#!/usr/bin/env python3
"""
Improved OCR specifically for handwritten prescriptions
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
import os

def enhance_prescription_image(image_path):
    """Enhanced image preprocessing for handwritten prescriptions"""
    print(f"Enhancing prescription image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast and brightness for faded text
    alpha = 1.5  # Contrast control
    beta = 30    # Brightness control
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(bilateral)
    
    # Multiple thresholding approaches
    # 1. Otsu threshold
    _, thresh1 = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Adaptive threshold - mean
    thresh2 = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 15, 10)
    
    # 3. Adaptive threshold - gaussian
    thresh3 = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 10)
    
    # Combine thresholds
    combined = cv2.bitwise_and(thresh1, thresh2)
    combined = cv2.bitwise_and(combined, thresh3)
    
    # Morphological operations to clean up
    kernel1 = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel1)
    
    kernel2 = np.ones((1,1), np.uint8)
    final = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel2)
    
    # Save enhanced image
    root, ext = os.path.splitext(image_path)
    enhanced_path = f"{root}_enhanced{ext}"
    cv2.imwrite(enhanced_path, final)
    
    return enhanced_path

def extract_with_multiple_configs(image_path):
    """Try multiple OCR configurations optimized for prescriptions"""
    
    configs = [
        # Configuration optimized for printed text
        '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()-: ',
        
        # Configuration for mixed content (printed + handwritten)
        '--oem 3 --psm 4',
        
        # Configuration for sparse text
        '--oem 3 --psm 8',
        
        # Configuration for single uniform block
        '--oem 3 --psm 6',
        
        # Configuration for vertical text
        '--oem 3 --psm 5',
        
        # Raw line configuration
        '--oem 3 --psm 13'
    ]
    
    best_result = ""
    results = []
    
    for i, config in enumerate(configs):
        try:
            text = pytesseract.image_to_string(Image.open(image_path), config=config)
            cleaned_text = text.strip()
            
            # Score based on length and readability
            score = len(cleaned_text)
            # Bonus for prescription-related keywords
            keywords = ['mg', 'ml', 'tablet', 'capsule', 'daily', 'twice', 'morning', 'evening', 'dose']
            for keyword in keywords:
                if keyword.lower() in cleaned_text.lower():
                    score += 50
            
            results.append((config, cleaned_text, score))
            print(f"Config {i+1}: {len(cleaned_text)} chars, score: {score}")
            
            if score > len(best_result):
                best_result = cleaned_text
                
        except Exception as e:
            print(f"Config {i+1} failed: {e}")
    
    return best_result, results

def main():
    image_path = "attached_assets/pres5_1757838184063.jpeg"
    
    print("🏥 Improved Prescription OCR")
    print("=" * 50)
    
    # Test original image
    print("\n1. Testing original image...")
    original_text, _ = extract_with_multiple_configs(image_path)
    
    # Test enhanced image
    print("\n2. Testing enhanced image...")
    enhanced_path = enhance_prescription_image(image_path)
    if enhanced_path:
        enhanced_text, all_results = extract_with_multiple_configs(enhanced_path)
        
        print(f"\n📊 COMPARISON:")
        print(f"Original: {len(original_text)} characters")
        print(f"Enhanced: {len(enhanced_text)} characters")
        
        # Use the better result
        final_text = enhanced_text if len(enhanced_text) > len(original_text) else original_text
        
        print(f"\n📝 BEST RESULT ({len(final_text)} characters):")
        print("=" * 50)
        print(final_text)
        
        # Analyze for prescription content
        lines = final_text.split('\n')
        meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        
        print(f"\n📋 ANALYSIS:")
        print(f"Total lines: {len(lines)}")
        print(f"Meaningful lines: {len(meaningful_lines)}")
        
        # Look for prescription patterns
        prescription_patterns = []
        for line in meaningful_lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['mg', 'ml', 'tablet', 'capsule', 'dose']):
                prescription_patterns.append(line)
        
        if prescription_patterns:
            print(f"\n💊 POTENTIAL PRESCRIPTIONS:")
            for pattern in prescription_patterns:
                print(f"  - {pattern}")
        else:
            print(f"\n⚠️  NO CLEAR PRESCRIPTION PATTERNS FOUND")
            print("This appears to be a handwritten prescription that's difficult to read.")
        
        # Clean up
        if os.path.exists(enhanced_path):
            os.remove(enhanced_path)
    
    else:
        print("❌ Could not enhance image")

if __name__ == "__main__":
    main()