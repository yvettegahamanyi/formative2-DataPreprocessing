#!/usr/bin/env python3
"""
Biometric Transaction Verification System with Product Recommendations
Multi-factor authentication using face and voice recognition
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
import librosa
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
IMAGE_SIZE = (128, 128)
SAMPLE_RATE = 22050
CONFIDENCE_THRESHOLD = 0.80  # 80% confidence required
IMAGE_MODEL_PATH = "./models/rf_image_member_classifier.joblib"
IMAGE_SCALER_PATH = "./models/rf_feature_scaler.joblib"
IMAGE_ENCODER_PATH = "./models/rf_label_encoder.joblib"
VOICE_MODEL_PATH = "./models/voiceprint_model.joblib"
PRODUCT_MODEL_PATH = "./models/product_recommendation_model.joblib"
CUSTOMER_DATA_PATH = "merged_dataset.csv"

# ==================== COLORS FOR TERMINAL ====================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    INFO = '\033[94m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

# ==================== IMAGE FEATURE EXTRACTION ====================
def extract_color_histogram(img, bins=32):
    """Extract color histogram features"""
    features = []
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
    return features

def extract_texture_features(img):
    """Extract texture features using Laplacian"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features = [
        np.mean(laplacian),
        np.std(laplacian),
        np.min(laplacian),
        np.max(laplacian)
    ]
    return features

def extract_shape_features(img):
    """Extract basic shape/edge features"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    features = [
        np.mean(edges),
        np.std(edges),
        np.sum(edges > 0)
    ]
    return features

def extract_image_features(img):
    """Combine all image feature extraction methods"""
    img_resized = cv2.resize(img, IMAGE_SIZE)
    features = []
    features.extend(extract_color_histogram(img_resized))
    features.extend(extract_texture_features(img_resized))
    features.extend(extract_shape_features(img_resized))
    return features

# ==================== AUDIO FEATURE EXTRACTION ====================
def extract_mfcc_features(y, sr, n_mfcc=13):
    """Extract MFCC features"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def extract_spectral_features(y, sr):
    """Extract spectral features"""
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing = librosa.feature.zero_crossing_rate(y)[0]
    
    features = [
        np.mean(spectral_centroids),
        np.std(spectral_centroids),
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(zero_crossing),
        np.std(zero_crossing)
    ]
    return features

def extract_energy_features(y):
    """Extract energy-based features"""
    rms = librosa.feature.rms(y=y)[0]
    features = [
        np.mean(rms),
        np.std(rms),
        np.max(rms),
        np.min(rms)
    ]
    return features

def extract_tempo_features(y, sr):
    """Extract tempo and rhythm features"""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features = [tempo, len(beats)]
    return features

def extract_audio_features(y, sr):
    """Combine all audio feature extraction methods"""
    features = []
    features.extend(extract_mfcc_features(y, sr))
    features.extend(extract_spectral_features(y, sr))
    features.extend(extract_energy_features(y))
    features.extend(extract_tempo_features(y, sr))
    return features

# ==================== MODEL LOADING ====================
def load_models():
    """Load all trained models"""
    try:
        # Load image recognition model
        image_clf = joblib.load(IMAGE_MODEL_PATH)
        image_scaler = joblib.load(IMAGE_SCALER_PATH)
        image_encoder = joblib.load(IMAGE_ENCODER_PATH)
        
        # Load voice recognition model
        voice_data = joblib.load(VOICE_MODEL_PATH)
        voice_model = voice_data['model']
        voice_scaler = voice_data['scaler']
        voice_encoder = voice_data['label_encoder']
        voice_features = voice_data['feature_columns']
        
        # Load product recommendation model
        product_bundle = joblib.load(PRODUCT_MODEL_PATH)
        product_model = product_bundle['model']
        product_scaler = product_bundle['scaler']
        product_encoder = product_bundle['encoder']
        product_features = product_bundle['feature_columns']
        
        # Load customer data
        customer_df = pd.read_csv(CUSTOMER_DATA_PATH)
        
        print_success("All models loaded successfully!")
        return {
            'image_clf': image_clf,
            'image_scaler': image_scaler,
            'image_encoder': image_encoder,
            'voice_model': voice_model,
            'voice_scaler': voice_scaler,
            'voice_encoder': voice_encoder,
            'voice_features': voice_features,
            'product_model': product_model,
            'product_scaler': product_scaler,
            'product_encoder': product_encoder,
            'product_features': product_features,
            'customer_df': customer_df
        }
    except Exception as e:
        print_error(f"Failed to load models: {e}")
        return None

# ==================== FACE VERIFICATION ====================
def verify_face(image_path, models):
    """Verify user identity through face recognition"""
    print_header("STEP 1: FACE VERIFICATION")
    
    try:
        # Load and validate image
        if not os.path.exists(image_path):
            print_error(f"Image file not found: {image_path}")
            return None, 0.0
        
        img = cv2.imread(image_path)
        if img is None:
            print_error("Could not load image. Please check the file format.")
            return None, 0.0
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print_info(f"Image loaded: {os.path.basename(image_path)}")
        
        # Extract features
        print_info("Extracting facial features...")
        features = extract_image_features(img)
        features = np.array(features).reshape(1, -1)
        
        # Scale and predict
        features_scaled = models['image_scaler'].transform(features)
        prediction = models['image_clf'].predict(features_scaled)[0]
        probabilities = models['image_clf'].predict_proba(features_scaled)[0]
        
        predicted_member = models['image_encoder'].inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # Display results
        print("\n" + "-"*70)
        print(f"{Colors.BOLD}Recognition Results:{Colors.END}")
        print(f"  Identified as: {Colors.BOLD}{predicted_member}{Colors.END}")
        print(f"  Confidence: {Colors.BOLD}{confidence*100:.2f}%{Colors.END}")
        
        # Show all confidence scores
        print(f"\n{Colors.BOLD}Confidence Breakdown:{Colors.END}")
        for member, prob in zip(models['image_encoder'].classes_, probabilities):
            bar_length = int(prob * 40)
            bar = "█" * bar_length
            marker = "→" if member == predicted_member else " "
            print(f"  {marker} {member:15s} {bar:40s} {prob*100:5.2f}%")
        print("-"*70)
        
        # Verification decision
        if confidence >= CONFIDENCE_THRESHOLD:
            print_success(f"Face verification PASSED (threshold: {CONFIDENCE_THRESHOLD*100:.0f}%)")
            print_info(f"Ready to call product recommendation model for: {predicted_member}")
            return predicted_member, confidence
        else:
            print_warning(f"Confidence too low! Required: {CONFIDENCE_THRESHOLD*100:.0f}%, Got: {confidence*100:.2f}%")
            print_error("Face verification FAILED")
            return None, confidence
            
    except Exception as e:
        print_error(f"Face verification error: {e}")
        return None, 0.0

# ==================== VOICE VERIFICATION ====================
def verify_voice(audio_path, expected_member, models):
    """Verify user identity through voice recognition"""
    print_header("STEP 2: VOICE VERIFICATION")
    
    try:
        # Load and validate audio
        if not os.path.exists(audio_path):
            print_error(f"Audio file not found: {audio_path}")
            return False, 0.0
        
        print_info(f"Loading audio: {os.path.basename(audio_path)}")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        duration = len(y) / sr
        print_info(f"Audio duration: {duration:.2f} seconds")
        
        # Extract features
        print_info("Extracting voice features...")
        features = extract_audio_features(y, sr)
        
        # Create DataFrame with correct column names
        feature_df = pd.DataFrame([features], columns=models['voice_features'])
        
        # Scale and predict
        features_scaled = models['voice_scaler'].transform(feature_df)
        prediction = models['voice_model'].predict(features_scaled)[0]
        probabilities = models['voice_model'].predict_proba(features_scaled)[0]
        
        predicted_member = models['voice_encoder'].inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # Display results
        print("\n" + "-"*70)
        print(f"{Colors.BOLD}Voice Recognition Results:{Colors.END}")
        print(f"  Identified as: {Colors.BOLD}{predicted_member}{Colors.END}")
        print(f"  Confidence: {Colors.BOLD}{confidence*100:.2f}%{Colors.END}")
        print(f"  Expected: {Colors.BOLD}{expected_member}{Colors.END}")
        
        # Show confidence breakdown
        print(f"\n{Colors.BOLD}Confidence Breakdown:{Colors.END}")
        for member, prob in zip(models['voice_encoder'].classes_, probabilities):
            bar_length = int(prob * 40)
            bar = "█" * bar_length
            marker = "→" if member == predicted_member else " "
            print(f"  {marker} {member:15s} {bar:40s} {prob*100:5.2f}%")
        print("-"*70)
        
        # Verification decision - MUST MATCH FACE
        if predicted_member != expected_member:
            print_error(f"❌ ACCESS DENIED: Voice does not match face!")
            print_error(f"   Face identified: {expected_member}")
            print_error(f"   Voice identified: {predicted_member}")
            print_error("   Security violation detected - Transaction blocked")
            return False, confidence
        elif confidence < CONFIDENCE_THRESHOLD:
            print_warning(f"Confidence too low! Required: {CONFIDENCE_THRESHOLD*100:.0f}%, Got: {confidence*100:.2f}%")
            print_error("Voice verification FAILED")
            return False, confidence
        else:
            print_success(f"Voice verification PASSED")
            print_success(f"Voice matches face identity: {expected_member}")
            return True, confidence
            
    except Exception as e:
        print_error(f"Voice verification error: {e}")
        return False, 0.0

# ==================== PRODUCT RECOMMENDATION ====================
def predict_single_customer(customer_features, models):
    """
    Predict product recommendation for a single customer
    
    Args:
        customer_features: dict with feature values
        models: dictionary containing all models
    
    Returns:
        predicted_category, all_probabilities
    """
    try:
        # Prepare features
        X_input = pd.DataFrame([customer_features], columns=models['product_features'])
        X_scaled = models['product_scaler'].transform(X_input)
        
        # Predict
        prediction = models['product_model'].predict(X_scaled)[0]
        probabilities = models['product_model'].predict_proba(X_scaled)[0]
        
        predicted_category = models['product_encoder'].inverse_transform([prediction])[0]
        
        # Get top 3 recommendations
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_categories = models['product_encoder'].inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        # Display results
        print(f"\n{Colors.BOLD}Customer Profile:{Colors.END}")
        for feature, value in customer_features.items():
            print(f"  {feature}: {Colors.BOLD}{value}{Colors.END}")
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎯 Top Prediction: {predicted_category}{Colors.END}")
        print(f"{Colors.BOLD}   Confidence: {probabilities[prediction]*100:.2f}%{Colors.END}")
        
        print(f"\n{Colors.BOLD}📊 Top 3 Recommendations:{Colors.END}")
        for i, (category, prob) in enumerate(zip(top_categories, top_probs), 1):
            bar_length = int(prob * 40)
            bar = "█" * bar_length
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"  {medal} {i}. {category:20s}")
        return predicted_category, probabilities
        
    except Exception as e:
        print_error(f"Prediction error: {e}")
        return None, None

def get_product_recommendations(customer_profile, models):
    """Get personalized product recommendations for authenticated user"""
    print_header("STEP 3: PRODUCT RECOMMENDATIONS")
    
    try:
        print_success("Generating personalized product recommendations...")
        
        print("\n" + "-"*70)
        print(f"{Colors.BOLD}{Colors.GREEN}🎁 PERSONALIZED PRODUCT RECOMMENDATIONS 🎁{Colors.END}")
        print("-"*70)
        
        # Use predict_single_customer for detailed recommendations
        predicted_category, probabilities = predict_single_customer(customer_profile, models)
        
        print("-"*70)
        
        return predicted_category, probabilities
        
    except Exception as e:
        print_error(f"Product recommendation error: {e}")
        return None, None

# ==================== TRANSACTION FLOW ====================
def run_transaction():
    """Execute complete biometric transaction with product recommendation"""
    print_header("🔐 BIOMETRIC TRANSACTION VERIFICATION SYSTEM 🔐")
    print(f"{Colors.BOLD}Multi-Factor Authentication: Face + Voice{Colors.END}")
    print(f"Confidence Threshold: {Colors.BOLD}{CONFIDENCE_THRESHOLD*100:.0f}%{Colors.END}")
    print(f"Timestamp: {Colors.BOLD}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}\n")
    
    # Load models
    print_info("Loading authentication and recommendation models...")
    models = load_models()
    if models is None:
        print_error("Cannot proceed without models. Exiting.")
        return
    
    print()
    
    # Step 1: Face Verification
    identified_member = None
    face_confidence = 0.0
    
    while True:
        image_path = input(f"{Colors.BOLD}Enter face image path (or 'quit' to exit): {Colors.END}").strip()
        
        if image_path.lower() == 'quit':
            print_info("Transaction cancelled by user.")
            return
        
        identified_member, face_confidence = verify_face(image_path, models)
        
        if identified_member:
            # Face verification passed
            break
        else:
            # Face verification failed
            retry = input(f"\n{Colors.YELLOW}Try another image? (yes/no): {Colors.END}").strip().lower()
            if retry != 'yes':
                print_error("Transaction aborted: Face verification failed")
                return
            print()
    
    # Step 1.5: Collect Customer Profile Information
    print_header("CUSTOMER PROFILE INPUT")
    print(f"{Colors.INFO}Please enter customer profile information for product recommendations:{Colors.END}\n")
    
    customer_profile = {}
    
    # Get required features
    required_features = {
        'engagement_score': 'Engagement Score (0-100)',
        'purchase_interest_score': 'Purchase Interest Score (0-10)',
        'purchase_amount': 'Purchase Amount ($)',
        'customer_rating': 'Customer Rating (0-5)'
    }
    
    # Optional features
    optional_features = {
        'age': 'Age',
        'monthly_visits': 'Monthly Visits',
        'total_transactions': 'Total Transactions'
    }
    
    # Collect required features
    for feature_key, feature_name in required_features.items():
        while True:
            try:
                value = input(f"{Colors.BOLD}  {feature_name}: {Colors.END}").strip()
                customer_profile[feature_key] = float(value)
                break
            except ValueError:
                print_error("  Invalid input. Please enter a numeric value.")
    
    # Collect optional features if they exist in the model
    print(f"\n{Colors.INFO}Optional features (press Enter to skip):{Colors.END}")
    for feature_key, feature_name in optional_features.items():
        if feature_key in models['product_features']:
            value = input(f"{Colors.BOLD}  {feature_name}: {Colors.END}").strip()
            if value:
                try:
                    customer_profile[feature_key] = float(value)
                except ValueError:
                    print_warning(f"  Invalid input for {feature_name}, skipping...")
    
    # Fill in any missing features with defaults
    for feature in models['product_features']:
        if feature not in customer_profile:
            customer_profile[feature] = 0
    
    print_success("Customer profile collected successfully!")
    print(f"\n{Colors.BOLD}Profile Summary:{Colors.END}")
    for feature, value in customer_profile.items():
        if value != 0:  # Only show non-zero values
            print(f"  {feature}: {Colors.BOLD}{value}{Colors.END}")
    
    # Step 2: Voice Verification (MUST MATCH FACE)
    print()
    voice_verified = False
    voice_confidence = 0.0
    
    while True:
        audio_path = input(f"{Colors.BOLD}Enter voice audio path (or 'quit' to exit): {Colors.END}").strip()
        
        if audio_path.lower() == 'quit':
            print_info("Transaction cancelled by user.")
            return
        
        voice_verified, voice_confidence = verify_voice(audio_path, identified_member, models)
        
        if voice_verified:
            # Both verifications passed!
            break
        else:
            # Voice verification failed
            print_error(f"\n⚠️  Security Alert: Voice must match face identity!")
            retry = input(f"\n{Colors.YELLOW}Try another voice sample? (yes/no): {Colors.END}").strip().lower()
            if retry != 'yes':
                print_error("Transaction aborted: Voice verification failed")
                print_header("❌ ACCESS DENIED ❌")
                return
            print()
    
    # Step 3: Product Recommendations (only if authenticated)
    print()
    recommended_product, probabilities = get_product_recommendations(customer_profile, models)
    
    # Final Summary
    print_header("✅ TRANSACTION COMPLETE ✅")
    print(f"{Colors.GREEN}{Colors.BOLD}User Successfully Authenticated!{Colors.END}")
    print(f"\n{Colors.BOLD}Transaction Summary:{Colors.END}")
    print(f"  Member: {Colors.BOLD}{Colors.GREEN}{identified_member}{Colors.END}")
    print(f"  Face Confidence: {Colors.BOLD}{face_confidence*100:.2f}%{Colors.END}")
    print(f"  Voice Confidence: {Colors.BOLD}{voice_confidence*100:.2f}%{Colors.END}")
    print(f"  Combined Security Score: {Colors.BOLD}{((face_confidence + voice_confidence)/2)*100:.2f}%{Colors.END}")
    if recommended_product:
        print(f"  Recommended Product: {Colors.BOLD}{Colors.GREEN}{recommended_product}{Colors.END}")
    print(f"  Timestamp: {Colors.BOLD}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print(f"\n{Colors.GREEN}✓ Transaction approved - Access granted{Colors.END}")
    print(f"{Colors.GREEN}✓ Product recommendations generated{Colors.END}")
    print_header("=" * 70)

# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    try:
        run_transaction()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Transaction interrupted by user.{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)