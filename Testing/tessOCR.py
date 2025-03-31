import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import sys
import json
import cv2
import numpy as np
import os

# Load SSD model for text detection
net = cv2.dnn.readNet("C:\\xampp\\htdocs\\Website\\frozen_east_text_detection.pb")

def normalize_units(text):
    """Normalize units in the OCR text."""
    replacements = {
        r'\b(\d+)\s*m9\b': r'\1 mg',  # Fix 'm9' -> 'mg'
        r'\b(\d+)\s*9\b': r'\1 g',    # Fix '9' -> 'g'
        r'\b(\d+)\s*ozz\b': r'\1 oz'  # Fix 'ozz' -> 'oz'
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def preprocess_image(image_path):
    """Preprocess the image to improve OCR accuracy."""
    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is None:
            print(f"Error: Unable to load image at {image_path}")
            return None
        
        # Reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Apply thresholding
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save preprocessed image
        temp_file = f"temp_preprocessed_{os.path.basename(image_path)}"
        cv2.imwrite(temp_file, img)
        
        return temp_file
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def decode_predictions(scores, geometry, score_thresh):
    """Decode the predictions of the EAST text detector."""
    (num_rows, num_cols) = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            score = scores_data[x]
            if score < score_thresh:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(float(score))

    return rectangles, confidences

def detect_text(image_path):
    """Detect text regions in an image using the EAST text detector."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    orig = image.copy()
    (H, W) = image.shape[:2]

    # Ensure the image dimensions are multiples of 32
    newW = (W // 32) * 32
    newH = (H // 32) * 32
    resized_image = cv2.resize(image, (newW, newH))

    # Prepare the image for the EAST detector
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    # Run the EAST detector to get the text regions
    try:
        scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    except Exception as e:
        print(f"Error during EAST model forward pass: {e}")
        return None

    # Validate the shape of scores and geometry
    if scores is None or geometry is None or len(scores.shape) != 4 or len(geometry.shape) != 4:
        print("Invalid output from EAST text detector.")
        return None

    # Decode the detected textboxes
    rectangles, confidences = decode_predictions(scores, geometry, 0.5)

    # Apply non-maxima suppression to filter out weak bounding boxes
    if len(rectangles) == 0 or len(confidences) == 0:
        print("No text detected.")
        return None

    indices = cv2.dnn.NMSBoxes(rectangles, confidences, score_threshold=0.5, nms_threshold=0.4)

    if len(indices) == 0:
        print("No text detected after NMS.")
        return None

    # Draw the bounding boxes on the image
    for i in indices:
        (start_x, start_y, end_x, end_y) = rectangles[i[0]]
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Text Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return "detected_text.png"

def extract_text_from_image(image_path):
    """Extract text from an image using multiple OCR configurations."""
    try:
        # Detect text regions using SSD model
        detected_text_image_path = detect_text(image_path)
        if detected_text_image_path:
            image_path = detected_text_image_path
        
        # Preprocess image first
        preprocessed_image_path = preprocess_image(image_path)
        if preprocessed_image_path is None:
            print("Image preprocessing failed, trying original image")
            preprocessed_image_path = image_path
        
        # Try multiple OCR configurations and combine results
        image = Image.open(preprocessed_image_path)
        
        # Configuration 1: Standard
        text1 = pytesseract.image_to_string(image)
        
        # Configuration 2: Page segmentation mode 4 (single column of text)
        text2 = pytesseract.image_to_string(image, config='--psm 4')
        
        # Configuration 3: Page segmentation mode 6 (single uniform block of text)
        text3 = pytesseract.image_to_string(image, config='--psm 6')
        
        # Configuration 4: Page segmentation mode 11 (sparse text - no specific formatting)
        text4 = pytesseract.image_to_string(image, config='--psm 11')
        
        # Configuration 5: LSTM OCR Engine mode with line segmentation
        text5 = pytesseract.image_to_string(image, config='--psm 3 --oem 1')
        
        # Combine texts (use the longest one as it probably has the most information)
        texts = [text1, text2, text3, text4, text5]
        best_text = max(texts, key=len)
        
        # Normalize units in the combined text
        normalized_text = normalize_units(best_text)
        
        # Clean up temporary file
        if preprocessed_image_path != image_path:
            try:
                os.remove(preprocessed_image_path)
            except:
                pass
                
        return normalized_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def contains_nutrition_keywords(text):
    """Check if the text contains nutrition-related keywords."""
    keywords = [
        "nutrition", "serving", "calories", "fat", "protein", 
        "carbohydrate", "sodium", "sugar", "vitamin", "mineral"
    ]
    for keyword in keywords:
        if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
            return True
    return False

def parse_nutrition_info(text):
    """Parse nutrition information from extracted text with more robust patterns."""
    if not contains_nutrition_keywords(text):
        print("Warning: Extracted text doesn't appear to contain nutrition information.")
    
    nutrition_data = {
        "serving_size": None,
        "calories": None,
        "total_fat": None,
        "saturated_fat": None,
        "trans_fat": None,
        "cholesterol": None,
        "sodium": None,
        "total_carbohydrate": None,
        "dietary_fiber": None,
        "sugars": None,
        "protein": None,
        "ingredients": None,
    }
    
    # Enhanced pattern matching with fallbacks
    
    # Extract serving size - try multiple patterns
    serving_patterns = [
        r"Serving\s+Size[:\s]*([^\.]*?)(Serving|Amount|Calories|Per)",
        r"Serving\s+Size[:\s]*([0-9]+\s*[a-zA-Z]*)",
        r"Serving[:\s]*([0-9]+\s*[a-zA-Z]*)",
    ]
    
    for pattern in serving_patterns:
        serving_match = re.search(pattern, text, re.IGNORECASE)
        if serving_match:
            serving_size = serving_match.group(1).strip()
            if serving_size:
                nutrition_data["serving_size"] = serving_size
                break
    
    # Extract calories - try multiple patterns
    calories_patterns = [
        r"Calories\s+(\d+)",
        r"Energy\s+(\d+)\s*kcal",
        r"Cal[:\s]*(\d+)",
    ]
    
    for pattern in calories_patterns:
        calories_match = re.search(pattern, text, re.IGNORECASE)
        if calories_match:
            nutrition_data["calories"] = calories_match.group(1).strip()
            break
    
    # Extract other nutrition facts with more flexible patterns
    
    # Total Fat
    fat_match = re.search(r"Total\s+Fat\s*[:\s]*\s*(\d+\.?\d*\s*[g%])", text, re.IGNORECASE)
    if fat_match:
        nutrition_data["total_fat"] = fat_match.group(1).strip()
    
    # Saturated Fat
    sat_fat_match = re.search(r"Saturated\s+Fat\s*[:\s]*\s*(\d+\.?\d*\s*[g%])", text, re.IGNORECASE)
    if sat_fat_match:
        nutrition_data["saturated_fat"] = sat_fat_match.group(1).strip()
    
    # Cholesterol
    chol_match = re.search(r"Cholesterol\s*[:\s]*\s*(\d+\s*mg)", text, re.IGNORECASE)
    if chol_match:
        nutrition_data["cholesterol"] = chol_match.group(1).strip()
    
    # Sodium
    sodium_match = re.search(r"Sodium\s*[:\s]*\s*(\d+\s*mg)", text, re.IGNORECASE)
    if sodium_match:
        nutrition_data["sodium"] = sodium_match.group(1).strip()
    
    # Total Carbohydrate
    carb_match = re.search(r"(Total\s+)?Carbohydrate\s*[:\s]*\s*(\d+\.?\d*\s*[g%])", text, re.IGNORECASE)
    if carb_match:
        nutrition_data["total_carbohydrate"] = carb_match.group(2).strip()
    
    # Dietary Fiber
    fiber_match = re.search(r"(Dietary\s+)?Fiber\s*[:\s]*\s*(\d+\.?\d*\s*[g%])", text, re.IGNORECASE)
    if fiber_match:
        nutrition_data["dietary_fiber"] = fiber_match.group(2).strip()
    
    # Sugars
    sugar_match = re.search(r"Sugars\s*[:\s]*\s*(\d+\.?\d*\s*[g%])", text, re.IGNORECASE)
    if sugar_match:
        nutrition_data["sugars"] = sugar_match.group(1).strip()
    
    # Protein
    protein_match = re.search(r"Protein\s*[:\s]*\s*(\d+\.?\d*\s*[g%])", text, re.IGNORECASE)
    if protein_match:
        nutrition_data["protein"] = protein_match.group(1).strip()
    
    # Extract ingredients - multiple strategies
    
    # Strategy 1: Look for an "Ingredients:" section
    ingredients_section = re.split(r'(?:Ingredients|INGREDIENTS)[:\s]', text, flags=re.IGNORECASE)
    if len(ingredients_section) > 1:
        # Get the text after "Ingredients:" until the next section
        potential_ingredients = ingredients_section[1].strip()
        # Limit to the first paragraph or section
        end_markers = [
            "\n\n", "Nutrition Facts", "Distributed by", "Keep Refrigerated", 
            "Allergen", "Contains", "Storage", "Best before", "how2recycle", "PLASTIC"
        ]
        for marker in end_markers:
            if marker.lower() in potential_ingredients.lower():
                potential_ingredients = potential_ingredients.split(marker, 1)[0].strip()
        
        # Ensure the extracted ingredients are of reasonable length
        if len(potential_ingredients) > 10:  # Reasonable minimum length for ingredients
            nutrition_data["ingredients"] = potential_ingredients

    # Strategy 2: Look for comma-separated lists that might be ingredients
    if not nutrition_data["ingredients"]:
        lines = text.split('\n')
        for line in lines:
            # If the line has multiple commas and looks like ingredients
            if line.count(',') >= 2 and len(line) > 30:
                # Check if it has common ingredient words
                ingredient_indicators = [
                    "water", "sugar", "salt", "oil", "extract", "acid", 
                    "flour", "starch", "natural", "artificial"
                ]
                if any(indicator in line.lower() for indicator in ingredient_indicators):
                    nutrition_data["ingredients"] = line.strip()
                    break
    
    # Strategy 3: Look for "Contains:" statements which often list allergens
    contains_match = re.search(r"Contains[:\s]\s*([^\.]*)", text, re.IGNORECASE)
    if contains_match and not nutrition_data["ingredients"]:
        nutrition_data["ingredients"] = "Contains: " + contains_match.group(1).strip()
    
    return nutrition_data

def check_for_allergens(ingredients_text):
    """Check for common allergens in ingredients with improved detection."""
    if not ingredients_text:
        return []
    
    common_allergens = [
        "milk", "dairy", "lactose", "whey", "casein",
        "egg", "eggs",
        "peanut", "peanuts",
        "tree nut", "tree nuts", "almond", "almonds", "walnut", "walnuts", "cashew", "cashews", 
        "pistachio", "pistachios", "hazelnut", "hazelnuts", "pecan", "pecans",
        "soy", "soya", "tofu", "edamame",
        "wheat", "gluten", "barley", "rye", "spelt", "triticale",
        "fish", "shellfish", "crustacean", "crustaceans", "shrimp", "crab", "lobster",
        "sulfite", "sulfites",
        "sesame", "mustard"
    ]
    
    found_allergens = []
    ingredients_lower = ingredients_text.lower()
    
    for allergen in common_allergens:
        if re.search(r'\b' + re.escape(allergen) + r'\b', ingredients_lower):
            found_allergens.append(allergen)
    
    # Special case for "may contain" statements
    may_contain_match = re.search(r"may\s+contain\s+([^\.]*)", ingredients_lower)
    if may_contain_match:
        may_contain_text = may_contain_match.group(1)
        for allergen in common_allergens:
            if re.search(r'\b' + re.escape(allergen) + r'\b', may_contain_text):
                found_allergens.append(allergen)
    
    # Deduplicate related allergens
    allergen_groups = {
        "milk": ["milk", "dairy", "lactose", "whey", "casein"],
        "eggs": ["egg", "eggs"],
        "peanuts": ["peanut", "peanuts"],
        "tree nuts": ["tree nut", "tree nuts", "almond", "almonds", "walnut", "walnuts", "cashew", "cashews", "pistachio", "pistachios", "hazelnut", "hazelnuts", "pecan", "pecans"],
        "soy": ["soy", "soya", "tofu", "edamame"],
        "wheat/gluten": ["wheat", "gluten", "barley", "rye", "spelt", "triticale"],
        "fish": ["fish"],
        "shellfish": ["shellfish", "crustacean", "crustaceans", "shrimp", "crab", "lobster"],
        "sulfites": ["sulfite", "sulfites"],
        "sesame": ["sesame"],
        "mustard": ["mustard"]
    }
    
    deduplicated_allergens = set()
    for found in found_allergens:
        for group, items in allergen_groups.items():
            if found in items:
                deduplicated_allergens.add(group)
                break
    
    return list(deduplicated_allergens)

def check_nutritional_concerns(nutrition_data):
    """Check for nutritional concerns based on the nutrition data."""
    concerns = []
    
    # Extract numeric values with more robust patterns
    def extract_numeric(value_str):
        if not value_str:
            return None
        match = re.search(r'(\d+\.?\d*)', value_str)
        if match:
            return float(match.group(1))
        return None
    
    # High sodium check
    sodium_value = extract_numeric(nutrition_data.get("sodium"))
    if sodium_value is not None and sodium_value > 500:
        concerns.append("high sodium")
    
    # High sugar check
    sugar_value = extract_numeric(nutrition_data.get("sugars"))
    if sugar_value is not None and sugar_value > 20:
        concerns.append("high sugar")
    
    # High fat check
    fat_value = extract_numeric(nutrition_data.get("total_fat"))
    if fat_value is not None and fat_value > 15:
        concerns.append("high fat")
    
    return concerns

def generate_response(nutrition_data, extracted_text):
    """Generate a response about nutrition and allergens."""
    # First, check if we got any meaningful data
    is_valid_data = any(value for key, value in nutrition_data.items() if key != "ingredients")
    
    if not is_valid_data:
        return "Sorry, I couldn't detect clear nutrition information from the image. Please try a clearer photo of the nutrition label."
    
    # Check for allergens in ingredients
    ingredients_text = nutrition_data.get("ingredients", "")
    allergens = check_for_allergens(ingredients_text)
    nutritional_concerns = check_nutritional_concerns(nutrition_data)
    
    # Format serving size
    serving_size = nutrition_data.get("serving_size", "Not detected")
    
    # Create response
    response = ""
    
    # Add allergen information
    if allergens:
        allergens_text = ", ".join(allergens)
        response += f"Alert: This product contains potential allergens ({allergens_text}). "
    else:
        if ingredients_text:
            response += "No common allergens detected in the ingredients. "
        else:
            response += "Could not detect ingredients, unable to check for allergens. "
    
    # Add nutritional concerns
    if nutritional_concerns:
        concerns_text = ", ".join(nutritional_concerns)
        response += f"Nutritional note: {concerns_text.capitalize()}. "
    
    # Add serving size information and question
    if serving_size != "Not detected":
        response += f"Serving size is {serving_size}. "
    
    response += "How much did you eat (whole package, half can, etc.)?"
    
    return response

def main():
    if len(sys.argv) < 2:
        print("Usage: python nutrition_analyzer.py [path_to_image]")
        return
    
    image_path = sys.argv[1]
    
    # Extract text from image with improved OCR
    extracted_text = extract_text_from_image(image_path)
    if not extracted_text:
        print("Failed to extract text from image.")
        return
    
    # Normalize units in the extracted text
    extracted_text = normalize_units(extracted_text)
    
    # Parse nutrition information with more robust patterns
    nutrition_data = parse_nutrition_info(extracted_text)
    
    # Generate response
    response = generate_response(nutrition_data, extracted_text)
    
    print("\n--- Extracted Text ---")
    print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
    
    print("\n--- Parsed Nutrition Data ---")
    print(json.dumps(nutrition_data, indent=2))
    
    print("\n--- Analysis Result ---")
    print(response)

if __name__ == "__main__":
    main()
