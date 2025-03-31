import io
import re
import json
from google.cloud import vision_v1
from google.oauth2 import service_account

# Credential setup (place this near the top of your script)
def get_vision_client():
    """
    Create and return a Google Cloud Vision client with credentials.
    
    IMPORTANT: Replace '/path/to/your/downloaded/keyfile.json' 
    with the actual path to your Google Cloud service account JSON key file.
    """
    credentials_path = r"C:\Users\ajani\OneDrive\Desktop\nutritionlabelocr-df0a14d24981.json"
    try:
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = vision_v1.ImageAnnotatorClient(credentials=credentials)
        return client
    except Exception as e:
        print(f"Error setting up Vision client: {e}")
        print("Ensure:")
        print("1. The credentials path is correct")
        print("2. The JSON key file exists")
        print("3. You have enabled the Vision API")
        return None

def detect_text(image_path):
    """Detects text in an image using Google Cloud Vision API"""
    try:
        # Use the function to get the client
        client = get_vision_client()
        
        if not client:
            print("Failed to create Vision client.")
            return ""

        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision_v1.types.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if not texts:
            print("No text detected.")
            return ""

        detected_text = texts[0].description  # First result is the full extracted text
        return detected_text
    except Exception as e:
        print(f"Error in text detection: {e}")
        return ""

def normalize_units(text):
    """Normalize units in the OCR text."""
    replacements = {
        r'\b(\d+)\s*m9\b': r'\1 mg',    # Fix 'm9' -> 'mg'
        r'\b(\d+)\s*9\b': r'\1 g',      # Fix '9' -> 'g'
        r'\b(\d+)\s*ozz\b': r'\1 oz',   # Fix 'ozz' -> 'oz'
        r'\b(\d+)\s*cal\b': r'\1 Cal'   # Standardize 'cal' to 'Cal'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def parse_nutrition_info(text):
    """Parse nutrition information from extracted text."""
    nutrition_data = {
        "serving_size": None,
        "calories": None,
        "total_fat": None,
        "saturated_fat": None,
        "cholesterol": None,
        "sodium": None,
        "total_carbohydrate": None,
        "dietary_fiber": None,
        "sugars": None,
        "protein": None,
        "ingredients": None,
    }
    
    # Serving size patterns
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
    
    # Calories patterns
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
    
    # Nutrition fact patterns
    nutrient_patterns = [
        ("total_fat", r"Total\s+Fat\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("saturated_fat", r"Saturated\s+Fat\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("cholesterol", r"Cholesterol\s*[:\s]*\s*(\d+\s*mg)"),
        ("sodium", r"Sodium\s*[:\s]*\s*(\d+\s*mg)"),
        ("total_carbohydrate", r"(Total\s+)?Carbohydrate\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("dietary_fiber", r"(Dietary\s+)?Fiber\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("sugars", r"Sugars\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("protein", r"Protein\s*[:\s]*\s*(\d+\.?\d*\s*[g%])")
    ]
    
    for key, pattern in nutrient_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Handle patterns with capture groups
            nutrition_data[key] = match.group(1).strip() if len(match.groups()) == 1 else match.group(2).strip()
    
    # Ingredients extraction
    ingredients_section = re.split(r'(?:Ingredients|INGREDIENTS|INGREDIENT|ingredient)[:\s]', text, flags=re.IGNORECASE)
    if len(ingredients_section) > 1:
        potential_ingredients = ingredients_section[1].strip()
        # Refine end markers to exclude unrelated text
        end_markers = [
            "\n\n", "Nutrition Facts", "Nutritional", "Allergen", "Contains", 
            "Storage", "Best before", "Dist.", "KEEP REFRIGERATED", "how2recycle.info", 
            "PLASTIC", "BOTTLE", "CA CRV", "CTRV", "HI 5¢", "ME 5¢", "% Daily Value", 
            "Serving size", "Amount per serving", "Calories", "Total Fat", "Cholesterol"
        ]
        for marker in end_markers:
            if marker.lower() in potential_ingredients.lower():
                potential_ingredients = potential_ingredients.split(marker, 1)[0].strip()
        
        # Ensure the extracted ingredients are meaningful
        if len(potential_ingredients) > 10:
            nutrition_data["ingredients"] = potential_ingredients
    
    return nutrition_data

def check_for_allergens(ingredients_text):
    """Check for common allergens in ingredients."""
    if not ingredients_text:
        return []
    
    common_allergens = [
        "milk", "dairy", "lactose", "whey", "casein",
        "egg", "eggs",
        "peanut", "peanuts",
        "tree nut", "tree nuts", "almond", "almonds", "walnut", "walnuts", 
        "cashew", "cashews", "pistachio", "pistachios", 
        "hazelnut", "hazelnuts", "pecan", "pecans",
        "soy", "soya", "tofu", "edamame",
        "wheat", "gluten", "barley", "rye", "spelt", "triticale",
        "fish", "shellfish", "crustacean", "crustaceans", 
        "shrimp", "crab", "lobster",
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
        "tree nuts": ["tree nut", "tree nuts", "almond", "almonds", "walnut", "walnuts", 
                     "cashew", "cashews", "pistachio", "pistachios", 
                     "hazelnut", "hazelnuts", "pecan", "pecans"],
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
    
    def extract_numeric(value_str):
        if not value_str:
            return None
        match = re.search(r'(\d+\.?\d*)', value_str)
        if match:
            return float(match.group(1))
        return None
    
    # Nutritional thresholds
    thresholds = {
        "sodium": 500,   # mg
        "sugars": 20,    # g
        "total_fat": 15  # g
    }
    
    for nutrient, threshold in thresholds.items():
        value = extract_numeric(nutrition_data.get(nutrient))
        if value is not None and value > threshold:
            concerns.append(f"high {nutrient}")
    
    return concerns

def generate_response(nutrition_data, extracted_text):
    """Generate a response about nutrition and allergens."""
    # Check if we have meaningful data
    is_valid_data = any(value for key, value in nutrition_data.items() if key != "ingredients")
    
    if not is_valid_data:
        return "Sorry, I couldn't detect clear nutrition information from the image. Please try a clearer photo of the nutrition label."
    
    # Check allergens and nutritional concerns
    ingredients_text = nutrition_data.get("ingredients", "")
    allergens = check_for_allergens(ingredients_text)
    nutritional_concerns = check_nutritional_concerns(nutrition_data)
    
    # Prepare response
    response = ""
    
    # Allergen information
    if allergens:
        allergens_text = ", ".join(allergens)
        response += f"Alert: This product contains potential allergens ({allergens_text}). "
    else:
        response += "No common allergens detected in the ingredients. "
    
    # Nutritional concerns
    if nutritional_concerns:
        concerns_text = ", ".join(nutritional_concerns)
        response += f"Nutritional note: {concerns_text.capitalize()}. "
    
    # Serving size
    serving_size = nutrition_data.get("serving_size", "Not detected")
    if serving_size != "Not detected":
        response += f"Serving size is {serving_size}. "
    
    # Calories
    calories = nutrition_data.get("calories")
    if calories:
        response += f"Calories per serving: {calories}. "
    
    response += "How much did you eat (whole package, half can, etc.)?"
    
    return response

def main(image_path):
    """Main function to process nutrition label image"""
    # Detect text
    extracted_text = detect_text(image_path)
    if not extracted_text:
        print("Failed to extract text from image.")
        return None
    
    # Normalize units
    extracted_text = normalize_units(extracted_text)
    
    # Parse nutrition information
    nutrition_data = parse_nutrition_info(extracted_text)
    
    # Generate response
    response = generate_response(nutrition_data, extracted_text)
    
    # Print results
    print("\n--- Extracted Text ---")
    print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
    
    print("\n--- Parsed Nutrition Data ---")
    print(json.dumps(nutrition_data, indent=2))
    
    print("\n--- Analysis Result ---")
    print(response)
    
    return {
        "extracted_text": extracted_text,
        "nutrition_data": nutrition_data,
        "response": response
    }

if __name__ == "__main__":
    # Example usage
    image_path = "C:\\Users\\ajani\\Downloads\\sorted_data\\test_lable1.jpg"
    main(image_path)
