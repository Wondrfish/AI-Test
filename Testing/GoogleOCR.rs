use std::path::Path;
use std::fs;
use serde::{Serialize, Deserialize};
use regex::Regex;
use anyhow::{Result, Context};
use google_cloud_vision::v1::{image_annotator_client::ImageAnnotatorClient, Feature, FeatureType, Image, AnnotateImageRequest};

mod google_cloud_vision {
    pub mod v1 {
        tonic::include_proto!("google.cloud.vision.v1");
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct NutritionData {
    serving_size: Option<String>,
    calories: Option<String>,
    total_fat: Option<String>,
    saturated_fat: Option<String>,
    cholesterol: Option<String>,
    sodium: Option<String>,
    total_carbohydrate: Option<String>,
    dietary_fiber: Option<String>,
    sugars: Option<String>,
    protein: Option<String>,
    ingredients: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnalysisResult {
    extracted_text: String,
    nutrition_data: NutritionData,
    response: String,
}

pub async fn detect_text(image_path: &str) -> Result<String> {
    let client = ImageAnnotatorClient::new().await?;
    
    let image_content = fs::read(image_path)?;
    let image = Image {
        content: image_content,
        ..Default::default()
    };
    
    let feature = Feature {
        r#type: FeatureType::TextDetection as i32,
        ..Default::default()
    };
    
    let request = AnnotateImageRequest {
        image: Some(image),
        features: vec![feature],
        ..Default::default()
    };
    
    let response = client.batch_annotate_images(vec![request]).await?;
    
    let text = response.get_ref().responses[0]
        .full_text_annotation
        .as_ref()
        .context("No text detected")?
        .text
        .clone();
    
    Ok(text)
}

fn normalize_units(text: &str) -> String {
    let replacements = vec![
        (r"(?i)\b(\d+)\s*m9\b", "$1 mg"),
        (r"(?i)\b(\d+)\s*9\b", "$1 g"),
        (r"(?i)\b(\d+)\s*ozz\b", "$1 oz"),
        (r"(?i)\b(\d+)\s*cal\b", "$1 Cal"),
    ];
    
    let mut result = text.to_string();
    for (pattern, replacement) in replacements {
        let re = Regex::new(pattern).unwrap();
        result = re.replace_all(&result, replacement).to_string();
    }
    
    result
}

fn parse_nutrition_info(text: &str) -> NutritionData {
    let mut nutrition_data = NutritionData {
        serving_size: None,
        calories: None,
        total_fat: None,
        saturated_fat: None,
        cholesterol: None,
        sodium: None,
        total_carbohydrate: None,
        dietary_fiber: None,
        sugars: None,
        protein: None,
        ingredients: None,
    };
    
    // Serving size patterns
    let serving_patterns = vec![
        r"(?i)Serving\s+Size[:\s]*([^\.]*?)(Serving|Amount|Calories|Per)",
        r"(?i)Serving\s+Size[:\s]*([0-9]+\s*[a-zA-Z]*)",
        r"(?i)Serving[:\s]*([0-9]+\s*[a-zA-Z]*)",
    ];
    
    for pattern in serving_patterns {
        if let Some(caps) = Regex::new(pattern).unwrap().captures(text) {
            if let Some(serving_size) = caps.get(1) {
                nutrition_data.serving_size = Some(serving_size.as_str().trim().to_string());
                break;
            }
        }
    }
    
    // Calories patterns
    let calories_patterns = vec![
        r"(?i)Calories\s+(\d+)",
        r"(?i)Energy\s+(\d+)\s*kcal",
        r"(?i)Cal[:\s]*(\d+)",
    ];
    
    for pattern in calories_patterns {
        if let Some(caps) = Regex::new(pattern).unwrap().captures(text) {
            if let Some(calories) = caps.get(1) {
                nutrition_data.calories = Some(calories.as_str().trim().to_string());
                break;
            }
        }
    }
    
    // Nutrient patterns
    let nutrient_patterns = vec![
        ("total_fat", r"(?i)Total\s+Fat\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("saturated_fat", r"(?i)Saturated\s+Fat\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("cholesterol", r"(?i)Cholesterol\s*[:\s]*\s*(\d+\s*mg)"),
        ("sodium", r"(?i)Sodium\s*[:\s]*\s*(\d+\s*mg)"),
        ("total_carbohydrate", r"(?i)(Total\s+)?Carbohydrate\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("dietary_fiber", r"(?i)(Dietary\s+)?Fiber\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("sugars", r"(?i)Sugars\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
        ("protein", r"(?i)Protein\s*[:\s]*\s*(\d+\.?\d*\s*[g%])"),
    ];
    
    for (field, pattern) in nutrient_patterns {
        if let Some(caps) = Regex::new(pattern).unwrap().captures(text) {
            let value = if caps.len() > 2 {
                caps.get(2).unwrap().as_str()
            } else {
                caps.get(1).unwrap().as_str()
            };
            
            match field {
                "total_fat" => nutrition_data.total_fat = Some(value.trim().to_string()),
                "saturated_fat" => nutrition_data.saturated_fat = Some(value.trim().to_string()),
                "cholesterol" => nutrition_data.cholesterol = Some(value.trim().to_string()),
                "sodium" => nutrition_data.sodium = Some(value.trim().to_string()),
                "total_carbohydrate" => nutrition_data.total_carbohydrate = Some(value.trim().to_string()),
                "dietary_fiber" => nutrition_data.dietary_fiber = Some(value.trim().to_string()),
                "sugars" => nutrition_data.sugars = Some(value.trim().to_string()),
                "protein" => nutrition_data.protein = Some(value.trim().to_string()),
                _ => (),
            }
        }
    }
    
    // Ingredients extraction
    if let Some(ingredients_section) = Regex::new(r"(?i)(?:Ingredients|INGREDIENTS|INGREDIENT|ingredient)[:\s](.*)")
        .unwrap()
        .captures(text)
    {
        let mut potential_ingredients = ingredients_section.get(1).unwrap().as_str().trim();
        
        let end_markers = vec![
            "\n\n", "Nutrition Facts", "Nutritional", "Allergen", "Contains", 
            "Storage", "Best before", "Dist.", "KEEP REFRIGERATED", "how2recycle.info", 
            "PLASTIC", "BOTTLE", "CA CRV", "CTRV", "HI 5¢", "ME 5¢", "% Daily Value", 
            "Serving size", "Amount per serving", "Calories", "Total Fat", "Cholesterol"
        ];
        
        for marker in end_markers {
            if let Some(pos) = potential_ingredients.to_lowercase().find(&marker.to_lowercase()) {
                potential_ingredients = &potential_ingredients[..pos];
            }
        }
        
        if potential_ingredients.len() > 10 {
            nutrition_data.ingredients = Some(potential_ingredients.trim().to_string());
        }
    }
    
    nutrition_data
}

fn check_for_allergens(ingredients_text: Option<&String>) -> Vec<String> {
    let mut found_allergens = Vec::new();
    
    let ingredients_text = match ingredients_text {
        Some(text) => text.to_lowercase(),
        None => return found_allergens,
    };
    
    let common_allergens = vec![
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
    ];
    
    for allergen in common_allergens {
        if Regex::new(&format!(r"\b{}\b", regex::escape(allergen)))
            .unwrap()
            .is_match(&ingredients_text)
        {
            found_allergens.push(allergen.to_string());
        }
    }
    
    // Check for "may contain" statements
    if let Some(caps) = Regex::new(r"(?i)may\s+contain\s+([^\.]*)")
        .unwrap()
        .captures(&ingredients_text)
    {
        let may_contain_text = caps.get(1).unwrap().as_str();
        for allergen in common_allergens {
            if Regex::new(&format!(r"\b{}\b", regex::escape(allergen)))
                .unwrap()
                .is_match(may_contain_text)
            {
                found_allergens.push(allergen.to_string());
            }
        }
    }
    
    // Deduplicate allergens
    let allergen_groups = vec![
        ("milk", vec!["milk", "dairy", "lactose", "whey", "casein"]),
        ("eggs", vec!["egg", "eggs"]),
        ("peanuts", vec!["peanut", "peanuts"]),
        ("tree nuts", vec!["tree nut", "tree nuts", "almond", "almonds", "walnut", "walnuts", 
                          "cashew", "cashews", "pistachio", "pistachios", 
                          "hazelnut", "hazelnuts", "pecan", "pecans"]),
        ("soy", vec!["soy", "soya", "tofu", "edamame"]),
        ("wheat/gluten", vec!["wheat", "gluten", "barley", "rye", "spelt", "triticale"]),
        ("fish", vec!["fish"]),
        ("shellfish", vec!["shellfish", "crustacean", "crustaceans", "shrimp", "crab", "lobster"]),
        ("sulfites", vec!["sulfite", "sulfites"]),
        ("sesame", vec!["sesame"]),
        ("mustard", vec!["mustard"]),
    ];
    
    let mut deduplicated = Vec::new();
    for (group, items) in allergen_groups {
        if found_allergens.iter().any(|a| items.contains(&a.as_str())) {
            deduplicated.push(group.to_string());
        }
    }
    
    deduplicated
}

fn check_nutritional_concerns(nutrition_data: &NutritionData) -> Vec<String> {
    let mut concerns = Vec::new();
    
    fn extract_numeric(value: &Option<String>) -> Option<f64> {
        value.as_ref().and_then(|v| {
            Regex::new(r"(\d+\.?\d*)").unwrap()
                .captures(v)
                .and_then(|caps| caps.get(1))
                .and_then(|m| m.as_str().parse().ok())
        })
    }
    
    let thresholds = vec![
        ("sodium", 500.0),   // mg
        ("sugars", 20.0),    // g
        ("total_fat", 15.0), // g
    ];
    
    for (nutrient, threshold) in thresholds {
        let value = match nutrient {
            "sodium" => extract_numeric(&nutrition_data.sodium),
            "sugars" => extract_numeric(&nutrition_data.sugars),
            "total_fat" => extract_numeric(&nutrition_data.total_fat),
            _ => None,
        };
        
        if let Some(v) = value {
            if v > threshold {
                concerns.push(format!("high {}", nutrient));
            }
        }
    }
    
    concerns
}

fn generate_response(nutrition_data: &NutritionData, extracted_text: &str) -> String {
    let is_valid_data = nutrition_data.serving_size.is_some() 
        || nutrition_data.calories.is_some()
        || nutrition_data.total_fat.is_some()
        || nutrition_data.sodium.is_some();
    
    if !is_valid_data {
        return "Sorry, I couldn't detect clear nutrition information from the image. Please try a clearer photo of the nutrition label.".to_string();
    }
    
    let mut response = String::new();
    
    // Allergen information
    let allergens = check_for_allergens(nutrition_data.ingredients.as_ref());
    if !allergens.is_empty() {
        response.push_str(&format!("Alert: This product contains potential allergens ({}). ", allergens.join(", ")));
    } else {
        response.push_str("No common allergens detected in the ingredients. ");
    }
    
    // Nutritional concerns
    let concerns = check_nutritional_concerns(nutrition_data);
    if !concerns.is_empty() {
        response.push_str(&format!("Nutritional note: {}. ", concerns.join(", ").to_lowercase()));
    }
    
    // Serving size
    if let Some(serving_size) = &nutrition_data.serving_size {
        response.push_str(&format!("Serving size is {}. ", serving_size));
    }
    
    // Calories
    if let Some(calories) = &nutrition_data.calories {
        response.push_str(&format!("Calories per serving: {}. ", calories));
    }
    
    response.push_str("How much did you eat (whole package, half can, etc.)?");
    
    response
}

pub async fn analyze_nutrition_label(image_path: &str) -> Result<AnalysisResult> {
    // Detect text
    let extracted_text = detect_text(image_path).await?;
    let normalized_text = normalize_units(&extracted_text);
    
    // Parse nutrition information
    let nutrition_data = parse_nutrition_info(&normalized_text);
    
    // Generate response
    let response = generate_response(&nutrition_data, &normalized_text);
    
    Ok(AnalysisResult {
        extracted_text: normalized_text,
        nutrition_data,
        response,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let image_path = "path/to/your/nutrition_label.jpg";
    
    let result = analyze_nutrition_label(image_path).await?;
    
    println!("--- Extracted Text ---");
    println!("{}", if result.extracted_text.len() > 500 {
        &result.extracted_text[..500]
    } else {
        &result.extracted_text
    });
    
    println!("\n--- Parsed Nutrition Data ---");
    println!("{}", serde_json::to_string_pretty(&result.nutrition_data)?);
    
    println!("\n--- Analysis Result ---");
    println!("{}", result.response);
    
    Ok(())
}
