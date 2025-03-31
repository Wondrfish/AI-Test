package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"

	"cloud.google.com/go/vision/apiv1"
	"google.golang.org/api/option"
)

type NutritionData struct {
	Calories             string `json:"calories"`
	TotalFat             string `json:"total_fat"`
	SaturatedFat         string `json:"saturated_fat"`
	TransFat             string `json:"trans_fat"`
	Cholesterol          string `json:"cholesterol"`
	Sodium               string `json:"sodium"`
	TotalCarbohydrates   string `json:"total_carbohydrates"`
	DietaryFiber         string `json:"dietary_fiber"`
	Sugars               string `json:"sugars"`
	Protein              string `json:"protein"`
	ServingSize          string `json:"serving_size"`
	ServingsPerContainer string `json:"servings_per_container"`
}

type FoodRepoProduct struct {
	ID           int                  `json:"id"`
	Name         string               `json:"name"`
	Nutrients    map[string]Nutrient  `json:"nutrients"`
	ComparedWith *NutritionComparison `json:"compared_with,omitempty"`
}

type Nutrient struct {
	Value float64 `json:"value"`
	Unit  string  `json:"unit"`
}

type FoodRepoResponse struct {
	Products []FoodRepoProduct `json:"products"`
}

type NutritionComparison struct {
	Calories           float64 `json:"calories_diff_percent"`
	TotalFat           float64 `json:"total_fat_diff_percent"`
	Protein            float64 `json:"protein_diff_percent"`
	TotalCarbohydrates float64 `json:"total_carbohydrates_diff_percent"`
	Sugars             float64 `json:"sugars_diff_percent"`
	Sodium             float64 `json:"sodium_diff_percent"`
}

type ComparisonResult struct {
	UserProduct  NutritionData      `json:"user_product"`
	SimilarItems []FoodRepoProduct  `json:"similar_items"`
}

const (
	FOODREPO_API_URL    = "https://www.foodrepo.org/api/v3"
	FOODREPO_API_KEY    = "YOUR_FOODREPO_API_KEY" // Replace with actual API key or use environment variable
)

func main() {
	http.HandleFunc("/", homeHandler)
	http.HandleFunc("/upload", uploadHandler)
	http.HandleFunc("/compare", compareHandler)
	
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	
	log.Printf("Server listening on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func homeHandler(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "templates/index.html")
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	file, header, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Error retrieving the file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Create a temporary file
	tempFile, err := os.CreateTemp("", "upload-*.jpg")
	if err != nil {
		http.Error(w, "Error creating temp file", http.StatusInternalServerError)
		return
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()

	// Copy the uploaded file to the temp file
	_, err = io.Copy(tempFile, file)
	if err != nil {
		http.Error(w, "Error saving file", http.StatusInternalServerError)
		return
	}

	// Process the image
	text, err := detectText(tempFile.Name())
	if err != nil {
		http.Error(w, fmt.Sprintf("Error processing image: %v", err), http.StatusInternalServerError)
		return
	}

	// Parse nutrition data
	nutritionData := parseNutritionInfo(text)

	// Return JSON response
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{
		"filename": "%s",
		"text": "%s",
		"nutrition_data": {
			"calories": "%s",
			"total_fat": "%s",
			"saturated_fat": "%s",
			"trans_fat": "%s",
			"cholesterol": "%s",
			"sodium": "%s",
			"total_carbohydrates": "%s",
			"dietary_fiber": "%s",
			"sugars": "%s",
			"protein": "%s",
			"serving_size": "%s",
			"servings_per_container": "%s"
		}
	}`,
		header.Filename,
		strings.ReplaceAll(text, `"`, `\"`),
		nutritionData.Calories,
		nutritionData.TotalFat,
		nutritionData.SaturatedFat,
		nutritionData.TransFat,
		nutritionData.Cholesterol,
		nutritionData.Sodium,
		nutritionData.TotalCarbohydrates,
		nutritionData.DietaryFiber,
		nutritionData.Sugars,
		nutritionData.Protein,
		nutritionData.ServingSize,
		nutritionData.ServingsPerContainer,
	)
}

func compareHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var nutritionData NutritionData
	err := json.NewDecoder(r.Body).Decode(&nutritionData)
	if err != nil {
		http.Error(w, "Error parsing nutrition data", http.StatusBadRequest)
		return
	}

	// Extract product name from the request (could be from form data or query params)
	productName := r.URL.Query().Get("product_name")
	if productName == "" {
		productName = "food" // Default search term
	}

	// Get similar products from FoodRepo
	similarProducts, err := findSimilarProducts(productName)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error finding similar products: %v", err), http.StatusInternalServerError)
		return
	}

	// Calculate comparisons
	enrichedProducts := compareNutrition(nutritionData, similarProducts)

	// Prepare response
	result := ComparisonResult{
		UserProduct:  nutritionData,
		SimilarItems: enrichedProducts,
	}

	// Return JSON response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func detectText(filePath string) (string, error) {
	ctx := context.Background()
	
	// Initialize client (credentials should be set via GOOGLE_APPLICATION_CREDENTIALS)
	client, err := vision.NewImageAnnotatorClient(ctx, option.WithCredentialsFile("path/to/your/credentials.json"))
	if err != nil {
		return "", fmt.Errorf("failed to create client: %v", err)
	}
	defer client.Close()

	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %v", err)
	}
	defer file.Close()

	image, err := vision.NewImageFromReader(file)
	if err != nil {
		return "", fmt.Errorf("failed to create image: %v", err)
	}

	annotations, err := client.DetectTexts(ctx, image, nil, 10)
	if err != nil {
		return "", fmt.Errorf("failed to detect text: %v", err)
	}

	if len(annotations) == 0 {
		return "", fmt.Errorf("no text found")
	}

	return annotations[0].Description, nil
}

func parseNutritionInfo(text string) NutritionData {
	data := NutritionData{}
	lowerText := strings.ToLower(text)
	
	// Define patterns and their corresponding fields
	patterns := map[string]*string{
		`calories.*?(\d+\s*(?:kcal|calories|cal))`:         &data.Calories,
		`total fat.*?(\d+\.?\d*\s*(?:g|mg))`:              &data.TotalFat,
		`saturated fat.*?(\d+\.?\d*\s*(?:g|mg))`:          &data.SaturatedFat,
		`trans fat.*?(\d+\.?\d*\s*(?:g|mg))`:              &data.TransFat,
		`cholesterol.*?(\d+\.?\d*\s*(?:mg|g))`:            &data.Cholesterol,
		`sodium.*?(\d+\.?\d*\s*(?:mg|g))`:                 &data.Sodium,
		`total carbohydrate.*?(\d+\.?\d*\s*(?:g|mg))`:     &data.TotalCarbohydrates,
		`dietary fiber.*?(\d+\.?\d*\s*(?:g|mg))`:          &data.DietaryFiber,
		`sugars.*?(\d+\.?\d*\s*(?:g|mg))`:                 &data.Sugars,
		`protein.*?(\d+\.?\d*\s*(?:g|mg))`:                &data.Protein,
		`serving size.*?(\d+\.?\d*\s*(?:g|ml|oz|cup|tbsp))`: &data.ServingSize,
		`servings per container.*?(\d+\.?\d*)`:            &data.ServingsPerContainer,
	}

	for pattern, field := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindStringSubmatch(lowerText)
		if len(matches) > 1 {
			*field = matches[1]
		}
	}

	return data
}

func findSimilarProducts(productName string) ([]FoodRepoProduct, error) {
	// Prepare the request to FoodRepo API
	url := fmt.Sprintf("%s/products/search?query=%s", FOODREPO_API_URL, productName)
	
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}
	
	// Add API key to headers
	req.Header.Add("Accept", "application/json")
	req.Header.Add("Authorization", fmt.Sprintf("Token token=%s", FOODREPO_API_KEY))
	
	// Execute the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()
	
	// Check response status
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status code %d", resp.StatusCode)
	}
	
	// Parse response
	var foodRepoResp FoodRepoResponse
	err = json.NewDecoder(resp.Body).Decode(&foodRepoResp)
	if err != nil {
		return nil, fmt.Errorf("error parsing response: %v", err)
	}
	
	return foodRepoResp.Products, nil
}

func compareNutrition(userData NutritionData, similarProducts []FoodRepoProduct) []FoodRepoProduct {
	// Extract numeric values from user data
	userCalories := extractNumericValue(userData.Calories)
	userTotalFat := extractNumericValue(userData.TotalFat)
	userProtein := extractNumericValue(userData.Protein)
	userCarbs := extractNumericValue(userData.TotalCarbohydrates)
	userSugars := extractNumericValue(userData.Sugars)
	userSodium := extractNumericValue(userData.Sodium)
	
	result := make([]FoodRepoProduct, 0, len(similarProducts))
	
	for _, product := range similarProducts {
		// Calculate percentage differences
		comparison := &NutritionComparison{}
		
		// Check if nutrient data exists in the product
		if calories, ok := product.Nutrients["energy"]; ok && userCalories > 0 {
			comparison.Calories = calculatePercentageDiff(calories.Value, userCalories)
		}
		
		if fat, ok := product.Nutrients["fat"]; ok && userTotalFat > 0 {
			comparison.TotalFat = calculatePercentageDiff(fat.Value, userTotalFat)
		}
		
		if protein, ok := product.Nutrients["proteins"]; ok && userProtein > 0 {
			comparison.Protein = calculatePercentageDiff(protein.Value, userProtein)
		}
		
		if carbs, ok := product.Nutrients["carbohydrates"]; ok && userCarbs > 0 {
			comparison.TotalCarbohydrates = calculatePercentageDiff(carbs.Value, userCarbs)
		}
		
		if sugars, ok := product.Nutrients["sugars"]; ok && userSugars > 0 {
			comparison.Sugars = calculatePercentageDiff(sugars.Value, userSugars)
		}
		
		if sodium, ok := product.Nutrients["sodium"]; ok && userSodium > 0 {
			comparison.Sodium = calculatePercentageDiff(sodium.Value, userSodium)
		}
		
		// Attach comparison data to the product
		product.ComparedWith = comparison
		result = append(result, product)
	}
	
	return result
}

func calculatePercentageDiff(a, b float64) float64 {
	if b == 0 {
		return 0 // Avoid division by zero
	}
	return ((a - b) / b) * 100
}

func extractNumericValue(valueStr string) float64 {
	// Extract numeric part from strings like "100 kcal" or "5.2 g"
	re := regexp.MustCompile(`(\d+\.?\d*)`)
	matches := re.FindStringSubmatch(valueStr)
	if len(matches) > 1 {
		var value float64
		fmt.Sscanf(matches[1], "%f", &value)
		return value
	}
	return 0
}
