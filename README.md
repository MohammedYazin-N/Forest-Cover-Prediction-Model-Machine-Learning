
# 🌲 Forest Cover Type Prediction Using Machine Learning

Predicting the type of forest cover based on cartographic and ecological variables using the **Covertype Dataset** from Scikit-Learn.

---

## 📄 **Dataset Description**

The **Covertype** dataset contains **581,012 rows** of forest observations from the Roosevelt National Forest of Northern Colorado, USA.
Each record represents a 30×30 meter cell and includes **cartographic variables, soil properties, and wilderness data**.
The goal is to build a machine learning model that predicts the **forest cover type** from seven possible categories based on ecological characteristics.

### 🔢 **Key Columns**

| Column Name                          | Description                                               |
| ------------------------------------ | --------------------------------------------------------- |
| `Elevation`                          | Elevation in meters                                       |
| `Aspect`                             | Direction the slope faces (0–360 degrees)                 |
| `Slope`                              | Slope steepness in degrees                                |
| `Horizontal_Distance_To_Hydrology`   | Distance to nearest surface water                         |
| `Vertical_Distance_To_Hydrology`     | Vertical distance to water                                |
| `Horizontal_Distance_To_Roadways`    | Distance to nearest roadway                               |
| `Hillshade_9am`                      | Hillshade index at 9 AM                                   |
| `Hillshade_Noon`                     | Hillshade index at noon                                   |
| `Hillshade_3pm`                      | Hillshade index at 3 PM                                   |
| `Horizontal_Distance_To_Fire_Points` | Distance to nearest wildfire ignition point               |
| `Wilderness_Area`                    | One-hot encoded binary columns (4 total)                  |
| `Soil_Type`                          | One-hot encoded binary columns (40 total)                 |
| **Target → `Cover_Type`**            | Forest cover type (1–7): Spruce/Fir, Lodgepole Pine, etc. |

📌 **Total Features:** 54
📌 **Type:** Multi-class Classification
📌 **Rows:** 581,012
📌 **Source:** Built-in dataset → `fetch_covtype()` from scikit-learn

---

## 🎯 **Objective of the Project**

To build a **machine learning classification model** that predicts the **forest cover type** based on geographic, hydrological, and soil characteristics.

This project walks through:

* Understanding the dataset
* Cleaning, transforming, and preparing data
* Handling class imbalance
* Training multiple ML models
* Hyperparameter tuning
* Final evaluation & predictions on unseen forest data

---

# 🔄 **PROJECT WORKFLOW**

---

## 🧼 **Stage 1: Dataset Understanding & Preparation**

✔ Loaded the dataset using Scikit-Learn’s `fetch_covtype()`
✔ Converted into a Pandas DataFrame
✔ Checked for:

* Feature count
* Data types
* Missing values (dataset contains **no missing values**)
  ✔ Verified class distribution for the target variable
  ✔ Identified imbalance across cover types

---

## 📊 **Stage 2: Exploratory Data Analysis (EDA)**

Key visualizations performed:

### 📌 1. **Distribution of Cover Types**

* Bar plot to show imbalance across 7 classes
* Identified that some cover types dominate the dataset

### 📌 2. **Elevation Distribution**

* Elevation showed clear separation patterns between cover types

### 📌 3. **Correlation Heatmap**

* Showed strong relationship among Hillshade features and distances

### 📌 4. **Topographic Influence**

* Plotted boxplots of Slope, Aspect vs. Cover Type

### 📌 5. **Soil Type & Wilderness Impact**

* Certain soil types strongly correlated with forest type

---

## 🛠 **Stage 3: Preprocessing**

### 🔹 **Handling Class Imbalance**

Explored handling via:

* Class weights for tree models
* Stratified train-test split

### 🔹 **Train/Test Split**

* **80% training**
* **20% testing**
* Used `stratify=y` to preserve cover type proportions

---

## 🤖 **Stage 4: Model Building & Evaluation**

Multiple classification models were trained and compared to identify the best performer.

### 📌 **Models Implemented**

| Model                          | Why Used                                  |
| ------------------------------ | ----------------------------------------- |
| Logistic Regression            | Baseline linear classifier                |
| Decision Tree Classifier       | Captures non-linear ecology patterns      |
| Random Forest Classifier       | Handles high-dimensional sparse data well |
| HistGradientBoostingClassifier | Excellent for large tabular datasets      |


---

## 📊 **Evaluation Metrics Used**

* Accuracy
* F1-Score (Macro)
* Confusion Matrix
* Cross-Validation Scores

---

## 🔎 **Key Observations**

### Bestb Performer:🌟 **Random Forest**

* Competitive performance
* More interpretable but heavier on memory


### 📌 Interpretation

The dataset is **high-dimensional and imbalanced**, which makes perfect classification challenging.
Models like RFR, XGBoost-style boosters perform significantly better due to:

* Handling sparse features
* Capturing nonlinear ecological relationships



---

## 🔮 **Stage 6: Predictions**

The final tuned model was used to predict cover types for **new ecological data**.

### How Prediction Works:

1. New observation must contain 54 features
2. Data is scaled using the same scaler
3. Final model predicts forest type (1–7)
4. Result used for land cover analysis, ecology management, wildfire planning, etc.

---

## 🔚 **Conclusion**

This project demonstrates an **end-to-end machine learning pipeline** for predicting forest cover type using Scikit-Learn’s `fetch_covtype` dataset.

### 📌 Achievements:

* Performed EDA on a large ecological dataset
* Built multiple ML models
* Achieved strong performance using Random Forest Classifier
* Deployed model for real-world predictions

### 📌 Future Improvements:


✔ Trying advanced models (XGBoost, CatBoost, LightGBM)
✔ Applying SMOTE variants for class imbalance
✔ Using dimensionality reduction (PCA) to speed up training


