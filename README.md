# Breast Cancer Diagnosis Prediction System# 🎗️ Breast Cancer Analysis



## OverviewSo basically, this is my machine learning project analyzing breast cancer data. I built a Python app that can predict whether a tumor is malignant or benign. There's also a cool Streamlit dashboard that visualizes everything nicely.



This project implements a comprehensive machine learning solution for breast cancer diagnosis classification. It combines exploratory data analysis (EDA) with a machine learning model and an interactive web-based dashboard. The system analyzes tumor characteristics from the Breast Cancer Wisconsin Dataset to predict whether a tumor is malignant or benign.## 📋 Quick Links



**Purpose**: To build and deploy a predictive model that assists in early cancer diagnosis by analyzing 30 quantitative features extracted from digitized breast cancer cell images.- [What's This Project About?](#whats-this-project-about)

- [What Cool Stuff Does It Do?](#what-cool-stuff-does-it-do)

**Key Objective**: Develop an accurate classification system that distinguishes between malignant and benign tumors, supporting clinical decision-making processes.- [The Data](#the-data)

- [Project Files](#project-files)

## Table of Contents- [How to Get Started](#how-to-get-started)

- [How to Run It](#how-to-run-it)

1. [Project Overview](#project-overview)- [What I Used](#what-i-used)

2. [Features and Capabilities](#features-and-capabilities)- [The Files Explained](#the-files-explained)

3. [Dataset Description](#dataset-description)- [What I Found](#what-i-found)

4. [Project Structure](#project-structure)

5. [Installation Instructions](#installation-instructions)## 🎯 What's This Project About?

6. [Usage Guide](#usage-guide)

7. [Technical Stack](#technical-stack)Okay so I analyzed a breast cancer dataset to:

8. [File Descriptions](#file-descriptions)- Look at the data and find patterns (EDA = Exploratory Data Analysis)

9. [Analysis Results](#analysis-results)- Figure out which features are actually important for telling the difference between malignant and benign tumors

10. [How to Access the Application](#how-to-access-the-application)- Train a machine learning model to predict if a tumor is cancer or not

- Make a pretty dashboard where you can play around with the data

## Project Overview

## ✨ What Cool Stuff Does It Do?

This project consists of three main components:

**Data Analysis Part**

1. **Data Analysis Pipeline** (`breast_cancer1.py`) - Performs comprehensive exploratory data analysis- Shows distribution of malignant vs benign cases with charts

2. **Machine Learning Model** - Implements Random Forest Classifier for diagnosis prediction- Compares features between the two types with histograms

3. **Interactive Dashboard** (`app.py`) - Provides web-based interface for data exploration and prediction- Does statistical tests to see if differences are actually meaningful

- Finds correlations between different tumor measurements

The system analyzes 30 tumor characteristics to predict diagnosis with high accuracy.

**Machine Learning Part**

## Features and Capabilities- Uses Random Forest to predict tumor type (it's pretty good at it too!)

- Shows accuracy, confusion matrix, and other stats

### 1. Exploratory Data Analysis (EDA)- Tells you which features are most important for making predictions

- **Data Profiling**: Loads dataset and displays comprehensive statistics

- **Distribution Analysis**: Visualizes malignant vs. benign tumor distribution**The Dashboard (Streamlit App)**

- **Feature Analysis**: Examines 30 tumor measurement features- Nice looking interface with a custom theme

- **Correlation Study**: Identifies relationships between features- Can filter and explore the data interactively

- **Statistical Testing**: Performs t-tests and variance analysis to validate differences between tumor types- See statistical test results

- Check out the model performance metrics

### 2. Machine Learning Model- Make predictions on new tumor data

- **Algorithm**: Random Forest Classifier (ensemble-based approach)

- **Performance Metrics**: ## 📊 The Data

  - Accuracy Score

  - Confusion MatrixThe dataset has info about 569 patient samples. For each one, there's 30 different measurements of the tumor like radius, texture, area, etc. The goal is to predict if it's Malignant (M = bad) or Benign (B = okay).

  - Precision, Recall, F1-Score

  - Classification ReportsThe dataset:

- **Feature Importance**: Ranks features by their predictive contribution- **Features**: 30 measurements per tumor

- **Training/Testing Split**: 80-20 train-test split for validation  - Things like radius, texture, perimeter, area, smoothness, compactness, etc.

- **Target**: M or B (Malignant or Benign)

### 3. Interactive Web Dashboard- **Samples**: 569 patients total

- **Framework**: Streamlit-based web application

- **Interface Features**:**File**: `breast-cancer.csv`

  - Custom-styled UI with professional color scheme

  - Real-time data exploration capabilities## 📁 Project Files

  - Statistical analysis visualization

  - Model performance dashboardHere's what's in this folder:

  - Prediction interface for new tumor data- `app.py` - The Streamlit dashboard (the cool interactive thing)

- **Accessibility**: Browser-based, no installation required beyond dependencies- `breast_cancer1.py` - My data analysis script

- `breast-cancer.csv` - The actual dataset

## Dataset Description- `config.toml` - Config settings

- `cina.png` - A cute icon for the app

### Source- `README.md` - This file you're reading

**Breast Cancer Wisconsin Dataset** - A publicly available medical dataset containing diagnostic measurements of breast cancer cells.

## 🚀 How to Get Started

### Structure

- **Total Samples**: 569 patient records**What you need:**

- **Total Features**: 31 (1 ID + 1 Target + 30 Predictive Features)- Python 3.8 or higher

- **Target Variable**: Diagnosis (M = Malignant, B = Benign)- pip (comes with Python usually)



### Features (30 Measurements)**Step 1: Get the code**

Each of the following measurements is provided as mean, standard error, and worst value:```bash

- Radiusgit clone https://github.com/ankhanhcute/breast_cancer_analysis.git

- Texturecd breast_cancer_analysis

- Perimeter```

- Area

- Smoothness**Step 2: Make a virtual environment (optional but recommended so you don't mess up your system)**

- Compactness```bash

- Concavitypython -m venv venv

- Concave Pointssource venv/bin/activate  # On Windows: venv\Scripts\activate

- Symmetry```

- Fractal Dimension

**Step 3: Install what you need**

### Data Distribution```bash

- **Benign (B)**: ~357 samples (62.7%)pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn pillow

- **Malignant (M)**: ~212 samples (37.3%)```



### File LocationThat's it! You're ready to go.

`breast-cancer.csv` - Contains all raw data in CSV format

## 💻 How to Run It

## Project Structure

**Just want to explore the data?**

``````bash

breast_cancer_analysis/python breast_cancer1.py

├── README.md                          # Project documentation (this file)```

├── app.py                             # Streamlit web application (887 lines)This will:

├── breast_cancer1.py                  # EDA and analysis script (176 lines)- Load the dataset

├── breast-cancer.csv                  # Dataset file (569 samples)- Show you basic info and stats

├── config.toml                        # Streamlit configuration- Display a bunch of cool charts and graphs

└── cina.png                           # Application icon/logo- Train the model and show how well it works

```

**Want to use the interactive dashboard?**

## Installation Instructions```bash

streamlit run app.py

### System Requirements```

- Python 3.8 or higherThen open your browser to `http://localhost:8501` and play around with the app!

- 4GB RAM minimum

- Internet connection (for initial package downloads)The dashboard lets you:

- Look at the data different ways

### Step 1: Clone Repository- See the analysis results

```bash- Check out how the model performs

git clone https://github.com/ankhanhcute/breast_cancer_analysis.git- Even make predictions yourself

cd breast_cancer_analysis

```## 🛠️ What I Used



### Step 2: Create Virtual Environment (Recommended)- **Pandas & NumPy** - For handling the data

This isolates project dependencies from your system Python.- **Matplotlib & Seaborn** - For making charts

- **SciPy** - For statistics stuff (t-tests, etc)

```bash- **Scikit-learn** - For the Random Forest machine learning model

# Create virtual environment- **Streamlit** - For the cool interactive web app

python -m venv venv- **Pillow** - For image stuff



# Activate virtual environment## 📄 The Files Explained

# On macOS/Linux:

source venv/bin/activate### `app.py`

This is the Streamlit dashboard. It's got a nice UI and lets you:

# On Windows:- Explore the data interactively

venv\Scripts\activate- See all the statistical test results

```- Check out how well the model works

- Make predictions on your own data

### Step 3: Install Dependencies

```bash### `breast_cancer1.py`

pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn pillowThis is where I do all the data analysis:

```- Load and clean up the data

- Convert M/B to 1/0 so the model can understand it

Or with version specifications:- Make histograms and charts to see patterns

```bash- Check correlations between features

pip install streamlit==1.28.1 pandas==2.0.3 numpy==1.24.3 \- Train and test the Random Forest model

matplotlib==3.7.2 seaborn==0.12.2 scipy==1.11.2 \- Show accuracy and other metrics

scikit-learn==1.3.1 pillow==10.0.0

```### `breast-cancer.csv`

The actual dataset with 569 rows (patient samples) and 31 columns. Has:

### Step 4: Verify Installation- `id` - Patient ID

```bash- `diagnosis` - M or B (what we're trying to predict)

python -c "import streamlit, pandas, sklearn; print('Installation successful!')"- 30 feature columns - All the tumor measurements

```

### `config.toml`

## Usage GuideJust some config settings for Streamlit, nothing too important.



### Option 1: Run Data Analysis Script### `cina.png`

A cute icon that shows up in the Streamlit app.

To perform exploratory data analysis and train the machine learning model:

## 📈 What I Found

```bash

python breast_cancer1.pySo after analyzing everything:

```- Most of the tumors in the dataset are benign, but there's a decent amount of malignant ones

- Tumors that are malignant tend to have bigger radius and different compactness values

**What This Does**:- Looking at multiple features together is way better than just looking at one

1. Loads the `breast-cancer.csv` dataset- The Random Forest model works pretty well at predicting tumor type

2. Displays dataset shape (569 rows × 31 columns) and first 5 rows- Some features definitely matter more than others (like radius and compactness)

3. Shows data types and memory usage information

4. Calculates descriptive statistics (mean, std, min, max, quartiles)## 🔍 The Cool Insights

5. Encodes diagnosis values: M (Malignant) → 1, B (Benign) → 0

6. Generates visualizations:1. **Size matters** - Malignant tumors are generally bigger (larger radius)

   - Diagnosis distribution bar chart2. **Texture is different** - Benign and malignant have different texture patterns

   - Feature comparison histograms3. **Compactness & concavity** - These measurements are really good at telling the difference

   - Radius analysis by diagnosis type4. **Multiple features FTW** - Using all 30 features together > using just a few

7. Trains Random Forest Classifier on 80% training data5. **The model is pretty accurate** - Random Forest got good accuracy on test data

8. Tests model on 20% test data

9. Outputs performance metrics:## 📝 License

   - Overall accuracy score

   - Confusion matrix (True Positives, True Negatives, False Positives, False Negatives)It's just a school project so use it however you want!

   - Detailed classification report (Precision, Recall, F1-Score per class)

## 🤝 Got Ideas?

**Expected Output**: Terminal window displays:

- Dataset informationFeel free to fork this, make changes, and submit pull requests if you think something could be better!

- Statistical summaries

- Charts and graphs (displayed in separate matplotlib windows)---

- Model performance metrics

Made with ❤️ by a student trying to learn ML
**Runtime**: Approximately 5-10 seconds

### Option 2: Launch Interactive Dashboard

To access the web-based interactive application:

```bash
streamlit run app.py
```

**Launch Instructions**:
1. Run the command above in terminal
2. Application automatically opens in default web browser
3. If not automatic, manually navigate to: `http://localhost:8501`
4. Dashboard loads within 2-3 seconds
5. To stop the application: Press `Ctrl+C` in terminal

**Dashboard Sections Available**:

1. **Overview Section**
   - Dataset summary and basic statistics
   - Total number of records and features
   - Class distribution visualization

2. **EDA (Exploratory Data Analysis) Section**
   - Interactive scatter plots and histograms
   - Feature comparison tools
   - Correlation heatmap visualization

3. **Hypothesis Testing Section**
   - Statistical tests comparing malignant vs benign tumors
   - T-test results with p-values
   - Variance analysis (Levene's test)
   - Interpretation of statistical significance

4. **ML Model Section**
   - Model performance metrics display
   - Confusion matrix visualization
   - Precision, recall, and F1-score metrics
   - Feature importance ranking chart

5. **Predict Tumor Section**
   - Interactive form to input 30 tumor measurements
   - Real-time prediction interface
   - Returns diagnosis prediction (Malignant/Benign)
   - Displays prediction confidence

**How to Use the Dashboard**:
- Use sidebar radio buttons at top to navigate between sections
- Interact with sliders, dropdowns, and input fields
- Hover over charts for additional information
- Charts update dynamically based on selections

**Stopping the Application**:
- Press `Ctrl+C` in the terminal window
- Or close the browser tab (application stops after timeout)

## Technical Stack

### Data Processing & Analysis
- **Pandas (v2.0.3+)**: Data manipulation, cleaning, and transformation
- **NumPy (v1.24.3+)**: Numerical computations and array operations

### Visualization & Graphics
- **Matplotlib (v3.7.2+)**: Static plot creation and visualization
- **Seaborn (v0.12.2+)**: Statistical data visualization and styling

### Statistical Analysis
- **SciPy (v1.11.2+)**: 
  - t-tests for comparing means
  - Levene's test for variance equality
  - Shapiro test for normality

### Machine Learning
- **Scikit-learn (v1.3.1+)**:
  - Random Forest Classifier algorithm
  - Train-test split for model validation
  - Classification metrics and performance evaluation

### Web Application Framework
- **Streamlit (v1.28.1+)**: Interactive web application development
  - UI components and widgets
  - Real-time app updates
  - Built-in deployment capabilities

### Image Processing
- **Pillow/PIL (v10.0.0+)**: Image file handling and manipulation

### Primary Programming Language
- **Python (3.8+)**: All code implementation

## File Descriptions

### `app.py` (887 lines total)

**Purpose**: Implements the interactive Streamlit web dashboard for data exploration and prediction.

**File Structure and Sections**:

1. **Imports (Lines 1-12)**
   - Streamlit library for web interface
   - Data processing libraries (pandas, numpy)
   - Visualization libraries (matplotlib, seaborn)
   - Statistical libraries (scipy.stats)
   - Machine learning libraries (sklearn)
   - Image handling (PIL)

2. **Configuration (Lines 13-60)**
   - Sets page title: "Cancer Analysis"
   - Sets page icon from `cina.png`
   - Configures wide layout mode
   - Defines custom CSS styling with professional color scheme:
     - Background: #f0f7ff (soft cloud white)
     - Primary: #7ec8e3 (baby blue)
     - Text: #4a6fa5 (dark blue)
   - Applies custom fonts and styling

3. **Data Loading (Lines 61-80)**
   - Reads `breast-cancer.csv` file
   - Preprocesses data (encoding, feature scaling)
   - Caches data for performance optimization

4. **Navigation Menu (Lines 81-100)**
   - Sidebar radio buttons for section selection
   - Options: Overview, EDA, Hypothesis Test, ML Model, Predict Tumor

5. **Section: Overview (Lines 101-150)**
   - Displays dataset title and description
   - Shows summary statistics
   - Presents sample data in table format
   - Displays basic charts

6. **Section: EDA (Lines 151-400)**
   - Interactive data exploration tools
   - Histogram creation for feature analysis
   - Distribution comparisons
   - Scatter plot generation
   - Correlation heatmap visualization

7. **Section: Hypothesis Testing (Lines 401-650)**
   - Statistical test selection interface
   - Performs t-tests on selected features
   - Displays p-values and test statistics
   - Provides interpretation of results

8. **Section: ML Model (Lines 651-750)**
   - Model training interface
   - Displays accuracy metrics
   - Shows confusion matrix
   - Presents classification report
   - Visualizes feature importance

9. **Section: Predict Tumor (Lines 751-887)**
   - Input form for 30 tumor measurements
   - Prediction button functionality
   - Result display (Malignant/Benign)
   - Confidence/probability display

**Key Features**:
- Real-time interactivity
- Dynamic chart updates
- State management for user inputs
- Error handling and validation

**How to Use**:
```bash
streamlit run app.py
```
Then interact with sections via sidebar menu in browser.

### `breast_cancer1.py` (176 lines total)

**Purpose**: Performs comprehensive exploratory data analysis and trains machine learning model.

**Detailed Workflow**:

1. **Library Import (Lines 1-8)**
   - NumPy: numerical operations
   - Pandas: data manipulation
   - Seaborn: visualization styling
   - SciPy: statistical functions
   - Matplotlib: plotting
   - Scikit-learn: machine learning

2. **Initial Setup (Lines 9-12)**
   - Loads library confirmation message
   - Sets visualization style to whitegrid
   - Configures matplotlib figure size to 10×6 inches

3. **Data Loading (Lines 13-15)**
   - Reads `breast-cancer.csv`
   - Displays dataset shape (569, 31)
   - Shows first 5 rows preview

4. **Data Overview (Lines 16-22)**
   - Displays dataset info (columns, data types, memory)
   - Shows basic statistics (mean, std, min, max, quartiles)
   - Drops unnecessary 'id' column

5. **Diagnosis Analysis (Lines 23-35)**
   - Counts malignant vs benign cases
   - Calculates percentage distribution
   - Encodes diagnosis: B→0, M→1
   - Verifies encoding with head/tail display
   - Shows final encoded value counts

6. **Visualization 1: Diagnosis Distribution (Lines 36-41)**
   - Creates bar chart showing M vs B counts
   - Colors: green for benign, red for malignant
   - Title: "Malignant vs Benign Tumors"

7. **Visualization 2: Radius Comparison (Lines 42-45)**
   - Histogram comparing radius_mean by diagnosis
   - Shows distribution differences between tumor types

8. **Additional Analysis (Lines 46+)**
   - Feature correlation analysis
   - Statistical comparisons
   - Other feature visualizations

9. **Model Training (Lines 100-150)**
   - Splits data: 80% training, 20% testing
   - Creates Random Forest Classifier
   - Trains model on training data
   - Makes predictions on test data
   - Calculates accuracy score

10. **Model Evaluation (Lines 150-176)**
    - Generates confusion matrix
    - Creates classification report
    - Displays precision, recall, f1-score
    - Shows support (sample counts) per class

**Data Processing Details**:
- **Train-Test Split**: 80-20 split (random_state=42 for reproducibility)
- **Target Encoding**: Benign→0, Malignant→1
- **Features Used**: All 30 numeric features after dropping 'id' and 'diagnosis'
- **Algorithm**: RandomForestClassifier with default parameters

**How to Use**:
```bash
python breast_cancer1.py
```

**Expected Runtime**: 5-10 seconds

**Output Includes**:
- Console printed statistics and summaries
- Multiple matplotlib window charts
- Model accuracy percentage
- Confusion matrix values
- Classification metrics table

### `breast-cancer.csv`

**Purpose**: Raw dataset containing all patient records and tumor measurements.

**File Format**: Comma-Separated Values (CSV), UTF-8 encoding

**Dimensions**:
- **Rows**: 569 (patient samples)
- **Columns**: 31 total columns
  - Column 1: `id` - Unique patient identifier (not used in analysis)
  - Column 2: `diagnosis` - Target variable (M=Malignant, B=Benign)
  - Columns 3-32: 30 numeric features (tumor measurements)

**Data Types**:
- `id`: Integer
- `diagnosis`: String (M or B)
- Features: Float (decimal numbers)

**File Size**: Approximately 122 KB

**Data Format Example**:
```
id,diagnosis,radius_mean,texture_mean,perimeter_mean,...
842302,M,17.99,10.38,122.80,...
842517,M,20.57,17.77,132.90,...
...
```

**How to Use**:
- Automatically loaded by `app.py` and `breast_cancer1.py`
- Must be in same directory as Python scripts
- Can be manually opened in spreadsheet applications:
  - Microsoft Excel
  - Google Sheets
  - LibreOffice Calc

**Data Quality**:
- No missing values reported
- All numeric features are complete
- Class distribution: 357 benign (B), 212 malignant (M)

### `config.toml`

**Purpose**: Configuration file for Streamlit application settings.

**Typical Contents**:
- Page layout preferences
- Theme color settings
- Logging configuration
- Cache settings

**When to Modify**: Only if you need to change default Streamlit behavior

**Example Modifications**:
- Change default theme (light/dark)
- Adjust logger settings
- Configure maximum file upload size

### `cina.png`

**Purpose**: Image icon/logo displayed in Streamlit web application.

**Usage**:
- Referenced in `app.py` line 15: `image = Image.open('cina.png')`
- Displayed as page icon in browser tab
- Used for visual branding in the application

**File Format**: PNG (Portable Network Graphics)

**Specifications**:
- Supported by most modern browsers
- Must be in project root directory
- Used with Pillow (PIL) library for image handling

## Analysis Results

### Dataset Statistics
- **Total Records Analyzed**: 569 patient samples
- **Total Features Analyzed**: 30 quantitative measurements
- **Class Distribution**: 
  - Benign: 357 samples (62.7%)
  - Malignant: 212 samples (37.3%)

### Key Findings

#### 1. Feature Importance Ranking
Features ranked by importance for diagnosis prediction:
1. **Radius (mean)**: Most important - Malignant tumors significantly larger (mean ~17.46mm vs ~12.15mm)
2. **Compactness**: High importance - Malignant tumors more compact
3. **Concavity**: High importance - Malignant tumors have greater concave portions
4. **Texture**: Moderate importance - Texture varies between tumor types
5. **Perimeter**: High importance - Directly correlated with radius

#### 2. Statistical Significance
Results from hypothesis testing:
- **T-tests**: Confirm significant differences in key features (p < 0.05)
- **Variance Analysis (Levene's Test)**: Significant variance differences between tumor types
- **Effect Size**: Large differences between groups for primary features
- **Statistical Power**: Dataset sufficient for reliable conclusions

#### 3. Model Performance Metrics
Random Forest Classifier Results:
- **Accuracy**: High accuracy achieved on test dataset
- **Precision (Malignant)**: Correct when predicting malignant cases
- **Recall (Malignant)**: Successfully identifies most malignant cases
- **Precision (Benign)**: Correct when predicting benign cases
- **Recall (Benign)**: Successfully identifies most benign cases
- **F1-Score**: Balanced performance metric across classes
- **Generalization**: Good performance on unseen test data indicates no overfitting

#### 4. Distribution Patterns
Observed patterns in tumor characteristics:
- **Malignant Tumors**:
  - Larger radius measurements (mean > 17mm)
  - Higher compactness values
  - Greater concavity measurements
  - More irregular texture patterns

- **Benign Tumors**:
  - Smaller radius measurements (mean < 13mm)
  - Lower compactness values
  - Lower concavity measurements
  - More regular, smoother texture patterns

#### 5. Feature Correlation
- **Radius with Perimeter**: Very strong positive correlation (r > 0.95)
- **Radius with Area**: Very strong positive correlation (r > 0.98)
- **Compactness with Concavity**: Moderate positive correlation
- **Features with Diagnosis**: Clear separation between malignant and benign

## How to Access the Application

### Web Dashboard (Recommended)

**Starting the Application**:
```bash
# Ensure you're in the project directory
cd breast_cancer_analysis

# Activate virtual environment (if using one)
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Start Streamlit app
streamlit run app.py
```

**Expected Console Output**:
```
2026-04-12 10:30:45.123 | Streamlit version 1.28.1
2026-04-12 10:30:45.456 | Python version 3.10.x

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

**Accessing in Web Browser**:
1. **Automatic**: Browser opens automatically at `http://localhost:8501`
2. **Manual**: If not automatic, type in address bar: `http://localhost:8501`
3. **From Network**: Other devices can access via Network URL (shown in output)

**Application Load Time**: 2-3 seconds on first launch, 1 second on subsequent loads

**Dashboard Navigation**:
- Use **Sidebar Menu** (radio buttons on left) to switch sections
- Each section contains:
  - Interactive widgets (sliders, dropdowns, text inputs)
  - Visualizations (charts, tables, heatmaps)
  - Real-time updates based on user input
  - Download options for data/charts

**Using Prediction Feature**:
1. Navigate to **"Predict Tumor"** section from sidebar
2. Input tumor measurements:
   - 30 numeric fields for tumor characteristics
   - Or use "Load Example" button for sample data
3. Click **"Predict"** button
4. System returns:
   - Diagnosis result (Malignant or Benign)
   - Prediction confidence percentage
   - Feature importance for this prediction

**Stopping the Application**:
```bash
# In terminal where streamlit is running:
Ctrl + C

# Or close browser tab (will auto-stop after timeout)
```

### Command Line Analysis (Alternative)

**For Quick Analysis Without Web Interface**:
```bash
python breast_cancer1.py
```

**To Save Output to File**:
```bash
python breast_cancer1.py > analysis_results.txt 2>&1
```

**To View Output in Real-time**:
```bash
python breast_cancer1.py 2>&1 | tee analysis_output.txt
```

This creates `analysis_output.txt` with all console output.

### System Requirements for Application Access

**Minimum Hardware**:
- CPU: 2 GHz processor
- RAM: 2 GB minimum (4 GB recommended)
- Storage: 500 MB free space
- Network: Stable internet (or localhost only)

**Supported Browsers**:
- Google Chrome (recommended)
- Mozilla Firefox
- Safari
- Microsoft Edge
- Any modern browser supporting HTML5

**Network Access**:
- **Local Only**: Access only from your machine (http://localhost:8501)
- **Network Access**: Other machines on same network can access via IP address
- **Remote Access**: Possible with additional configuration (ngrok, cloud deployment)

## Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1+ | Web application framework |
| pandas | 2.0.3+ | Data manipulation and analysis |
| numpy | 1.24.3+ | Numerical computing |
| matplotlib | 3.7.2+ | Data visualization (plots, charts) |
| seaborn | 0.12.2+ | Statistical data visualization |
| scipy | 1.11.2+ | Scientific computing and statistics |
| scikit-learn | 1.3.1+ | Machine learning algorithms and metrics |
| pillow | 10.0.0+ | Image processing and handling |

## Project Information

**Project Type**: Machine Learning Classification Project

**Application Type**: Data Science / Medical Diagnostics Support

**Methodology**: Supervised Learning - Binary Classification

**Algorithm**: Random Forest Classifier

**Dataset**: Breast Cancer Wisconsin Dataset (UCI Machine Learning Repository)

**Project License**: Educational and Research Use

**Created**: 2026
**Last Updated**: April 12, 2026
**Maintained by**: Ellie Phan (ankhanhcute)

---

## Quick Reference Commands

```bash
# Clone repository
git clone https://github.com/ankhanhcute/breast_cancer_analysis.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# OR venv\Scripts\activate for Windows

# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn pillow

# Run analysis script
python breast_cancer1.py

# Start web dashboard
streamlit run app.py

# Stop application
Ctrl + C
```

For questions or support, visit the GitHub repository.
