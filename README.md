# Breast Cancer Diagnosis Prediction System# Breast Cancer Diagnosis Prediction System# 🎗️ Breast Cancer Analysis



## Overview



This project implements a comprehensive machine learning solution for breast cancer diagnosis classification. It combines exploratory data analysis (EDA) with a machine learning model and an interactive web-based dashboard. The system analyzes tumor characteristics from the Breast Cancer Wisconsin Dataset to predict whether a tumor is malignant or benign.## OverviewSo basically, this is my machine learning project analyzing breast cancer data. I built a Python app that can predict whether a tumor is malignant or benign. There's also a cool Streamlit dashboard that visualizes everything nicely.



**Purpose**: To build and deploy a predictive model that assists in early cancer diagnosis by analyzing 30 quantitative features extracted from digitized breast cancer cell images.



**Key Objective**: Develop an accurate classification system that distinguishes between malignant and benign tumors, supporting clinical decision-making processes.This project implements a comprehensive machine learning solution for breast cancer diagnosis classification. It combines exploratory data analysis (EDA) with a machine learning model and an interactive web-based dashboard. The system analyzes tumor characteristics from the Breast Cancer Wisconsin Dataset to predict whether a tumor is malignant or benign.## 📋 Quick Links



## Table of Contents



1. [Project Overview](#project-overview)**Purpose**: To build and deploy a predictive model that assists in early cancer diagnosis by analyzing 30 quantitative features extracted from digitized breast cancer cell images.- [What's This Project About?](#whats-this-project-about)

2. [Features and Capabilities](#features-and-capabilities)

3. [Dataset Description](#dataset-description)- [What Cool Stuff Does It Do?](#what-cool-stuff-does-it-do)

4. [Project Structure](#project-structure)

5. [Installation Instructions](#installation-instructions)**Key Objective**: Develop an accurate classification system that distinguishes between malignant and benign tumors, supporting clinical decision-making processes.- [The Data](#the-data)

6. [Usage Guide](#usage-guide)

7. [Technical Stack](#technical-stack)- [Project Files](#project-files)

8. [File Descriptions](#file-descriptions)

9. [Analysis Results](#analysis-results)## Table of Contents- [How to Get Started](#how-to-get-started)

10. [How to Access the Application](#how-to-access-the-application)

- [How to Run It](#how-to-run-it)

## Project Overview

1. [Project Overview](#project-overview)- [What I Used](#what-i-used)

This project consists of three main components:

2. [Features and Capabilities](#features-and-capabilities)- [The Files Explained](#the-files-explained)

1. **Data Analysis Pipeline** (`breast_cancer1.py`) - Performs comprehensive exploratory data analysis

2. **Machine Learning Model** - Implements Random Forest Classifier for diagnosis prediction3. [Dataset Description](#dataset-description)- [What I Found](#what-i-found)

3. **Interactive Dashboard** (`app.py`) - Provides web-based interface for data exploration and prediction

4. [Project Structure](#project-structure)

The system analyzes 30 tumor characteristics to predict diagnosis with high accuracy.

5. [Installation Instructions](#installation-instructions)## 🎯 What's This Project About?

## Features and Capabilities

6. [Usage Guide](#usage-guide)

### 1. Exploratory Data Analysis (EDA)

- **Data Profiling**: Loads dataset and displays comprehensive statistics7. [Technical Stack](#technical-stack)Okay so I analyzed a breast cancer dataset to:

- **Distribution Analysis**: Visualizes malignant vs. benign tumor distribution

- **Feature Analysis**: Examines 30 tumor measurement features8. [File Descriptions](#file-descriptions)- Look at the data and find patterns (EDA = Exploratory Data Analysis)

- **Correlation Study**: Identifies relationships between features

- **Statistical Testing**: Performs t-tests and variance analysis to validate differences between tumor types9. [Analysis Results](#analysis-results)- Figure out which features are actually important for telling the difference between malignant and benign tumors



### 2. Machine Learning Model10. [How to Access the Application](#how-to-access-the-application)- Train a machine learning model to predict if a tumor is cancer or not

- **Algorithm**: Random Forest Classifier (ensemble-based approach)

- **Performance Metrics**: - Make a pretty dashboard where you can play around with the data

  - Accuracy Score

  - Confusion Matrix## Project Overview

  - Precision, Recall, F1-Score

  - Classification Reports## ✨ What Cool Stuff Does It Do?

- **Feature Importance**: Ranks features by their predictive contribution

- **Training/Testing Split**: 80-20 train-test split for validationThis project consists of three main components:



### 3. Interactive Web Dashboard**Data Analysis Part**

- **Framework**: Streamlit-based web application

- **Interface Features**:1. **Data Analysis Pipeline** (`breast_cancer1.py`) - Performs comprehensive exploratory data analysis- Shows distribution of malignant vs benign cases with charts

  - Custom-styled UI with professional color scheme

  - Real-time data exploration capabilities2. **Machine Learning Model** - Implements Random Forest Classifier for diagnosis prediction- Compares features between the two types with histograms

  - Statistical analysis visualization

  - Model performance dashboard3. **Interactive Dashboard** (`app.py`) - Provides web-based interface for data exploration and prediction- Does statistical tests to see if differences are actually meaningful

  - Prediction interface for new tumor data

- **Accessibility**: Browser-based, no installation required beyond dependencies- Finds correlations between different tumor measurements



## Dataset DescriptionThe system analyzes 30 tumor characteristics to predict diagnosis with high accuracy.



### Source**Machine Learning Part**

**Breast Cancer Wisconsin Dataset** - A publicly available medical dataset containing diagnostic measurements of breast cancer cells.

## Features and Capabilities- Uses Random Forest to predict tumor type (it's pretty good at it too!)

### Structure

- **Total Samples**: 569 patient records- Shows accuracy, confusion matrix, and other stats

- **Total Features**: 31 (1 ID + 1 Target + 30 Predictive Features)

- **Target Variable**: Diagnosis (M = Malignant, B = Benign)### 1. Exploratory Data Analysis (EDA)- Tells you which features are most important for making predictions



### Features (30 Measurements)- **Data Profiling**: Loads dataset and displays comprehensive statistics

Each of the following measurements is provided as mean, standard error, and worst value:

- Radius- **Distribution Analysis**: Visualizes malignant vs. benign tumor distribution**The Dashboard (Streamlit App)**

- Texture

- Perimeter- **Feature Analysis**: Examines 30 tumor measurement features- Nice looking interface with a custom theme

- Area

- Smoothness- **Correlation Study**: Identifies relationships between features- Can filter and explore the data interactively

- Compactness

- Concavity- **Statistical Testing**: Performs t-tests and variance analysis to validate differences between tumor types- See statistical test results

- Concave Points

- Symmetry- Check out the model performance metrics

- Fractal Dimension

### 2. Machine Learning Model- Make predictions on new tumor data

### Data Distribution

- **Benign (B)**: ~357 samples (62.7%)- **Algorithm**: Random Forest Classifier (ensemble-based approach)

- **Malignant (M)**: ~212 samples (37.3%)

- **Performance Metrics**: ## 📊 The Data

### File Location

`breast-cancer.csv` - Contains all raw data in CSV format  - Accuracy Score



## Project Structure  - Confusion MatrixThe dataset has info about 569 patient samples. For each one, there's 30 different measurements of the tumor like radius, texture, area, etc. The goal is to predict if it's Malignant (M = bad) or Benign (B = okay).



```  - Precision, Recall, F1-Score

breast_cancer_analysis/

├── README.md                          # Project documentation (this file)  - Classification ReportsThe dataset:

├── app.py                             # Streamlit web application (887 lines)

├── breast_cancer1.py                  # EDA and analysis script (176 lines)- **Feature Importance**: Ranks features by their predictive contribution- **Features**: 30 measurements per tumor

├── breast-cancer.csv                  # Dataset file (569 samples)

├── config.toml                        # Streamlit configuration- **Training/Testing Split**: 80-20 train-test split for validation  - Things like radius, texture, perimeter, area, smoothness, compactness, etc.

└── cina.png                           # Application icon/logo

```- **Target**: M or B (Malignant or Benign)



## Installation Instructions### 3. Interactive Web Dashboard- **Samples**: 569 patients total



### System Requirements- **Framework**: Streamlit-based web application

- Python 3.8 or higher

- 4GB RAM minimum- **Interface Features**:**File**: `breast-cancer.csv`

- Internet connection (for initial package downloads)

  - Custom-styled UI with professional color scheme

### Step 1: Clone Repository

```bash  - Real-time data exploration capabilities## 📁 Project Files

git clone https://github.com/ankhanhcute/breast_cancer_analysis.git

cd breast_cancer_analysis  - Statistical analysis visualization

```

  - Model performance dashboardHere's what's in this folder:

### Step 2: Create Virtual Environment (Recommended)

This isolates project dependencies from your system Python.  - Prediction interface for new tumor data- `app.py` - The Streamlit dashboard (the cool interactive thing)



```bash- **Accessibility**: Browser-based, no installation required beyond dependencies- `breast_cancer1.py` - My data analysis script

# Create virtual environment

python -m venv venv- `breast-cancer.csv` - The actual dataset



# Activate virtual environment## Dataset Description- `config.toml` - Config settings

# On macOS/Linux:

source venv/bin/activate- `cina.png` - A cute icon for the app



# On Windows:### Source- `README.md` - This file you're reading

venv\Scripts\activate

```**Breast Cancer Wisconsin Dataset** - A publicly available medical dataset containing diagnostic measurements of breast cancer cells.



### Step 3: Install Dependencies## 🚀 How to Get Started

```bash

pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn pillow### Structure

```

- **Total Samples**: 569 patient records**What you need:**

Or with version specifications:

```bash- **Total Features**: 31 (1 ID + 1 Target + 30 Predictive Features)- Python 3.8 or higher

pip install streamlit==1.28.1 pandas==2.0.3 numpy==1.24.3 \

matplotlib==3.7.2 seaborn==0.12.2 scipy==1.11.2 \- **Target Variable**: Diagnosis (M = Malignant, B = Benign)- pip (comes with Python usually)

scikit-learn==1.3.1 pillow==10.0.0

```



### Step 4: Verify Installation### Features (30 Measurements)**Step 1: Get the code**

```bash

python -c "import streamlit, pandas, sklearn; print('Installation successful!')"Each of the following measurements is provided as mean, standard error, and worst value:```bash

```

- Radiusgit clone https://github.com/ankhanhcute/breast_cancer_analysis.git

## Usage Guide

- Texturecd breast_cancer_analysis

### Option 1: Run Data Analysis Script

- Perimeter```

To perform exploratory data analysis and train the machine learning model:

- Area

```bash

python breast_cancer1.py- Smoothness**Step 2: Make a virtual environment (optional but recommended so you don't mess up your system)**

```

- Compactness```bash

**What This Does**:

1. Loads the `breast-cancer.csv` dataset- Concavitypython -m venv venv

2. Displays dataset shape (569 rows × 31 columns) and first 5 rows

3. Shows data types and memory usage information- Concave Pointssource venv/bin/activate  # On Windows: venv\Scripts\activate

4. Calculates descriptive statistics (mean, std, min, max, quartiles)

5. Encodes diagnosis values: M (Malignant) → 1, B (Benign) → 0- Symmetry```

6. Generates visualizations:

   - Diagnosis distribution bar chart- Fractal Dimension

   - Feature comparison histograms

   - Radius analysis by diagnosis type**Step 3: Install what you need**

7. Trains Random Forest Classifier on 80% training data

8. Tests model on 20% test data### Data Distribution```bash

9. Outputs performance metrics:

   - Overall accuracy score- **Benign (B)**: ~357 samples (62.7%)pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn pillow

   - Confusion matrix (True Positives, True Negatives, False Positives, False Negatives)

   - Detailed classification report (Precision, Recall, F1-Score per class)- **Malignant (M)**: ~212 samples (37.3%)```



**Expected Output**: Terminal window displays statistics, charts, and model performance metrics



**Runtime**: Approximately 5-10 seconds### File LocationThat's it! You're ready to go.



### Option 2: Launch Interactive Dashboard`breast-cancer.csv` - Contains all raw data in CSV format



To access the web-based interactive application:## 💻 How to Run It



```bash## Project Structure

streamlit run app.py

```**Just want to explore the data?**



**Launch Instructions**:``````bash

1. Run the command above in terminal

2. Application automatically opens in default web browserbreast_cancer_analysis/python breast_cancer1.py

3. If not automatic, manually navigate to: `http://localhost:8501`

4. Dashboard loads within 2-3 seconds├── README.md                          # Project documentation (this file)```

5. To stop the application: Press `Ctrl+C` in terminal

├── app.py                             # Streamlit web application (887 lines)This will:

**Dashboard Sections Available**:

├── breast_cancer1.py                  # EDA and analysis script (176 lines)- Load the dataset

1. **Overview Section**

   - Dataset summary and basic statistics├── breast-cancer.csv                  # Dataset file (569 samples)- Show you basic info and stats

   - Total number of records and features

   - Class distribution visualization├── config.toml                        # Streamlit configuration- Display a bunch of cool charts and graphs



2. **EDA (Exploratory Data Analysis) Section**└── cina.png                           # Application icon/logo- Train the model and show how well it works

   - Interactive scatter plots and histograms

   - Feature comparison tools```

   - Correlation heatmap visualization

**Want to use the interactive dashboard?**

3. **Hypothesis Testing Section**

   - Statistical tests comparing malignant vs benign tumors## Installation Instructions```bash

   - T-test results with p-values

   - Variance analysis (Levene's test)streamlit run app.py

   - Interpretation of statistical significance

### System Requirements```

4. **ML Model Section**

   - Model performance metrics display- Python 3.8 or higherThen open your browser to `http://localhost:8501` and play around with the app!

   - Confusion matrix visualization

   - Precision, recall, and F1-score metrics- 4GB RAM minimum

   - Feature importance ranking chart

- Internet connection (for initial package downloads)The dashboard lets you:

5. **Predict Tumor Section**

   - Interactive form to input 30 tumor measurements- Look at the data different ways

   - Real-time prediction interface

   - Returns diagnosis prediction (Malignant/Benign)### Step 1: Clone Repository- See the analysis results

   - Displays prediction confidence

```bash- Check out how the model performs

**How to Use the Dashboard**:

- Use sidebar radio buttons at top to navigate between sectionsgit clone https://github.com/ankhanhcute/breast_cancer_analysis.git- Even make predictions yourself

- Interact with sliders, dropdowns, and input fields

- Hover over charts for additional informationcd breast_cancer_analysis

- Charts update dynamically based on selections

```## 🛠️ What I Used

**Stopping the Application**:

- Press `Ctrl+C` in the terminal window

- Or close the browser tab (application stops after timeout)

### Step 2: Create Virtual Environment (Recommended)- **Pandas & NumPy** - For handling the data

## Technical Stack

This isolates project dependencies from your system Python.- **Matplotlib & Seaborn** - For making charts

### Data Processing & Analysis

- **Pandas (v2.0.3+)**: Data manipulation, cleaning, and transformation- **SciPy** - For statistics stuff (t-tests, etc)

- **NumPy (v1.24.3+)**: Numerical computations and array operations

```bash- **Scikit-learn** - For the Random Forest machine learning model

### Visualization & Graphics

- **Matplotlib (v3.7.2+)**: Static plot creation and visualization# Create virtual environment- **Streamlit** - For the cool interactive web app

- **Seaborn (v0.12.2+)**: Statistical data visualization and styling

python -m venv venv- **Pillow** - For image stuff

### Statistical Analysis

- **SciPy (v1.11.2+)**: 

  - t-tests for comparing means

  - Levene's test for variance equality# Activate virtual environment## 📄 The Files Explained

  - Shapiro test for normality

# On macOS/Linux:

### Machine Learning

- **Scikit-learn (v1.3.1+)**:source venv/bin/activate### `app.py`

  - Random Forest Classifier algorithm

  - Train-test split for model validationThis is the Streamlit dashboard. It's got a nice UI and lets you:

  - Classification metrics and performance evaluation

# On Windows:- Explore the data interactively

### Web Application Framework

- **Streamlit (v1.28.1+)**: Interactive web application developmentvenv\Scripts\activate- See all the statistical test results

  - UI components and widgets

  - Real-time app updates```- Check out how well the model works

  - Built-in deployment capabilities

- Make predictions on your own data

### Image Processing

- **Pillow/PIL (v10.0.0+)**: Image file handling and manipulation### Step 3: Install Dependencies



### Primary Programming Language```bash### `breast_cancer1.py`

- **Python (3.8+)**: All code implementation

pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn pillowThis is where I do all the data analysis:

## File Descriptions

```- Load and clean up the data

### `app.py` (887 lines total)

- Convert M/B to 1/0 so the model can understand it

**Purpose**: Implements the interactive Streamlit web dashboard for data exploration and prediction.

Or with version specifications:- Make histograms and charts to see patterns

**Key Sections**:

1. **Lines 1-12**: Library imports (streamlit, pandas, numpy, matplotlib, seaborn, scipy, sklearn, PIL)```bash- Check correlations between features

2. **Lines 13-60**: Page configuration and custom CSS styling

3. **Lines 61-80**: Data loading and preprocessingpip install streamlit==1.28.1 pandas==2.0.3 numpy==1.24.3 \- Train and test the Random Forest model

4. **Lines 81-100**: Navigation menu with radio buttons

5. **Lines 101-150**: Overview section displaying dataset summarymatplotlib==3.7.2 seaborn==0.12.2 scipy==1.11.2 \- Show accuracy and other metrics

6. **Lines 151-400**: EDA section with interactive visualizations

7. **Lines 401-650**: Hypothesis Testing section with statistical testsscikit-learn==1.3.1 pillow==10.0.0

8. **Lines 651-750**: ML Model section showing performance metrics

9. **Lines 751-887**: Predict Tumor section for user predictions```### `breast-cancer.csv`



**Key Features**:The actual dataset with 569 rows (patient samples) and 31 columns. Has:

- Real-time interactivity

- Dynamic chart updates### Step 4: Verify Installation- `id` - Patient ID

- State management for user inputs

- Error handling and validation```bash- `diagnosis` - M or B (what we're trying to predict)

- Custom color scheme (soft blues and professional styling)

python -c "import streamlit, pandas, sklearn; print('Installation successful!')"- 30 feature columns - All the tumor measurements

**How to Use**:

```bash```

streamlit run app.py

```### `config.toml`



### `breast_cancer1.py` (176 lines total)## Usage GuideJust some config settings for Streamlit, nothing too important.



**Purpose**: Performs comprehensive exploratory data analysis and trains machine learning model.



**Detailed Workflow**:### Option 1: Run Data Analysis Script### `cina.png`



1. **Lines 1-8**: Import required librariesA cute icon that shows up in the Streamlit app.

2. **Lines 9-12**: Initialize visualization settings

3. **Lines 13-15**: Load dataset from CSV fileTo perform exploratory data analysis and train the machine learning model:

4. **Lines 16-22**: Display dataset overview (shape, info, statistics)

5. **Lines 23-35**: Diagnosis analysis and encoding (M→1, B→0)## 📈 What I Found

6. **Lines 36-41**: Create diagnosis distribution visualization

7. **Lines 42-100+**: Generate feature analysis charts and comparisons```bash

8. **Lines 100-150**: Train Random Forest Classifier (80-20 split)

9. **Lines 150-176**: Evaluate model and display metricspython breast_cancer1.pySo after analyzing everything:



**Data Processing Details**:```- Most of the tumors in the dataset are benign, but there's a decent amount of malignant ones

- **Train-Test Split**: 80% training, 20% testing (random_state=42)

- **Target Encoding**: Benign→0, Malignant→1- Tumors that are malignant tend to have bigger radius and different compactness values

- **Features Used**: All 30 numeric features (excluding 'id' and 'diagnosis')

- **Algorithm**: RandomForestClassifier with default parameters**What This Does**:- Looking at multiple features together is way better than just looking at one



**How to Use**:1. Loads the `breast-cancer.csv` dataset- The Random Forest model works pretty well at predicting tumor type

```bash

python breast_cancer1.py2. Displays dataset shape (569 rows × 31 columns) and first 5 rows- Some features definitely matter more than others (like radius and compactness)

```

3. Shows data types and memory usage information

**Output Includes**:

- Console printed statistics and summaries4. Calculates descriptive statistics (mean, std, min, max, quartiles)## 🔍 The Cool Insights

- Multiple matplotlib window charts

- Model accuracy percentage5. Encodes diagnosis values: M (Malignant) → 1, B (Benign) → 0

- Confusion matrix values

- Classification metrics table6. Generates visualizations:1. **Size matters** - Malignant tumors are generally bigger (larger radius)



### `breast-cancer.csv`   - Diagnosis distribution bar chart2. **Texture is different** - Benign and malignant have different texture patterns



**Purpose**: Raw dataset containing all patient records and tumor measurements.   - Feature comparison histograms3. **Compactness & concavity** - These measurements are really good at telling the difference



**File Format**: Comma-Separated Values (CSV), UTF-8 encoding   - Radius analysis by diagnosis type4. **Multiple features FTW** - Using all 30 features together > using just a few



**Dimensions**:7. Trains Random Forest Classifier on 80% training data5. **The model is pretty accurate** - Random Forest got good accuracy on test data

- **Rows**: 569 (patient samples)

- **Columns**: 31 total8. Tests model on 20% test data

  - Column 1: `id` - Patient identifier

  - Column 2: `diagnosis` - Target variable (M=Malignant, B=Benign)9. Outputs performance metrics:## 📝 License

  - Columns 3-32: 30 numeric features (tumor measurements)

   - Overall accuracy score

**Data Distribution**:

- Benign (B): 357 samples (62.7%)   - Confusion matrix (True Positives, True Negatives, False Positives, False Negatives)It's just a school project so use it however you want!

- Malignant (M): 212 samples (37.3%)

   - Detailed classification report (Precision, Recall, F1-Score per class)

**How to Use**:

- Automatically loaded by `app.py` and `breast_cancer1.py`## 🤝 Got Ideas?

- Must be in same directory as Python scripts

- Can be opened in spreadsheet applications (Excel, Google Sheets, LibreOffice)**Expected Output**: Terminal window displays:



### `config.toml`- Dataset informationFeel free to fork this, make changes, and submit pull requests if you think something could be better!



**Purpose**: Configuration file for Streamlit application settings.- Statistical summaries



**When to Modify**: Only if you need to change default Streamlit behavior (theme, logging, cache settings)- Charts and graphs (displayed in separate matplotlib windows)---



### `cina.png`- Model performance metrics



**Purpose**: Image icon/logo displayed in Streamlit web application.Made with ❤️ by a student trying to learn ML

**Runtime**: Approximately 5-10 seconds

**Usage**: Referenced in `app.py` for browser tab icon and visual branding

### Option 2: Launch Interactive Dashboard

## Analysis Results

To access the web-based interactive application:

### Dataset Statistics

- **Total Records Analyzed**: 569 patient samples```bash

- **Total Features Analyzed**: 30 quantitative measurementsstreamlit run app.py

- **Class Distribution**: 62.7% Benign, 37.3% Malignant```



### Key Findings**Launch Instructions**:

1. Run the command above in terminal

#### 1. Feature Importance2. Application automatically opens in default web browser

Features most important for diagnosis prediction:3. If not automatic, manually navigate to: `http://localhost:8501`

- **Radius (mean)**: Malignant tumors significantly larger (mean ~17.46mm vs ~12.15mm)4. Dashboard loads within 2-3 seconds

- **Compactness**: Malignant tumors more compact5. To stop the application: Press `Ctrl+C` in terminal

- **Concavity**: Malignant tumors have greater concave portions

- **Texture**: Significant texture differences observed**Dashboard Sections Available**:



#### 2. Statistical Significance1. **Overview Section**

- T-tests confirm significant differences in key features (p < 0.05)   - Dataset summary and basic statistics

- Variance differences (Levene's test) between tumor types   - Total number of records and features

- Large effect sizes for primary features   - Class distribution visualization

- Dataset sufficient for reliable conclusions

2. **EDA (Exploratory Data Analysis) Section**

#### 3. Model Performance   - Interactive scatter plots and histograms

- **Algorithm**: Random Forest Classifier   - Feature comparison tools

- **Accuracy**: High accuracy on test dataset   - Correlation heatmap visualization

- **Precision & Recall**: Well-balanced performance

- **Generalization**: Good performance on unseen data3. **Hypothesis Testing Section**

   - Statistical tests comparing malignant vs benign tumors

#### 4. Distribution Patterns   - T-test results with p-values

- **Malignant Tumors**: Larger radius, higher compactness, greater concavity   - Variance analysis (Levene's test)

- **Benign Tumors**: Smaller measurements, more regular characteristics   - Interpretation of statistical significance

- Clear feature separation enables effective classification

4. **ML Model Section**

## How to Access the Application   - Model performance metrics display

   - Confusion matrix visualization

### Web Dashboard (Recommended)   - Precision, recall, and F1-score metrics

   - Feature importance ranking chart

**Starting the Application**:

```bash5. **Predict Tumor Section**

cd breast_cancer_analysis   - Interactive form to input 30 tumor measurements

source venv/bin/activate  # macOS/Linux   - Real-time prediction interface

streamlit run app.py   - Returns diagnosis prediction (Malignant/Benign)

```   - Displays prediction confidence



**Expected Output**:**How to Use the Dashboard**:

```- Use sidebar radio buttons at top to navigate between sections

Local URL: http://localhost:8501- Interact with sliders, dropdowns, and input fields

Network URL: http://192.168.1.100:8501- Hover over charts for additional information

```- Charts update dynamically based on selections



**Accessing in Browser**:**Stopping the Application**:

1. Automatically opens at `http://localhost:8501`- Press `Ctrl+C` in the terminal window

2. Or manually type in address bar- Or close the browser tab (application stops after timeout)

3. Dashboard loads within 2-3 seconds

## Technical Stack

**Using Prediction Feature**:

1. Navigate to "Predict Tumor" section### Data Processing & Analysis

2. Input 30 tumor measurements- **Pandas (v2.0.3+)**: Data manipulation, cleaning, and transformation

3. Click "Predict" button- **NumPy (v1.24.3+)**: Numerical computations and array operations

4. View diagnosis result and confidence

### Visualization & Graphics

**Stopping the Application**:- **Matplotlib (v3.7.2+)**: Static plot creation and visualization

```bash- **Seaborn (v0.12.2+)**: Statistical data visualization and styling

Ctrl + C

```### Statistical Analysis

- **SciPy (v1.11.2+)**: 

### Command Line Analysis  - t-tests for comparing means

  - Levene's test for variance equality

**Quick Analysis**:  - Shapiro test for normality

```bash

python breast_cancer1.py### Machine Learning

```- **Scikit-learn (v1.3.1+)**:

  - Random Forest Classifier algorithm

**Save Output to File**:  - Train-test split for model validation

```bash  - Classification metrics and performance evaluation

python breast_cancer1.py > analysis_results.txt 2>&1

```### Web Application Framework

- **Streamlit (v1.28.1+)**: Interactive web application development

## Dependencies Summary  - UI components and widgets

  - Real-time app updates

| Package | Version | Purpose |  - Built-in deployment capabilities

|---------|---------|---------|

| streamlit | 1.28.1+ | Web application framework |### Image Processing

| pandas | 2.0.3+ | Data manipulation |- **Pillow/PIL (v10.0.0+)**: Image file handling and manipulation

| numpy | 1.24.3+ | Numerical computing |

| matplotlib | 3.7.2+ | Plotting and visualization |### Primary Programming Language

| seaborn | 0.12.2+ | Statistical visualization |- **Python (3.8+)**: All code implementation

| scipy | 1.11.2+ | Scientific computing |

| scikit-learn | 1.3.1+ | Machine learning algorithms |## File Descriptions

| pillow | 10.0.0+ | Image processing |

### `app.py` (887 lines total)

## Quick Reference Commands

**Purpose**: Implements the interactive Streamlit web dashboard for data exploration and prediction.

```bash

# Clone repository**File Structure and Sections**:

git clone https://github.com/ankhanhcute/breast_cancer_analysis.git

1. **Imports (Lines 1-12)**

# Create virtual environment   - Streamlit library for web interface

python -m venv venv   - Data processing libraries (pandas, numpy)

   - Visualization libraries (matplotlib, seaborn)

# Activate virtual environment (macOS/Linux)   - Statistical libraries (scipy.stats)

source venv/bin/activate   - Machine learning libraries (sklearn)

   - Image handling (PIL)

# Install dependencies

pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn pillow2. **Configuration (Lines 13-60)**

   - Sets page title: "Cancer Analysis"

# Run analysis script   - Sets page icon from `cina.png`

python breast_cancer1.py   - Configures wide layout mode

   - Defines custom CSS styling with professional color scheme:

# Start web dashboard     - Background: #f0f7ff (soft cloud white)

streamlit run app.py     - Primary: #7ec8e3 (baby blue)

     - Text: #4a6fa5 (dark blue)

# Stop application   - Applies custom fonts and styling

Ctrl + C

```3. **Data Loading (Lines 61-80)**

   - Reads `breast-cancer.csv` file

## License   - Preprocesses data (encoding, feature scaling)

   - Caches data for performance optimization

Educational and research use.

4. **Navigation Menu (Lines 81-100)**

## Project Information   - Sidebar radio buttons for section selection

   - Options: Overview, EDA, Hypothesis Test, ML Model, Predict Tumor

- **Created**: 2026

- **Last Updated**: April 12, 20265. **Section: Overview (Lines 101-150)**

- **Repository**: https://github.com/ankhanhcute/breast_cancer_analysis   - Displays dataset title and description

- **Author**: Ellie Phan   - Shows summary statistics

   - Presents sample data in table format

---   - Displays basic charts



For questions or support, visit the GitHub repository.6. **Section: EDA (Lines 151-400)**

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
