## LoanWise – Credit Eligibility Prediction 📊  

### 🔍 Overview  
LoanWise is a machine learning project designed to predict **credit eligibility** using **advanced predictive models**. This project leverages **LightGBM, XGBoost, and Logistic Regression**, with LightGBM achieving the best results. The goal is to enhance the credit approval process by improving accuracy, recall, and overall decision-making efficiency.  

### 📂 Dataset  
- **Original Source:** [Kaggle - Credit Card Capability Data](https://www.kaggle.com/datasets/zeesolver/credit-cared)
- Several changes were made to adapt it to the local currency, the dataset used in this exercise is "credit_dataset_cop_null.csv" located in this repository.
- **Size:** 9,609 instances with 18 features  
- **Key Features:**  
  - **Demographics:** Gender, number of children, family size  
  - **Financial Data:** Employment status, income, property ownership  
  - **Behavioral Data:** Phone/email availability, account length  
  - **Target Variable:** Credit eligibility (1 = Eligible, 0 = Not Eligible)  

### 🚀 Models & Experiments  
The project went through **nine experiments**, optimizing models through:  
✅ **Data balancing:** Applied SMOTE to improve minority class detection.  
✅ **Hyperparameter tuning:** LightGBM and XGBoost optimization to enhance precision and recall.  
✅ **Feature selection:** Reduced complexity while maintaining accuracy.  

#### 🏆 **Best Model: Optimized LightGBM**  
- **Accuracy:** 88.37%  
- **Precision:** 90.65%  
- **Recall:** 85.22%  
- **F1-Score:** 87.85%  
- **Specificity:** 91.44%  

LightGBM outperformed all other models, making it the best candidate for **real-world credit risk analysis**.  

### 📊 Exploratory Data Analysis (EDA)  
- **Data Cleaning & Transformation**: Handling missing values, scaling numerical features, encoding categorical variables.  
- **Outlier Detection**: Identified high variance in income and employment length.  
- **Feature Engineering**: Selected the most relevant variables based on correlation and model performance.  

### 🛠️ Installation & Requirements  
To run this project, install the required libraries:  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm
```

### 📌 How to Use  
1. Clone the repository:  
   ```bash
   git clone https://github.com/ragarciaga/loanwise.git
   cd loanwise
   ```
2. Run the **Jupyter Notebook** to explore the data and train models.  
3. Modify parameters to experiment with different configurations.  

### 📈 Results & Visualizations  
The project includes **heatmaps, performance curves, and feature importance plots** to illustrate model effectiveness.  

### 🤝 Contributions  
This is a personal academic project, but **collaborators are welcome**! Feel free to submit pull requests or issues.  

### 📜 License  
This project is open-source under the **MIT License**.  

### 📬 Contact  
- **GitHub:** [https://github.com/ragarciaga](#)  
- **LinkedIn:** [https://www.linkedin.com/in/ragarciaga/](#)  
