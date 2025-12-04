# Machine_learning-project
ML class assignment
# 🛒 Mall Customer Segmentation Analysis

This project utilizes unsupervised machine learning techniques to identify distinct customer segments within a shopping mall dataset. By applying **K-Means Clustering** to customer Income and Spending Scores, the analysis reveals 5 actionable profiles for targeted marketing strategies.

## 🐍 Analysis Workflow

The analysis is performed in a Python script (or Jupyter Notebook) following a linear data pipeline:

### 1. Data Preprocessing
The raw data requires cleaning before it can be interpreted by the model:
* **String Cleaning:** The `Income` column, originally formatted as strings (e.g., "15,000 USD"), is cleaned by removing currency symbols and commas, then converted to numeric integers.
* **Feature Selection:** The `CustomerID` column is dropped to prevent the model from identifying patterns in arbitrary ID numbers.
* **Normalization:** A `StandardScaler` (Z-score normalization) is applied to `Income` and `SpendingScore`. [Image of z-score formula] This ensures that income (ranging in the thousands) does not overpower the spending score (0-100) during distance calculations.

### 2. Determining Cluster Count (Elbow Method)
To scientifically determine the number of segments, the script uses the **Elbow Method**.
[Image of elbow method plot]
* The K-Means algorithm is run for $k=1$ through $k=10$.
* The inertia (sum of squared distances) is plotted for each $k$.
* **Result:** The "elbow" of the curve appears at **$k=5$**, indicating this is the optimal number of clusters for this dataset.

### 3. Clustering & Visualization
The model segments the 200 customers into 5 distinct groups. [Image of k-means clustering scatter plot] A scatter plot is generated using **Seaborn** to visualize these clusters based on their Annual Income (x-axis) and Spending Score (y-axis).

### 4. Demographic Profiling
To understand *who* makes up these clusters, the script analyzes demographic data:
* **Gender:** Converted to numeric values using one-hot encoding.
* **Age:** Aggregated by cluster to find the mean age.

---

## 📊 Key Findings

The analysis successfully identified **5 Unique Customer Profiles**. These insights can be directly applied to marketing campaigns:

### 1. The "Target" Group (High Income, High Spending)
* **Characteristics:** These customers earn high annual incomes and have the highest spending scores.
* **Demographics:** This is a relatively young group (Mean Age ~33) and is gender-balanced.
* **Strategy:** This is the most valuable segment. Target them with luxury brands, exclusive membership offers, and new product launches.

### 2. The "Carefree" Group (Low Income, High Spending)
* **Characteristics:** Despite having the lowest income bracket, these customers spend at a very high rate.
* **Demographics:** This is the **youngest** segment (Mean Age ~25).
* **Strategy:** Likely students or young adults supported by others. Target them with fast fashion, trends, food court promotions, and social-media-driven campaigns.

### 3. The "Careful" Group (High Income, Low Spending)
* **Characteristics:** High earners who spend very little in the mall.
* **Demographics:** An older demographic (Mean Age ~41) with a slightly higher male population.
* **Strategy:** Potential "savers" or those shopping for specific needs. Target them with high-quality, practical items ("buy it for life") or tech gadgets rather than impulse buys.

### 4. The "Thrifty" Group (Low Income, Low Spending)
* **Characteristics:** Low income and low spending scores.
* **Demographics:** This is the **oldest** segment (Mean Age ~45).
* **Strategy:** Highly price-sensitive. Target with discount coupons, clearance sales, and value-oriented bundles.

### 5. The "Standard" Group (Average Income, Average Spending)
* **Characteristics:** The largest cluster, falling right in the middle for both income and spending.
* **Demographics:** Middle-aged (Mean Age ~43).
* **Strategy:** The "bread and butter" shopper. Standard mall-wide promotions and traditional advertising work best here.

---

## 🚀 How to Run in Jupyter Notebooks

This analysis is best viewed and executed in a Jupyter Notebook environment (Anaconda, JupyterLab, or VS Code).

### Prerequisites
You will need the following Python libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
