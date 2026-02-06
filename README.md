# Customer Segmentation & Retention Analysis 📊

## Executive Summary
In the subscription economy (Spotify, Netflix, SaaS), growth is driven by retention. This project provides an end-to-end data pipeline to identify which customers are leaving, why they are leaving, and which ones are worth saving. By combining **K-Means Clustering** and **Random Forest Classifiers**, this tool transforms raw usage data into a prioritized business action plan.

## Key Features
* **Behavioral Segmentation:** Grouping users by engagement patterns rather than just demographics using K-Means.
* **Churn Prediction:** A Random Forest model that predicts the probability of a user canceling their subscription.
* **CLV Calculation:** Estimating the financial value of each user to prioritize high-value retention.
* **Automated Intervention Logic:** Categorizes users into:
    * **Retention Targets:** High-risk, high-value users.
    * **Brand Ambassadors:** Low-risk power users for early feature access.
    * **Sunk Costs:** High-risk, low-value users to be excluded from expensive campaigns.

## Technical Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Local Deployment (Optimized for Jupyter/VS Code)

## The Business Logic (The "Why")
Standard models often fail by treating all churners the same. This project implements a **Decision Matrix**:

| Segment | Risk | Value | Strategy |
| :--- | :--- | :--- | :--- |
| **Champions** | Low | High | Early Access / Referral Rewards |
| **At-Risk** | High | High | **Immediate Discount/Retention Offer** |
| **Hibernating** | High | Low | Automated Re-engagement (Low Cost) |



## 🔗 Live Interactive Notebook
You can run this project directly in your browser without any setup:
[Execute on Google Colab](https://colab.research.google.com/drive/1q5ZWcMMuGW6TuJEQxOXezAjEcqnrysrr?usp=sharing)

## Implementation
The analysis follows a strict data science workflow:
1.  **EDA:** Identifying the "Aha! Moment" (usage threshold where churn drops).
2.  **Preprocessing:** Feature scaling and handling class imbalance.
3.  **Modeling:** Training a Propensity Model (Random Forest) with hyperparameter tuning.
4.  **Actionable Output:** Exporting specific UserID lists for Marketing and Product teams.

## How to Run Locally
1. Clone the repo.
2. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn`
3. Run the main script: `python customer_segmentation_analysis.py`
## How to Run
1. Clone the repo.
2. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn`.
3. Run the main script: `python customer_segmentation_analysis.py`.
