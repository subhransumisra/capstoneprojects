# Bank Marketing Campaign - Classifier Performance Comparison

## Executive Summary
This project analyzes data from a Portuguese banking institution's telemarketing campaigns to predict whether a customer will subscribe to a term deposit (the target variable `y`). We evaluate four distinct machine learning classifiers to determine which model offers the best predictive power and business utility.

### Key Findings & Actionable Insights
* **The Core Driver (Duration):** The single most influential factor in securing a term deposit is the **call duration**. Longer conversations strongly correlate with successful subscriptions.
* **Target Audience Profile:** Customers with higher average yearly balances, those who have been successfully contacted in previous campaigns, and those without existing housing loans show a significantly higher propensity to subscribe.
* **Economic Timing:** Campaign success rates peak during specific months (e.g., March, September, October), indicating that timing outreach with seasonal financial planning yields better results.

### Strategic Recommendations
1. **Optimize Call Engagement:** Train agents on conversational techniques that naturally extend call time, focusing on quality discovery rather than rushing through script checklists.
2. **Smart Filtering:** Prioritize leads who have a history of positive responses (`poutcome == 'success'`) or lack major liabilities like housing loans.
3. **Resource Re-allocation:** Reduce aggressive cold-calling frequencies per campaign, as data shows a diminishing return (and potential customer fatigue) after 3-4 contact attempts.

---

## Evaluation Metric Rationale
We selected **Recall (True Positive Rate)** and **ROC-AUC** as our primary evaluation metrics. 
* **Why Recall?** In a bank telemarketing campaign, missing a potential subscribing customer (a False Negative) represents lost revenue that far outweighs the operational cost of making an extra phone call (a False Positive). We want to capture as many true subscribers as possible.

---

## Project Structure
compare_classifier_models/
│
├── data/
│   └── bank.csv
├── compare_classifier_models.ipynb
└── README.md