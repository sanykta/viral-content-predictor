# 🎥 Viral Content Performance Prediction and Optimization using NLP

## 📌 Overview

Creating viral content on YouTube often depends on a combination of content quality, audience behavior, timing, and presentation. While there is no guaranteed formula for virality, machine learning can identify patterns associated with high-performing videos.

This project develops an end-to-end machine learning pipeline that predicts the **viral potential of a YouTube video** using Natural Language Processing (NLP), feature engineering, and supervised learning models. The model analyzes video titles along with metadata to estimate the likelihood of strong performance and generate a **Viral Potential Score**.

---

# 🎯 Problem Statement

Content creators upload thousands of videos every day, but it is difficult to predict which ones are likely to perform well.

The objective of this project is to:

* Analyze historical YouTube video data.
* Identify features associated with high-performing content.
* Build a machine learning model capable of predicting viral potential.
* Demonstrate how NLP can be combined with structured metadata for predictive analytics.

---

# 📊 Dataset

The dataset was collected using the **YouTube Data API** and contains over **4,200 YouTube videos** from multiple channels.

### Data includes:

* Video Title
* Published Date
* View Count
* Like Count
* Comment Count
* Video Duration
* Subscriber Count
* Channel Average Views
* Upload Time
* Upload Day
* Video Metadata

---

# 🔍 Exploratory Data Analysis (EDA)

The project includes comprehensive exploratory data analysis to understand the characteristics of high-performing videos.

Key analyses include:

* Distribution of video views
* Upload frequency
* Upload day and hour trends
* Video duration analysis
* Correlation analysis
* Feature distributions
* Outlier detection
* Viral vs non-viral comparisons

---

# ⚙️ Feature Engineering

Several new features were created to improve predictive performance.

### Text Features

* Title Length
* Word Count
* Uppercase Ratio
* Has Number
* Has Question
* Has Exclamation
* Has Hashtag
* Hashtag Count

### Temporal Features

* Upload Month
* Upload Hour
* Weekend Upload Indicator

### Channel Features

* Subscriber Count
* Channel Average Views
* Views per Subscriber

### NLP Features

* Text Cleaning
* TF-IDF Vectorization
* Sentiment Analysis (VADER)
* Emotion Detection (GoEmotions Transformer)
* Emotional Trigger Score

---

# 🤖 Machine Learning Models

The following classification models were trained and evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Model evaluation was performed using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Among the evaluated models, **XGBoost achieved the best overall performance**, making it the final model used in the prediction pipeline.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* NLTK
* VADER Sentiment Analysis
* Hugging Face Transformers (GoEmotions)
* Joblib
* Google Colab
* YouTube Data API

---

# 📂 Project Structure

```
├── Data Collection.ipynb
├── EDA & Feature Engineering.ipynb
├── Model Training.ipynb
├── Prediction Pipeline.ipynb
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
```

---

# 🚀 Project Workflow

1. Collect YouTube video data using the YouTube Data API.
2. Clean and preprocess the dataset.
3. Perform exploratory data analysis.
4. Engineer numerical and NLP-based features.
5. Convert video titles into numerical representations using TF-IDF.
6. Train multiple machine learning models.
7. Evaluate model performance.
8. Save the best-performing model.
9. Generate predictions and a Viral Potential Score for new videos.

---

# 📈 Results

The project demonstrates that combining structured metadata with NLP features significantly improves predictive performance.

The final model:

* Utilizes engineered numerical and text-based features.
* Predicts viral potential for unseen video titles and metadata.
* Generates an interpretable Viral Potential Score.
* Can serve as a decision-support tool for content creators during content planning.

---

# 💡 Future Improvements

Potential enhancements include:

* Deep learning models using BERT or RoBERTa embeddings.
* Thumbnail image analysis using Computer Vision.
* Time-series analysis of channel growth.
* Incorporating YouTube Shorts data.
* Deployment as a web application.
* Real-time predictions through API integration.

---

# 📚 Learning Outcomes

This project strengthened practical skills in:

* Data Collection
* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Natural Language Processing
* Machine Learning
* Model Evaluation
* Predictive Analytics
* Model Deployment

---

# 📬 Contact

If you have any questions, suggestions, or feedback, feel free to connect with me on LinkedIn or explore the project repository.

Thank you for visiting this project!
