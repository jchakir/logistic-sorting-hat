# 🧙‍♂️ Harry Potter and a Data Scientist: Rebuilding the Sorting Hat

Welcome to the Hogwarts Sorting Hat Reconstruction Project! As part of the 1337 (42 Network) curriculum, this project dives into data science and machine learning to classify Hogwarts students into their respective houses. In this journey, we will wield our muggle tools to restore order to Hogwarts.

---

## ✨ Overview

The Sorting Hat at Hogwarts has malfunctioned, leaving Professor McGonagall in distress. Enter you, a muggle data scientist! Using logistic regression and visualization techniques, your task is to rebuild the Sorting Hat—classifying students into Gryffindor, Ravenclaw, Hufflepuff, or Slytherin.

This project is part of the curriculum at **1337 (42 Network)**, emphasizing foundational machine learning and data analysis concepts. Join the magical quest to save Hogwarts!

---

## 🎯 Objectives

This project hones your skills in:

- **Data Analysis:** Explore datasets to extract meaningful insights.
- **Visualization Techniques:** Use histograms, scatter plots, and pair plots to clean and interpret features.
- **Logistic Regression:** Build a multi-class classifier with a one-vs-all approach.
- **Mathematics of ML:** Understand sigmoid activation and cross-entropy loss functions.

---

## 🛠 Steps

### 1. 🧐 Data Analysis

Using the `describe.py` script, statistical properties of the dataset are computed without relying on built-in functions. Key statistics include mean, variance, standard deviation, and quartiles, offering a comprehensive view of the dataset.

### 2. 📊 Visualization Techniques

We employ powerful visualizations to uncover patterns:

- **Histograms:** Display score distributions across houses.
- **Scatter Plots:** Identify feature relationships.
- **Pair Plots:** Analyze inter-feature correlations to select features for logistic regression.

These techniques are implemented in `visualize.py`, aiding in feature engineering and cleaning.



## 3. 🧙‍♂️ Logistic Regression Magic

The `logreg_train.py` and `logreg_predict.py` scripts train and evaluate a logistic regression model:

- **One-vs-All Approach:** Separate models for each house predict probabilities.
- **Optimization Techniques:** Gradient descent and Adam optimizer ensure efficient learning.

#### What is Logistic Regression?

Logistic regression is a supervised learning algorithm used for binary classification problems. It predicts the probability of an instance belonging to a particular class by applying a logistic (sigmoid) function to a linear combination of input features and weights.

#### Sigmoid Activation Function

The sigmoid function maps any real-valued number into the range (0, 1). Mathematically, it is expressed as:

Here,  is the weighted sum of inputs. This function is essential in logistic regression as it outputs probabilities, which are critical for classification tasks.

#### Cross-Entropy Loss Function

The cross-entropy loss function quantifies the difference between predicted probabilities and actual class labels. It is defined as:

Where:

- &#x20;is the predicted probability.
- &#x20;is the actual class label (0 or 1).
- &#x20;is the number of training examples.

This loss function ensures that the model penalizes incorrect predictions, encouraging better alignment with true labels.

### 4. 📐 Activation and Loss Functions

- **Sigmoid Activation Function:** Outputs probabilities for binary classification.
- **Cross-Entropy Loss:** Measures model performance, with numerical stability to handle extreme cases.

### 5. 🚀 Bonus Section

For those seeking extra challenges:

- Implement advanced optimization techniques.
- Expand statistical calculations in `describe.py`.

---

## 📜 Project Storyline

Imagine yourself as a data science wizard. With Professor McGonagall’s support, you use your computational spells to analyze and visualize student data. The journey culminates in crafting a magical Sorting Hat model that rivals the original. Accuracy above 98% earns you her highest praise!

---

## 🛠 Setup Instructions

### 📦 Required Packages

To run this project, the following Python packages are required:

- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For creating static visualizations.
- `seaborn`: For advanced statistical visualizations.
- `prettytable`: For creating well-formatted tables.

Install the required packages using:

```bash
pip install numpy pandas matplotlib seaborn prettytable
```

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/hogwarts-sorting-hat.git
   ```

2. Run data analysis:

   ```bash
   python describe.py <dataset.csv>
   ```

3. Visualize data:

   ```bash
   python visualize.py <dataset.csv> <plot_type>
   ```

4. Train the model:

   ```bash
   python logreg_train.py <dataset.csv>
   ```

5. Predict outcomes:

   ```bash
   python logreg_predict.py <dataset_test.csv> <weights_file>
   ```

---
