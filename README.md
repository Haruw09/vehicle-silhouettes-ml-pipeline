# vehicle-silhouettes-ml-pipeline

Machine learning coursework project based on the Girafe AI / Neitchev ML course assignment.

This project implements a complete end-to-end machine learning pipeline for multiclass vehicle classification using the Statlog Vehicle Silhouettes dataset.

## Project Includes

- Data preprocessing and feature scaling
- Correlation analysis and feature selection
- Logistic Regression with hyperparameter tuning
- PCA dimensionality reduction and explained variance analysis
- Decision Tree optimization with cross-validation
- Bagging ensembles
- Random Forest analysis
- Multiclass ROC curve evaluation
- Learning curves and model comparison

## Models Evaluated

- Logistic Regression
- Decision Tree
- Bagging Classifier
- Random Forest

## Metrics

The following evaluation metrics were used:

- Accuracy
- Macro F1-score
- ROC AUC

## Dataset

Statlog (Vehicle Silhouettes) Dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes

## Course Assignment

Girafe AI ML Course — Lab 1 ML Pipeline:

https://github.com/girafe-ai/ml-course/blob/22f_basic/homeworks/lab01_ml_pipeline/Lab1_part2_ml_pipeline.ipynb

## Repository Structure

```text
vehicle-silhouettes-ml-pipeline/
│
├── ml_pipeline.ipynb
├── README.md
├── requirements.txt
```

## Requirements

Main libraries used in the project:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Conclusion

The project demonstrates a complete machine learning workflow including preprocessing, dimensionality reduction, hyperparameter optimization, ensemble learning, and model evaluation.

Among the evaluated models, tuned Logistic Regression demonstrated strong and stable generalization performance, while ensemble tree-based methods showed signs of overfitting on the dataset.