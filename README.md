# ICD10_medical_encoding_prediction

In order to tackle this multi-label classification problem, two main approaches were explored. Each approach utilized different model types and techniques to optimize performance, with a focus on maximizing the micro F2-score.

**Approach 1: Traditional Machine Learning Models**

The first approach involved breaking down the multi-label classification task into multiple binary classification problems. By treating each ICD10 code as a separate binary label, the following models were tested:

Logistic Regression: A baseline model was built using logistic regression, given its ability to handle binary classification tasks efficiently and its interpretability.

Decision Trees: Decision Trees were also considered; however, due to computational constraints and slower training times, this model was later abandoned in favor of faster alternatives like logistic regression. To further improve performance, several data reduction techniques were applied:

Principal Component Analysis (PCA): PCA was used to reduce feature dimensionality while retaining 95% of the variance. Low-Frequency Label Pruning: Labels with occurrences below a certain threshold were removed from the dataset, helping to reduce noise and improve focus on labels with sufficient data representation. Despite these optimizations, the micro F2-score on the leaderboard plateaued at 0.3, suggesting the need for a more powerful model.

**Approach 2: Deep Learning Models**

Given the limited success of traditional models, a second approach using deep learning was employed.

Dataset: https://www.kaggle.com/competitions/da5401-2024-ml-challenge/data 

Refer report.pdf for a detailed outline of the project.

output.py - to make predictions on the test data
