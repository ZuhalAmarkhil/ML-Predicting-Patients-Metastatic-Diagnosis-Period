## Predicting Metastatic Diagnosis Period with Machine Learning ##

### Description ###
This project analyzes a real-world evidence dataset from Health Verity (HV), one of the largest healthcare data ecosystems in the US, to predict the patient's Metastatic Diagnosis Period (metastatic_diagnosis_period). The HV dataset contains health-related information of patients diagnosed with metastatic triple-negative breast cancer in the US. It includes roughly 19k records, with each row corresponding to a single patient and her metastatic diagnosis period.

The project encompasses data preprocessing, data exploration, feature engineering and selection, modeling for predicting the metastatic diagnosis period, model evaluation, and outputting the results in a CSV file.

### Libraries Utilized ###
- **pandas**: Used for data manipulation and analysis.
- **numpy**: Utilized for numerical operations and array handling.
- **seaborn**: For statistical data visualization.
- **matplotlib.pyplot**: For creating visualizations.
- **sklearn.impute.SimpleImputer**: To handle missing values.
- **sklearn.preprocessing.LabelEncoder**: For encoding categorical labels.
- **sklearn.impute.KNNImputer**: For imputing missing values using k-nearest neighbors.
- **matplotlib.colors.LinearSegmentedColormap**: For defining custom colormaps.
- **sklearn.model_selection.train_test_split**: To split the dataset into training and testing sets.
- **sklearn.metrics**: For evaluating model performance using mean absolute error, mean squared error.
- **catboost.CatBoostRegressor**: For modeling using gradient boosting with categorical features.
- **lightgbm.LGBMRegressor**: For efficient gradient boosting modeling with large datasets.
- **xgboost**: For optimized distributed gradient boosting modeling.

### Project Structure ###
- **Plots and Charts/**: Directory containing visualizations and plots generated during the analysis.
- **Metadata.md:** Contains metadata information about the dataset.
- **README.md:** Project documentation and overview.
- **analysis_prediction.ipynb:** Jupyter notebook with data analysis, modeling, and prediction.
- **train.csv, test.csv:** These files contain datasets for training and testing machine learning models. The training dataset (train.csv) is used to train the model, while the testing dataset (test.csv) is used to evaluate its performance.
