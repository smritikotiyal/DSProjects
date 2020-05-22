import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import train_test_split
# Standard ML Models for comparison
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
# Metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Model Tuning
from sklearn.model_selection import RandomizedSearchCV

# Supervised Classification model to calssify whether a person has Heart Disease or Not
class CW_AML:
    df_heart_disease = pd.read_csv('Heart_Disease.csv')
    df_heart_disease_copy = pd.DataFrame(df_heart_disease)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # List of Features in the Dataset
    def featuresList(self):
        feat_list = []
        for i in range(0, len(self.df_heart_disease.axes[1])):
            feat_list.append(self.df_heart_disease.axes[1][i])
        return feat_list

    def dataPreprocessing(self):
        print('Before Feature Renaming.')
        print(self.df_heart_disease.head())

        # Renaming the features for better understanding
        self.df_heart_disease.rename(columns={"age": "Age",
                                              "sex": "Sex",
                                              "cp": "Chest_Pain",
                                              "trestbps": "Resting_BP",
                                              "chol": "Cholestrol",
                                              "fbs": "Fasting_BS",
                                              "restecg": "Rest_ECG",
                                              "thalach": "Max_HR",
                                              "exang": "Ex_Angina",
                                              "oldpeak": "Ex_ST_Depression",
                                              "slope": "Slope",
                                              "ca": "Colored_Vessels",
                                              "thal": "Defect",
                                              "target": "Target"},
                                     errors="raise", inplace=True)
        print('After Feature Renaming.')
        print(self.df_heart_disease.head())

        # Dataset Description
        # pd.set_option('display.max_columns', None)
        include = ['object', 'float64', 'int64']
        print(self.df_heart_disease.describe(include=include))

        # Detect any Outliers and Treat them
        self.outlierTreatment()
        print('After Outlier Detection and Treatment')
        print(self.df_heart_disease.describe(include=include))

    def outlierTreatment(self):
        # List of all features of the dataset
        featList = self.featuresList()
        cols = []

        for i in featList:
            #points greater than Q3 + 1.5*IQR
            # Calculates for the Q1, Q3 and IQR to find high and low outliers' threshold
            q1 = np.percentile(self.df_heart_disease[i], [25])[0]
            q3 = np.percentile(self.df_heart_disease[i], [75])[0]
            iqr = q3 - q1
            high_outliers = q3 + (1.5 * iqr)
            low_outliers = q1 - (1.5 * iqr)

            # If the data points are higher of lower than the thresholds
            # Capping high outliers to 99 percentile value and flooring low outliers to 1 percentile values.
            if(self.df_heart_disease[i].max() > high_outliers):
                # self.plotOutliers(i)
                new_value = np.percentile(self.df_heart_disease[i], [99])[0]
                self.df_heart_disease.loc[self.df_heart_disease[i] > high_outliers, i] = new_value
                cols.append(i)
            if(self.df_heart_disease[i].min() < low_outliers):
                # self.plotOutliers(i)
                new_value = np.percentile(self.df_heart_disease[i], [1])[0]
                self.df_heart_disease.loc[self.df_heart_disease[i] < low_outliers, i] = new_value
                cols.append(i)

        print('List of Columns with Outliers :\n ', cols)

    def plotOutliers(self, feature):
        # Box Plot Visualisation of Outliers
        sns.boxplot(y=feature, data=self.df_heart_disease)
        plt.title('Box Plot for feature ' + str(feature))
        plt.show()

    def exploratoryDataAnalysis(self):
        # Diplays the statistical details of all the columns of the dataframe
        # Structure, data types, Unique counts in each feature and number of Null Values
        print('Number of Columns and Rows in the dataset : ', self.df_heart_disease.shape)
        # pd.set_option('display.max_columns', None)
        print('Datatypes of dataset columns : \n', self.df_heart_disease.dtypes)

        # List of all features of the dataset
        featList = self.featuresList()
        for i in featList:
            print('Count of Unique Values in Column \'' + i + '\' :\n', self.df_heart_disease.loc[:, i].value_counts())
            print('Number of Unique values : ', len(self.df_heart_disease.loc[:, i].value_counts()))
            print('Count of Null Values in Column \'' + str(i) + '\' : ', str(self.df_heart_disease[i].isnull().sum()))
            print('\n')

            if(len(self.df_heart_disease.loc[:, i].value_counts()) > 4):
                self.df_heart_disease.loc[:, i].value_counts().plot(kind='bar')
                plt.xlabel(i)
                plt.ylabel('Total Count. Null count = ' + str(self.df_heart_disease[i].isnull().sum()))
                plt.title('Value Count of ' + str(i) + ' ')
                plt.show()

        # Analysis of features' distributions, relationships and correlations
        self.distributionPlots()
        self.featuresComparisonPlots()
        self.featureCorrelation()

    def distributionPlots(self):

        # Distribution of features with Target
        self.df_heart_disease.hist(bins=14)
        plt.subplots_adjust(wspace=0.5)
        plt.suptitle('Distribution of Parameters of the Dataframe')
        plt.show()

        # Density Plots for Individual Features according to the class label and features' unique values
        featList = self.featuresList()
        for i in featList:
            if(i == 'Sex'):
                print('inside sex')
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 0, 'Target'],
                            label='Female', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 1, 'Target'],
                            label='Male', shade=True)
                k = 'Sex'
            elif (i == 'Chest_Pain'):
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 0, 'Target'],
                            label='Typical Angina', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 1, 'Target'],
                            label='Atypical Angina', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 2, 'Target'],
                            label='Non-Anginal Pain', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 3, 'Target'],
                            label='Asymptomatic', shade=True)
                k = 'Chest Pain'

            elif (i == 'Fasting_BS'):
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 0, 'Target'],
                            label='False', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 1, 'Target'],
                            label='True', shade=True)
                k = 'Fasting Blood Sugar'

            elif (i == 'Rest_ECG'):
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 0, 'Target'],
                            label='Normal', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 1, 'Target'],
                            label='ST-T Wave Abnormality', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 2, 'Target'],
                            label='Left Ventricular Hypertrophy', shade=True)
                k = 'Resting ECG'

            elif (i == 'Ex_Angina'):
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 0, 'Target'],
                            label='Yes', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 1, 'Target'],
                            label='No', shade=True)
                k = 'Exercise Induced Angina'


            elif (i == 'Slope'):
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 0, 'Target'],
                            label='Upsloping', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 1, 'Target'],
                            label='Flat', shade=True)
                sns.kdeplot(self.df_heart_disease.loc[self.df_heart_disease[i] == 2, 'Target'],
                            label='Downsloping', shade=True)
                k = 'Slope of Peak Exercise'

            else:
                continue

            plt.xlabel('Possibility of Heart Disease')
            plt.ylabel(k)
            plt.title('Density Plot of Possibility of Heart Disease by ' + k)
            plt.show()

    def featuresComparisonPlots(self):
        feat_list = self.featuresList()
        for i in feat_list:
            for j in feat_list:
                df_unique_i = self.df_heart_disease.loc[:, i].value_counts()
                df_unique_j = self.df_heart_disease.loc[:, j].value_counts()
                if (j != i):

                    # Scatter plots of features with distributed values
                    if(len(df_unique_i) > 4 and len(df_unique_j) > 4):
                        yes = plt.scatter(self.df_heart_disease[i][self.df_heart_disease['Target'] == 1],
                                          self.df_heart_disease[j][self.df_heart_disease['Target'] == 1], color='Red')
                        no = plt.scatter(self.df_heart_disease[i][self.df_heart_disease['Target'] == 0],
                                         self.df_heart_disease[j][self.df_heart_disease['Target'] == 0], color='Green')
                        plt.xlabel(i)
                        plt.ylabel(j)
                        plt.title('Relationship between ' + i + ' and ' + j)
                        plt.legend([yes, no], ['Disease', 'No Disease'])
                        plt.show()
            feat_list.remove(i)

    def featureCorrelation(self):
        # Correlation Matrix of all the features
        corr_matrix = self.df_heart_disease.corr()
        fig, ax = plt.subplots(figsize=(15, 10))
        ax = sns.heatmap(corr_matrix,
                         annot=True,
                         linewidths=0.5,
                         fmt=".2f",
                         cmap="PuBuGn")
        plt.title('Correlation Matrix', fontsize=16)
        plt.show()

        # Sorted List of Features' Correlation with Target
        feat_list = self.featuresList()
        corr_dict = {}
        # Creating a dictionary with feature correlation with target column
        for i in feat_list:
            corr_dict[i] = self.df_heart_disease[i].corr(self.df_heart_disease['Target'])

        # Sorting and Printing Dictionary of features correlation
        corr_dict = {k: v for k, v in sorted(corr_dict.items(), key=lambda item: item[1])}
        for k,v in corr_dict.items():
            print(k, ':', v)

    # -----------------------------------------------------------------------------------------------------------------
    # Statistial Data Modelling
    def dataModelling(self):
        self.df_heart_disease_copy = self.df_heart_disease.drop(columns='Target')
        target = self.df_heart_disease['Target']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df_heart_disease_copy,
                                                                                target, test_size=0.30)
        print('Test Label Data Unique Value count :\n ', self.y_test.value_counts())
        print('Train Label Data Unique Value count :\n ', self.y_train.value_counts())
        print('Training Data without Label : \n', self.x_train)
        print('Training Label Data : \n', self.y_train)
        print('Testing Data without Label : \n', self.x_test)
        print('Testing Label Data : \n', self.y_test)

    def modelFitting(self):
        # Baseline Models
        model_list = ['Naive Bayes', 'KNN', 'Random Forest', 'Logistic Regression']
        model_names = {'Naive Bayes': GaussianNB(), 'KNN': KNeighborsClassifier(),
                       'Random Forest': RandomForestClassifier(),
                       'Logistic Regression': LogisticRegression()}
        # Evaluation Metrics
        classification_metrics = pd.DataFrame(columns=['estimate_score', 'Cross-Validation'], index=model_list)
        train_score = pd.DataFrame(columns=['estimate_score_train', 'Cross-Validation_train'], index=model_list)
        auc_metrics = pd.DataFrame(columns=['AUC_score'], index=model_list)
        parameters_metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1_Score'], index=model_list)

        for model_name, model_function in model_names.items():
            # Model Fitting and Prediction
            model_function.fit(self.x_train, self.y_train)
            y_prediction = model_function.predict(self.x_test)

            # Evaluation Metrics Caluclations
            classification_metrics = self.estimateModelPerformance(classification_metrics, model_name, model_function)
            train_score = self.estimateModelPerformanceTrainingSet(train_score, model_name, model_function)
            auc_metrics = self.evaluateModelsROC(auc_metrics, model_name, model_function)
            parameters_metrics = self.estimateClassificationParameters(parameters_metrics,
                                                                       model_name, model_function)
            self.confusionMatrix(model_name, model_function, y_prediction)

        print(classification_metrics)
        print(train_score)

        # Plotting the Evaluation Metrics
        self.plotClassificationMetrics(classification_metrics)
        self.plotAUCMetrics(auc_metrics)
        self.plotParametersMetrics(parameters_metrics)

    def estimateModelPerformance(self, classification_metrics, model_name, model_function):
        # 5-fold Cross Validation Score on Test set
        cross_val_score_None = cross_val_score(model_function, self.x_test, self.y_test, cv=5)
        classification_metrics.loc[model_name, :] = [model_function.score(self.x_test, self.y_test),
                                                     np.mean(cross_val_score_None)]
        return classification_metrics

    def estimateModelPerformanceTrainingSet(self, train_score, model_name, model_function):
        # 5-fold Cross Validation Score on Training set
        cross_val_score_train = cross_val_score(model_function, self.x_train, self.y_train, cv=5)
        train_score.loc[model_name, :] = [model_function.score(self.x_train, self.y_train),
                                                     np.mean(cross_val_score_train)]
        return train_score

    def plotClassificationMetrics(self, classification_metrics):
        plt.figure(figsize=(12, 8))
        matplotlib.rcParams['font.size'] = 11
        # Score() on y_test Results
        ax = plt.subplot(1, 2, 1)
        classification_metrics.sort_values('estimate_score', ascending=False).plot.bar(y='estimate_score', color='g', ax=ax)
        plt.title('estimate_score')
        plt.ylabel('estimate_score')
        # Cross-Validated Results
        ax = plt.subplot(1, 2, 2)
        classification_metrics.sort_values('Cross-Validation', ascending=False).plot.bar(y='Cross-Validation', color='r', ax=ax)
        plt.title('Cross-Validation')
        plt.ylabel('Cross-Validation')
        plt.tight_layout()
        plt.show()

    def evaluateModelsROC(self, auc_metrics, model_name, model_function):
        # Make Predictions with probabilities for ROC Curve. Model is trying to predict 0 or 1
        tester_probability = model_function.predict_proba(self.x_test)
        tester_probability_positive = tester_probability[:, 1]

        false_positive, true_positive, threshold = roc_curve(self.y_test, tester_probability_positive)
        auc_score = roc_auc_score(self.y_test, tester_probability_positive)
        auc_metrics.loc[model_name, :] = [auc_score]

        plt.plot(false_positive, true_positive, color='orange', label='ROC')
        # plot line with no predictive power baseline
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for ' + model_name + '. \n Area Under ROC Curve = ' + str(auc_score))
        plt.legend()
        plt.show()

        return auc_metrics

    def plotAUCMetrics(self, auc_metrics):
        print(auc_metrics)
        auc_metrics.sort_values('AUC_score', ascending=False).plot.bar(y='AUC_score', color='b')
        plt.xticks(rotation=10)
        plt.ylabel('AUC Score')
        plt.title('AUC Score for each classifier.')
        plt.show()

    def estimateClassificationParameters(self, parameters_metrics, model_name, model_function):
        accuracy = np.mean(cross_val_score(model_function, self.x_test, self.y_test, cv=5, scoring='accuracy'))
        precision = np.mean(cross_val_score(model_function, self.x_test, self.y_test, cv=5, scoring='precision'))
        recall = np.mean(cross_val_score(model_function, self.x_test, self.y_test, cv=5, scoring='recall'))
        f1Score = np.mean(cross_val_score(model_function, self.x_test, self.y_test, cv=5, scoring='f1'))

        parameters_metrics.loc[model_name, :] = [accuracy, precision, recall, f1Score]
        return parameters_metrics

    def plotParametersMetrics(self, parameters_metrics):
        print(parameters_metrics)
        # Plot standard metrics
        plt.figure(figsize=(12,8))

        # Plot Accuracy
        ax = plt.subplot(1, 4, 1)
        parameters_metrics.sort_values('Accuracy', ascending=False).plot.bar(y='Accuracy', color='g', ax=ax)
        plt.title('Accuracy')
        plt.ylabel('Accuracy')
        # Plot Precision
        ax = plt.subplot(1, 4, 2)
        parameters_metrics.sort_values('Precision', ascending=False).plot.bar(y='Precision', color='r', ax=ax)
        plt.title('Precision')
        plt.ylabel('Precision')
        # Plot Recall
        plt.tight_layout()
        ax = plt.subplot(1, 4, 3)
        parameters_metrics.sort_values('Recall', ascending=False).plot.bar(y='Recall', color='b', ax=ax)
        plt.title('Recall')
        plt.ylabel('Recall')
        # Plot F1 Score
        plt.tight_layout()
        ax = plt.subplot(1, 4, 4)
        parameters_metrics.sort_values('F1_Score', ascending=False).plot.bar(y='F1_Score', color='black', ax=ax)
        plt.title('F1 Score')
        plt.ylabel('F1 Score')
        plt.tight_layout()
        plt.show()

    def confusionMatrix(self, model_name, model_function, y_prediction):
        conf_mat = confusion_matrix(self.y_test, y_prediction)
        print(classification_report(self.y_test, y_prediction))
        print(conf_mat)
        # Using Seaborn HeatMaps
        # fig, ax = plt.subplots(figsize=(8, 6))
        sns.set(font_scale=1.5)
        sns.heatmap(conf_mat, annot=True, fmt='')
        # # Annotate the boxes with conf_mat info
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Confusion Matrix obtained by applying '+ model_name + '  Model.')
        plt.show()

    def improveModelKNN(self):
        model_function = KNeighborsClassifier()
        grid = {'n_neighbors': np.arange(6, 30, 1),
                'p': [1, 2]}
        n_neighbors = 0
        p = 0
        random_search = RandomizedSearchCV(model_function, param_distributions=grid,
                                           cv=5, n_iter=20, verbose=True)
        random_search.fit(self.x_train, self.y_train)
        print(random_search.best_params_)
        print(random_search.score(self.x_test, self.y_test))
        # Extracting best parameter values from random search CV modelling on the model
        for k, v in random_search.best_params_.items():
            if (k == 'n_neighbors'):
                n_neighbors = v
            elif (k == 'p'):
                p = v

        return n_neighbors, p

    def improveModelLogisticRegression(self):
        model_function = LogisticRegression()
        grid = {'C': np.logspace(-4, 4, 20),
                'solver': ['liblinear']}
        C = 0
        solver = ''
        random_search = RandomizedSearchCV(model_function, param_distributions=grid,
                                           cv=5, n_iter=20, verbose=True)
        random_search.fit(self.x_train, self.y_train)

        # Extracting best parameter values from random search CV modelling on the model
        for k,v in random_search.best_params_.items():
            if(k == 'C'):
                C = v
            elif(k=='solver'):
                solver = v

        return C, solver

    def improveModelRandomForest(self):
        model_function = RandomForestClassifier()
        grid = {'n_estimators': np.arange(100, 1000, 50),
                'max_features': ['auto', 'sqrt'],
                'max_depth': [3, 5, 10],
                'min_samples_split': np.arange(2, 20, 2),
                'min_samples_leaf': np.arange(1, 20, 2)}

        n_estimators = 0
        max_features = ''
        max_depth = 0
        min_samples_split = 0
        min_samples_leaf = 0

        random_search = RandomizedSearchCV(model_function, param_distributions=grid,
                                           cv=5, n_iter=20, verbose=True)
        random_search.fit(self.x_train, self.y_train)

        # Extracting best parameter values from random search CV modelling on the model
        for k,v in random_search.best_params_.items():
            if(k == 'n_estimators'):
                n_estimators = v
            elif (k == 'max_features'):
                max_features = v
            elif(k=='max_depth'):
                max_depth = v
            elif(k == 'min_samples_split'):
                min_samples_split = v
            elif(k=='min_samples_leaf'):
                min_samples_leaf = v
        print('n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf',
              n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf)
        return n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf

    def modelFittingAfterTuning(self):
        # Baseline Models
        model_list = ['Naive Bayes', 'KNN', 'Random Forest', 'Logistic Regression']
        n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf = self.improveModelRandomForest()
        C, solver = self.improveModelLogisticRegression()
        n_neighbors, p = self.improveModelKNN()

        # Models with the best hyperparameters as selected by random search CV method.
        model_names = {'Naive Bayes': GaussianNB(), 'KNN': KNeighborsClassifier(n_neighbors=n_neighbors, p=p),
                       'Random Forest': RandomForestClassifier(n_estimators=n_estimators,
                                                               max_features=max_features,
                                                               max_depth=max_depth,
                                                               min_samples_split=min_samples_split,
                                                               min_samples_leaf=min_samples_leaf,
                                                               bootstrap=True),
                       'Logistic Regression': LogisticRegression(C=C, solver=solver)}

        # Metrics DFs
        classification_metrics = pd.DataFrame(columns= ['estimate_score', 'Cross-Validation'], index=model_list)
        auc_metrics = pd.DataFrame(columns=['AUC_score'], index=model_list)
        parameters_metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1_Score'], index=model_list)
        train_score = pd.DataFrame(columns=['estimate_score_train', 'Cross-Validation_train'], index=model_list)

        for model_name, model_function in model_names.items():
            # Model Fitting and Prediction
            model_function.fit(self.x_train, self.y_train)
            y_prediction = model_function.predict(self.x_test)

            # Metrics Calculations
            classification_metrics = self.estimateModelPerformance(classification_metrics, model_name, model_function)
            train_score = self.estimateModelPerformanceTrainingSet(train_score, model_name, model_function)
            auc_metrics = self.evaluateModelsROC(auc_metrics, model_name, model_function)
            parameters_metrics = self.estimateClassificationParameters(parameters_metrics,
                                                                       model_name, model_function)
            self.confusionMatrix(model_name, model_function, y_prediction)

        print(classification_metrics)
        print(train_score)
        self.plotClassificationMetrics(classification_metrics)
        self.plotAUCMetrics(auc_metrics)
        self.plotParametersMetrics(parameters_metrics)



# Function Calls
# Creating an object of class CW_AML
cwObj = CW_AML()

# Data Normalisation and Analysis
cwObj.dataPreprocessing()
cwObj.exploratoryDataAnalysis()

# Model Implementation, Plotting and Tuning
cwObj.dataModelling()
cwObj.modelFitting()
cwObj.modelFittingAfterTuning()

