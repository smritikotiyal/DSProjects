import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sympy import stats, sqrt, exp, pi
from sklearn.preprocessing import LabelEncoder
import copy
import os, glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statistics import median
# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import pymc3 as pm
# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# Statistical Data Analysis for Student Performance
class CW_SDA:
    # Part 1
    '''---------------------------------------------------------------------------------------------------------------------------------------------------'''
    ## Dataset Exploration
    # Class Variables
    # Loads the dataset into Python
    df_merged = pd.read_csv('Merged.csv')
    df_normalised = pd.DataFrame(df_merged)
    labels = df_merged['G3']
    X_train, X_test, y_train, y_test = train_test_split(df_merged, labels, test_size=0.30)
    # df_merged_copy = df_merged.drop(columns=['school', 'G1', 'G2'])

    def statisticalDataExploration(self):
        # Displays the top 5 rows of the dataset with a view of the dataframe
        print(self.df_merged.head())

        # Displays all the columns of the dataframe
        pd.set_option('display.max_columns', None)

        # Displays top 5 rows of the dataset
        print(self.df_merged.head())

        # Diaplays the statistical details of all the columns of the dataframe
        include = ['object', 'float', 'int64']
        print(self.df_merged.describe(include = include))

        # Diplays the total number of rows and columns in the dataframe
        print('\nStructure of the dataframe in count of (rows, columns) format : \n', self.df_merged.shape,'\n ')

        print('Number of rows for each school : \n', self.df_merged.loc[:, 'school'].value_counts())

        # Gives the datatype of the columns
        print('Datatype of Each Column : \n',self.df_merged.dtypes)
        dataTypeDict = dict(self.df_merged.dtypes)

        # Gives the number of students from each school for maths and portuguese grading
        print('Number of rows for each school : \n', self.df_merged.loc[:, 'school'].value_counts())

        # Checking for Null values
        print('Number of Null Values : \n', pd.isnull(self.df_merged).sum())

        # Histogram of grades - Distribution of Grades
        plt.hist(self.df_merged['G3'], bins=14)
        plt.xlabel('G3')
        plt.ylabel('Count')
        plt.title('Distribution of Final Grades')
        plt.show()

        #self.df_merged.hist()
        self.df_merged.hist(bins = 14)
        plt.subplots_adjust(wspace=0.5)
        plt.suptitle('Distribution of Numerical Parameters of the Student Dataset')
        plt.show()

        for key, val in dataTypeDict.items():
            k = ''
            if (val == 'object' and key != 'school'):
                if (key == 'sex'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'F', 'G3'],
                                label='Female', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'M', 'G3'],
                                label='Male', shade=True)
                    k = 'Sex'

                elif (key == 'address'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'U', 'G3'],
                                label='Urban', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'R', 'G3'],
                                label='Rural', shade=True)
                    k = 'Address'

                elif (key == 'famsize'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'LE3', 'G3'],
                                label='Less then 3', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'GT3', 'G3'],
                                label='Greater', shade=True)
                    k = 'Family Size'
                elif (key == 'Pstatus'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'T', 'G3'],
                                label='Together', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'A', 'G3'],
                                label='Apart', shade=True)
                    k = 'Parents\' Living Arrangement'
                elif (key == 'Mjob'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'teacher', 'G3'],
                                label='Teacher', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'health', 'G3'],
                                label='Health Care', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'services', 'G3'],
                                label='Civil Services', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'at_home', 'G3'],
                                label='Homemaker', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'other', 'G3'],
                                label='Other', shade=True)
                    k = 'Mother\'s Job Status'
                elif (key == 'Fjob'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'teacher', 'G3'],
                                label='Teacher', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'health', 'G3'],
                                label='Health Care', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'services', 'G3'],
                                label='Civil Services', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'at_home', 'G3'],
                                label='Homemaker', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'other', 'G3'],
                                label='Other', shade=True)
                    k = 'Father\'s Job Status'
                elif (key == 'reason'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'home', 'G3'],
                                label='Closer to Home', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'course', 'G3'],
                                label='Course Spectrum', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'reputation', 'G3'],
                                label='Reputed', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'other', 'G3'],
                                label='Other', shade=True)
                    k = 'Reason of choice of School'

                elif (key == 'guardian'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'mother', 'G3'],
                                label='Mother', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'father', 'G3'],
                                label='Father', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'other', 'G3'],
                                label='Other', shade=True)
                    k = 'Guardian'

                elif (key == 'schoolsup'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'yes', 'G3'],
                                label='Taken', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'no', 'G3'],
                                label='Not Taken', shade=True)
                    k = 'Extra Educational Support'


                elif (key == 'famsup'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'yes', 'G3'],
                                label='Taken', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'no', 'G3'],
                                label='Not Taken', shade=True)
                    k = 'Family Educational Support'
                elif (key == 'paid'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'yes', 'G3'],
                                label='Taken', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'no', 'G3'],
                                label='Not Taken', shade=True)
                    k = 'Extra Paid Classes'
                elif (key == 'activities'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'yes', 'G3'],
                                label='Participated', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'no', 'G3'],
                                label='Not Participated', shade=True)
                    k = 'Extra-Curricular Activities'

                elif (key == 'nursery'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'yes', 'G3'],
                                label='Attended', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'no', 'G3'],
                                label='Not Attended', shade=True)
                    k = 'Nursery School'

                elif (key == 'higher'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'yes', 'G3'],
                                label='Yes', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'no', 'G3'],
                                label='No', shade=True)
                    k = 'Interest in Higher Studies'
                elif (key == 'internet'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'yes', 'G3'],
                                label='Internet Access Available', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'no', 'G3'],
                                label='Unavailable', shade=True)
                    k = 'Internet Access'

                elif(key == 'romantic'):
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'yes', 'G3'],
                                label='Yes', shade=True)
                    sns.kdeplot(self.df_merged.loc[self.df_merged[key] == 'no', 'G3'],
                                label='No', shade=True)
                    k = 'Romantic Relationship'

                plt.xlabel('Grade')
                plt.ylabel(k)
                plt.title('Density Plot of Final Grades by ' + k)
                plt.show()


    def listOfColumns(self):
        # List of Columns
        list1 = []
        for i in range(0, len(self.df_merged.axes[1])):
            if (self.df_merged.axes[1][i] != 'school' and self.df_merged.axes[1][i] != 'G1' and self.df_merged.axes[1][i] != 'G2'):
                list1.append(self.df_merged.axes[1][i])
        return list1

    # Function for Data Transformation
    def dataTransformation(self):

            print('Dataset before transformation : \n')
            print(self.df_merged.head())
            print('\n')

            # Converting Categorical Data to Numerical Data for correlation
            replace_map = {'school': {'GP': 1, 'MS': 2},
                           'sex': {'F': 1, 'M': 2},
                           'address': {'U': 1, 'R': 2},
                           'famsize': {'LE3': 1, 'GT3': 2},
                           'Pstatus': {'T': 1, 'A': 2},
                           'Mjob': {'teacher': 1, 'health': 2, 'services': 3, 'at_home': 4, 'other': 5},
                           'Fjob': {'teacher': 1, 'health': 2, 'services': 3, 'at_home': 4, 'other': 5},
                           'reason': {'home': 1, 'reputation': 2, 'course': 3, 'other': 4},
                           'guardian': {'mother': 1, 'father': 2, 'other': 3},
                           'schoolsup': {'yes': 1, 'no': 2},
                           'famsup': {'yes': 1, 'no': 2},
                           'paid': {'yes': 1, 'no': 2},
                           'activities': {'yes': 1, 'no': 2},
                           'nursery': {'yes': 1, 'no': 2},
                           'higher': {'yes': 1, 'no': 2},
                           'internet': {'yes': 1, 'no': 2},
                           'romantic': {'yes': 1, 'no': 2}}

            self.df_merged.replace(replace_map, inplace=True)
            print('Dataset after transformation : \n')
            print(self.df_merged.head())
            print('\n')
            self.df_merged.to_csv('merged_ML.csv', index=False, header=True)

    # Correlation between n Features
    def getCorrelation(self, n):
        corr_dict = {}
        corr_dict_copy = {}
        list1 = self.listOfColumns()
        for i in list1:
            corr_dict[i] = self.df_merged[i].corr(self.df_merged['G3'])

        # Sort the dictionary with column : correlation key-value pairs and feed the columns in a list.
        sorted_corr_dict = {k: v for k, v in sorted(corr_dict.items(), key=lambda item: item[1]) if k != 'G3'}
        print('The sorted list of ', len(sorted_corr_dict), ' feautures and their correlation with the final grades: \n')
        for key, val in sorted_corr_dict.items():
            print(key, ':', val)
            corr_dict_copy[key] = abs(val)

        sorted_corr_dict = {k: v for k, v in sorted(corr_dict_copy.items(), key=lambda item: item[1], reverse = True) if k != 'G3'}
        # Using the sorted list to fetch the 'n' most negatively and positively correlated columns
        i = 1
        max_corr_list = []
        for key, val in sorted_corr_dict.items():
            if(i <= n):
                max_corr_list.append(key)
                i += 1

        for i in max_corr_list:
            x = self.df_merged[i]
            y = self.labels
            plt.style.use('ggplot')
            plt.xlabel(i)
            plt.ylabel('G3')
            plt.title('correlation = ' + str(self.df_merged[i].corr(self.labels).round(2)))
            plt.scatter(x, y)
            plt.show()
        return max_corr_list

    def datasetSplit(self, final_col_list):
        # New data frame creation with selected features and labels
        self.df_merged = self.df_merged.drop(columns=['school', 'G1', 'G2'])
        self.df_normalised = pd.DataFrame(self.df_merged, columns=[final_col_list[0]])
        for i in range(1, len(final_col_list)):
            self.df_normalised[final_col_list[i]] = self.df_merged[final_col_list[i]].values
            self.df_normalised['G3'] = self.df_merged['G3'].values
            labels = self.df_normalised['G3']

        # Dataset Split - 70%-30%
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_normalised, labels, test_size=0.30)
        print('Training Dataframe : \n',self.X_train)
        print('Testing Dataframe : \n',self.X_test)
        print('Label Training Dataframe : \n',self.y_train)
        print('Label Testing Dataframe : \n',self.y_test)

    '''-------------------------------------------------------------------------------------------------------------------------------------------------------'''
    def calculateBenchmark(self):
        median_X_train = median(self.X_train['G3'])
        print('Median of G3 in the training dataframe : ', median_X_train)
        predictions = [median_X_train for i in range(len(self.X_test))]
        true = self.X_test['G3']
        mae = np.mean(abs(predictions - true))
        rmse = np.sqrt(np.mean((predictions - true) ** 2))
        return mae, rmse

    # Evaluate several ml models by training on training set and testing on testing set
    def statsModelsImplementation(self, benchmark_mae, benchmark_rmse):
        # Names of models
        model_name_list = ['Linear Regression', 'Naive Bayes',
                           'Random Forest', 'Logistic Regression',
                           'Gradient Boosted']
        # self.X_train = self.X_train.drop(columns='G3')
        # self.X_test = self.X_test.drop(columns='G3')
        X_train = self.X_train.drop(columns='G3')
        X_test = self.X_test.drop(columns='G3')

        # Instantiate the models
        model1 = LinearRegression()
        model2 = GaussianNB()
        model3 = RandomForestRegressor(n_estimators=100)
        model4 = LogisticRegression()
        model5 = GradientBoostingRegressor(n_estimators=100)

        # Dataframe for standard_metrics
        standard_metrics = pd.DataFrame(columns=['MAE', 'RMSE'], index=model_name_list)

        # Train and predict with each model
        for i, model in enumerate([model1, model2, model3, model4, model5]):
            model.fit(X_train, self.y_train)
            predictions = model.predict(X_test)

            # Metrics
            mae = np.mean(abs(predictions - self.y_test))
            rmse = np.sqrt(np.mean((predictions - self.y_test) ** 2))

            # Insert standard_metrics into the dataframe
            model_name = model_name_list[i]
            standard_metrics.loc[model_name, :] = [mae, rmse]

        standard_metrics.loc['Benchmark', :] = [benchmark_mae, benchmark_rmse]
        # self.plotStandardMetrics(standard_metrics)
        return standard_metrics

    def plotStandardMetrics(self, standard_metrics):
        # Plot standard metrics
        plt.figure(figsize=(12, 8))
        matplotlib.rcParams['font.size'] = 14
        # Root mean squared error
        ax = plt.subplot(1, 2, 1)
        standard_metrics.sort_values('MAE', ascending=True).plot.bar(y='MAE', color='g', ax=ax)
        plt.title('Model Mean Absolute Error')
        plt.ylabel('MAE')
        # Median absolute percentage error
        ax = plt.subplot(1, 2, 2)
        standard_metrics.sort_values('RMSE', ascending=True).plot.bar(y='RMSE', color='r', ax=ax)
        plt.title('Model Root Mean Squared Error')
        plt.ylabel('RMSE')
        plt.tight_layout()
        plt.show()

    def bayesianLRFormula(self, trace):
        blr_formula = 'G3 = %0.2f * Intercept' % (np.mean(trace['Intercept']))

        for i in trace.varnames:
            print(i)
            if (i != 'Intercept'):
                blr_formula += ' + %0.2f * %s' % (np.mean(trace[i]), i)
        print('blr : ',blr_formula)
        return blr_formula

    def applyBayesianSampling(self, final_col_list):
        formula = 'G3 ~ ' + ' + '.join([i for i in final_col_list])
        print('formula : ', formula)


        # Context for the model
        with pm.Model() as normal_model:
            # Setting the likelihood as a normal distribution
            family = pm.glm.families.Normal()

            # Creating the model using the family, data and formula
            pm.GLM.from_formula(formula, data=self.X_train, family=family)
            # Perform Markov Chain Monte Carlo sampling
            normal_trace = pm.sample(draws=3000, chains=2, tune=1200, cores=-1)

            print(pm.summary(normal_trace))
            pm.traceplot(normal_trace)
            plt.show()
            pm.plot_posterior(normal_trace)
            plt.show()
            blr_formula = self.bayesianLRFormula(normal_trace)
            print('blr_formula :', blr_formula)

        return normal_trace, blr_formula

    def posteriorPlots(self, final_col_list, trace):
        X_train_copy = self.X_train.drop(columns='G3')
        for i in final_col_list:
           # Variables that do not change
           constant_vars = list(X_train_copy.columns)
           constant_vars.remove(i)

           # Linear Model that estimates a grade based on the value of the query variable
           # and one sample from the trace
           def lm(value, sample):
               # Prediction is the estimate given a value of the query variable
               prediction = sample['Intercept'] + sample[i] * value

               # Each non-query variable is assumed to be at the median value
               for var in constant_vars:
                   # Multiply the weight by the median value of the variable
                   prediction += sample[var] * X_train_copy[var].median()

               return prediction

           plt.figure(figsize=(7, 7))
           # Find the minimum and maximum values for the range of the i
           var_min = X_train_copy[i].min()
           var_max = X_train_copy[i].max()
           pm.plot_posterior_predictive_glm(trace, samples=300,
                                            eval=np.linspace(var_min, var_max, 100), lm=lm,
                                            label='posterior predictive regression lines',
                                            lw=3., c='r')

           # Plot formatting
           plt.xlabel('%s' % i, size=16)
           plt.ylabel('Grade', size=16)
           plt.title("Posterior of Grade vs %s" % i, size=18)
           plt.show()

    def estimateGrades(self, trace, standard_metrics):

        X_test_copy = self.X_test
        y_test_copy = self.y_test

        # Dictionary of all sampled values for each parameter
        var_dict = {}
        for variable in trace.varnames:
            var_dict[variable] = trace[variable]
        # Results into a dataframe
        var_weights = pd.DataFrame(var_dict)
        # Means for all the weights
        var_means = var_weights.mean(axis=0)

        # Create an intercept column
        X_test_copy['Intercept'] = 1

        # Align names of the test observations and means
        X_test_copy = X_test_copy.drop(columns='G3')
        names = X_test_copy.columns[:]
        X_test_copy = X_test_copy.loc[:, names]
        var_means = var_means[names]

        # Calculate estimate for each test observation using the average weights
        results = pd.DataFrame(index=X_test_copy.index, columns=['estimate'])
        for row in X_test_copy.iterrows():
            results.loc[row[0], 'estimate'] = np.dot(np.array(var_means), np.array(row[1]))

        # Metrics
        actual = np.array(y_test_copy)
        pred = np.array(results['estimate'])
        errors = results['estimate'] - actual
        mae = np.mean(abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        plt.plot(actual)
        plt.plot(pred, color='red')
        plt.gca().set(title='Actual V/S Estimated Students Final Grades',
                      xlabel='Number of Rows',
                      ylabel='Grades')

        plt.show()

        standard_metrics.loc['Bayesian Linear Regression', :] = [mae, rmse]
        print(standard_metrics)
        self.plotStandardMetrics(standard_metrics)

    def makePredictions(self, trace, test_observation):
        # Print out the test observation data
        print('Test Observation:')
        print(test_observation)
        var_dict = {}
        for variable in trace.varnames:
            var_dict[variable] = trace[variable]

        # Results into a dataframe
        var_weights = pd.DataFrame(var_dict)
        print('var_weights : ', var_weights)

        # Standard deviation of the likelihood
        sd_value = var_weights['sd'].mean()
        print('sd_value : ', sd_value)

        # Actual Value
        actual = test_observation['G3']

        # Add in intercept term
        test_observation['Intercept'] = 1
        test_observation = test_observation.drop('G3')

        # Align weights and test observation
        var_weights = var_weights[test_observation.index]
        print('test_observation.index : ', test_observation.index)
        print('test_observation : ', test_observation)

        # Means for all the weights
        var_means = var_weights.mean(axis=0)
        print('var means : ', var_means)

        # Location of mean for observation
        mean_loc = np.dot(var_means, test_observation)
        print('mean_loc : ', mean_loc)

        # Estimates of grade
        estimates = np.random.normal(loc=mean_loc, scale=sd_value,
                                     size=1000)

        # print(estimates)
        # Plot all the estimates
        plt.figure(figsize=(8, 8))
        sns.kdeplot(estimates, alpha=0.5, shade=True, color='green')

        # Plot the actual grade
        plt.vlines(x=actual, ymin=0, ymax=0.2,
                   linestyles='-', colors='black',
                   label='True Grade',
                   linewidth=2.5)

        # Plot the mean estimate
        plt.vlines(x=mean_loc, ymin=0, ymax=0.2,
                   linestyles='-', colors='red',
                   label='Mean Estimate',
                   linewidth=2.5)

        mark_n_68 = mean_loc - sd_value
        mark_n_95 = mean_loc - (2 * sd_value)

        mark_p_68 = mean_loc + sd_value
        mark_p_95 = mean_loc + (2 * sd_value)

        # Plot the 68% area of empirical rule
        plt.vlines(x=mark_n_68, ymin=0, ymax=0.075,
                   linestyles='dashed', colors='blue',
                   label='mu +- sd',
                   linewidth=2.5)
        plt.vlines(x=mark_p_68, ymin=0, ymax=0.075,
                   linestyles='dashed', colors='blue',
                   linewidth=2.5)

        # Plot the 95% area of empirical rule
        plt.vlines(x=mark_n_95, ymin=0, ymax=0.025,
                   linestyles='dashed', colors='pink',
                   label='mu +- 2sd',
                   linewidth=2.5)
        plt.vlines(x=mark_p_95, ymin=0, ymax=0.025,
                   linestyles='dashed', colors='pink',
                   linewidth=2.5)

        plt.legend(loc=1)
        plt.title('Density Plot for Test Observation')
        plt.xlabel('Grade')
        plt.ylabel('Density')
        plt.show()
        # Prediction information
        print('True Grade = ', actual)
        print('Average Estimate = ', mean_loc)

# Function Calls - Part 1
# Data Transformation Call
obj1 = CW_SDA()
obj1.statisticalDataExploration()
obj1.dataTransformation()
num_features = int(input('How many student parameters do you want to find the correlation of the final Grades with? \n'))
final_col_list = obj1.getCorrelation(num_features)
print('max_corr_list ', final_col_list)
obj1.datasetSplit(final_col_list)

# Function Calls - Part 2
# Naive baseline is the median
benchmark_mae, benchmark_rmse = obj1.calculateBenchmark()
# Display the naive baseline metrics
print('Benchmark  MAE: ', round(benchmark_mae, 4))
print('Benchmark RMSE: ', round(benchmark_rmse, 4))

standard_metrics = obj1.statsModelsImplementation(benchmark_mae, benchmark_rmse)
print(standard_metrics)
obj1.plotStandardMetrics(standard_metrics)

# Implement BLR
trace, blr_formula = obj1.applyBayesianSampling(final_col_list)
obj1.posteriorPlots(final_col_list,trace)
obj1.estimateGrades(trace, standard_metrics)

# Prediction 
obj1.makePredictions(trace, obj1.X_test.iloc[41])
obj1.makePredictions(trace, obj1.X_test.iloc[100])
obj1.makePredictions(trace, obj1.X_test.iloc[111])
obj1.makePredictions(trace, obj1.X_test.iloc[15])
obj1.makePredictions(trace, obj1.X_test.iloc[200])
obj1.makePredictions(trace, obj1.X_test.iloc[11])
