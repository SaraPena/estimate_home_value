"Walk through modeling for zillow data"

import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind, pearsonr
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

import env
import acquire
import prep
import split_scale
import features
import model


""" 
Deliverables:

1. A report (in for the form of a presentaion, both verbal and through slide)

Summarize your findings about the drivers of the Tax Value Count.
This will come from the analysis you do during the exploration phase of the pipeline.
In the report you will have charts that visually tell the story of what is driving the errors.

2. A github repository containing the jupyter notebook that walks through the pipeline along with the .py files necessary to reproduce your model.
"""
"""

The Pipeline

***PROJECT PLANNING AND README***
"""
"""
Zillow Data Science Team Project Directive: 
Find qualities of properties that strongly influence the value of a home. Create a model that could be used to begin to predict a value of a home before an assesment is submitted. 
The data science team wants your model to have 95% a confidence interval. 

Which independent variables do you think influence the value of a home?

Hypothesis:

Variables that can influence the value of home are square feet,  and room count(bedroom, and bath). 
These two variables can have an influence no matter where the property located.

The more square feet of a property the more the value of the home will rise.
The higher the room count the cost will increase.
The total square footage of the property will be a stronger influence than room count

When you start to get into specific locations other factors can influence the value of a home such as zip code, school district, distance to destinations (work, home, restaurants, grocery).
Also home amenities will influence the value of a property, such as backyards with pools, garage size, hoa.
The age of home would also have influence.

To begin our modeling we will look at the features of squarefootage and room count to see how these will influence the target variable - value of the property. Further exploration of other property, and location specific features can be explored after this first phase is built.
""" 

"""
***ACQUIRE***

Goal: Gather and describe data from zillow dataset that creates a dataframe with:
    Index : 
        parcelid,fips 
    Features:
        calculatedfinishedsquarefeet
        bathroomcnt
        bedroomcnt
        lotsizesquarefeet
    
    Include head of the dataset, datatypes, summary stats, shape of the dataframe.

"""

# get url for zillow database using your env file. Your env file should include variables that are assigned to your username, password, and host for mysql.

url = acquire.get_db_url('zillow')

# Create dataframe for your features and target variable using your query, and url variable.
# The features I was interested in exploring were bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, and lotsizesquarefeet. The target variable I want to predict is taxvaluedollarcnt

query = """ SELECT parcelid,
                   fips,
                   bathroomcnt, 
                   bedroomcnt, 
                   calculatedfinishedsquarefeet, 
                   lotsizesquarefeet,
                   taxvaluedollarcnt
            FROM properties_2017
            JOIN predictions_2017 using(parcelid)
            WHERE (propertylandusetypeid = 261) and (transactiondate BETWEEN '2017-05-01' and '2017-06-30') """

# From the acuire.py create your dataframe with get_data_from_mysql(query,db)

df = acquire.get_data_from_mysql(query,'zillow')

# Head of dataframe
df.head()

# data types of dataframe:
df.dtypes

"""
All of our data types are floats or integers which are types that we can measure and quatify. There is not a need at this point to convert the features to another data type.
"""

# inital summary stats:
initial_df = df.describe()
initial_max = df.max()
initial_min = df.min()

initial_df
initial_max
initial_min

# dataframe columns:
df.columns

# shape of dataframe:
df.shape

"""
## PREP ##

Goal: 
    Create a dataset that is ready to be analyzed. 
    Datatypes are appropriate, null values, and integrity issues have been addressed.

"""

# Find amount of nulls in dataframe:

df.isnull().sum()

# There are 24 null values only in 'calculatedfinishedsquarefeet', and 124 Null values in 'lotsizesquarefeet'.  Because we have 15,034 rows in a our data frame we will drop out those values, because we believe that we have enough data to begin to make our models for predictions.
# Use prep.py to clean the dataframe by dropping the rows with null values. As stated earlier because we are working with numeric data types we do not need to change the datatypes of any of our variables.

clean_df = prep.clean_data(df)

# Lets look at our descriptive stats to see if there any major changes

clean_df.describe()
clean_df.shape


# My preference is to have my index be my parcel_id, and fips number so that when I am splitting into train, test and scaling. I be won't be scaling my 'parcelid' , and 'fips' (identifies county of the property ) and the integrity of those columns will be intact.
# Now that the data is clean I will rename it to be the dataframe - 'df'.

df = clean_df .set_index(['parcelid','fips'])
df.head()

"""
plot the distributions of independent variables. 
"""

# Bathroomcnt box plots
sns.boxplot(df.bathroomcnt, palette = 'husl')
plt.xlabel('Bathroom count')
plt.title('Bathroom Box Plot')
plt.show()

"""
There are outliers in properties that have more than 4 bathrooms
"""

# bathroom count distribution, because bathrooms are a discrete variable we will plot the distribution using a histogram

plt.hist(df.bathroomcnt, bins = [x*.5 for x in range(0,23)], color = 'red')
plt.xlabel('bathroom count')
plt.ylabel('count')
plt.title('Bathroom Distribution')
plt.show()

df.bathroomcnt.value_counts().sort_index()


# Bedroomcnt box plots

sns.boxplot(df.bedroomcnt, color = 'blue')
plt.xlabel('Bedroom count')
plt.title('Bedroom Box Plot')
plt.show()

# bedroom count distribution, because bedrooms are a discrete variables we will plot the distibution using a histogram.

plt.hist(df.bedroomcnt, bins = [x for x in range(0,13)], color = 'red')
plt.xlabel('Bedroom count')
plt.ylabel('count')
plt.title('Bedroom Distribution')
plt.show()

df.bedroomcnt.value_counts().sort_index()

"""
We see outliers as well for bedrooms with less than 2 and more than 5
"""

"""
For bedroom and bathroom counts most of the values lie between 1 and 6.
"""

#Calculatedfinished squarefeet box plots
sns.boxplot(df.calculatedfinishedsquarefeet, color = 'green')
plt.xlabel('count')
plt.title('Calculated Finished Square Feet Box Plot')
plt.show()

# calculatedfinishedsquarefeet is also a discrete variable.

plt.hist(df.calculatedfinishedsquarefeet, bins = 70)
plt.xlabel('Square Feet')
plt.ylabel('count')
plt.title('Square Footage Distribution')
plt.show()

df.calculatedfinishedsquarefeet.value_counts(bins = 70).sort_index()


# For square footage most properties are between 950 and 4400 square feet.

# lotsizesquarefeet is a discrete variables. We will split the data into bins to view where most of the properties are at.

df.lotsizesquarefeet.value_counts(bins = 100).sort_index().idxmax()

plt.hist(df.lotsizesquarefeet, bins = 100)
plt.xlabel('Square Feet')
plt.xscale('log')
plt.ylabel('count')
plt.title('Lot Size Distribution')


"""
Data Dictionary:
Independent Variables:

bathroomcnt: Count of total number of bathrooms on the property. Did not choose to use calculatedbathnbr because this only took into account bathrooms attached to bedrooms.
bedroomcnt: Count of total number of bedrooms on the property. Did not choose to use calculatedbathnbr because this only took into account bathrooms attached to bedrooms.
calculatedfinishedsquarefeet: Total amount of squarefootage of the property. This does not include lot size. There were other finishedsquare feet columns 12, 13, 50, 15, 6. These other finished square feet columns contained significant amounts of NULL values that would have effected how many data samples we had. Not that having more data is better, we just had another columns with finishedsquarefeet that could represent the living squarefootage of the property.
lotsizesquarefeet: Total amount of square feet of the lot. This will be a good value to use because we are looking at property square footage as well that is only the amount of living squarefeet. The lotsize can tell us about the value of the property as well.

Target Variable:
taxvaluedollarcount: This data calculated both the structuretaxvaluedollarcnt and landtaxvaluedollarcnt. Taxvaluedollarcnt was the similar way to predict how much someone might pay for a property.

Index:
parcelid: to keep some of the integrity of the data we will use the parcelid number of the properties as index values. This will help in future research of looking at characteristics of specific properties, and how they influence their property value.
fips: assigned county of the property
    fips locations:
        6037 - California - Los Angeles County
        6059 - California - Orange County
        6111 - California - Venture County

Scaling our data:
Our independent variables, and target variables are measured in two different ways:
    1. Square footage - calculatedfinishedsquarefeet, lotsizesquarefeet
    2. Quanity Count - bedroomcnt, bathroomcnt
    3. Cost - taxvaluedollarcnt

There are some outliers in our data where they have a high bedroom, bathroom, or calculated squarefootage. Because of these values we will use a Robust Scaler. The centering and scaling of the Robust scaler are based on percentiles. This means that the scaling is not influenced by a few number of very large marginal outliers. 
<https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html>

Erroneous or Invalid Data:
There was not outstanding erroneous data found in the preperation. We are going to keep properties that have bedrooms, or bathrooms as 0 because they still have a calculatedfinsihedsquarefeet. The squarefootage will drive the values of the property more than the room count. 
The values that were droped from the dataframe are the null values in calculated finished square feet.
"""

""" 
*** Split and Scale:

Goal: 

Create 2 dataframes (train & test) from the prepared dataframe. 
Use the test & train to create scaled dataframes for our features (bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, lotsizesquarefeet), and target variable (taxvaluedollarcnt)

To scale the data we will use the Robust Scaler, as stated in our prep phase.
"""

# use split_scale.py function split_my_data(df) to create train and test dateframes
train, test = split_scale.split_my_data(df)


# use split_scale.py function iqr_robust_scaler(train,test) to scale the data so they will be on the same unit scale. We do this because in the prep phase we saw that some variables are measured in squarefeet, and rooms are just a count value.
scaler, train_scaled, test_scaled = split_scale.iqr_robust_scaler(train, test)

# Look at our scaled data
train_scaled.head()
test_scaled.head()

# Create our X and y dataframes from our scaled data

# X_train_scaled dataframe will be our indepenent variables : bathroomcnt, bedroomcnt,  caluclatedfinishedsquarefeet, lotsizesquarefeet.
# y_train_scaled dataframe will be out target variable: taxvaluedollarcnt

X_train_scaled = train_scaled[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet']]
y_train_scaled = train_scaled[['taxvaluedollarcnt']]

# X_test_scaled dataframe will be 20% of our dataframe and reflect the variables in our training datasets.
X_test_scaled = test_scaled[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet']]
y_test_scaled = test_scaled[['taxvaluedollarcnt']]

"""
***Data Exploration***
Goal: Address questions posed in the planning brainstorming phase, and any other questions that have come up along the way through visual or statiscal analysis
"""
""" 
** Recap from hypothesis above:

Which independent variables do you think influence the value of a home?

Hypothesis:

Variables that can influence the value of home are square feet,  and room count(bedroom, and bath). 
These two variables can have an influence no matter where the property located.

The more square feet of a property the more the value of the home will rise.
The higher the room count the cost will increase.
The total square footage of the property will be a stronger influence than room count

When you start to get into specific locations other factors can influence the value of a home such as zip code, school district, distance to destinations (work, home, restaurants, grocery).
Also home amenities will influence the value of a property, such as backyards with pools, garage size, hoa.
The age of home would also have influence.
"""

# look at the correlation values between independent variables, and grab the correlation column that compares the variables to our target taxvaluedollarcnt.
train_scaled.corr()
train_scaled.corr().taxvaluedollarcnt

# Look at the correlation of a variables with sns.PairGrid
g = sns.PairGrid(train_scaled, palette = 'reds')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);

# Look at the correlation of variables with sns.heatmap 
plt.figure(figsize = (8,6))
sns.heatmap(train.corr(), cmap = 'Blues', annot = True )

"""
# T test - Is the average taxvaluedollarcnt different for one bedroom vs. three bedrooms?
# H[0]: The average taxvaluedollarcount of properties with one bedroom or three bedrooms is the same
"""

# We'll need to create two seperate datasets that contain the values for taxvaluedollarcnt for properties with one bedroom and properties with two bedrooms

bedrooms_1 = train_scaled[train_scaled['bedroomcnt'] == 1]
bedrooms_3 = train_scaled[train_scaled['bedroomcnt'] == 3]

test_results = ttest_ind(bedrooms_1.taxvaluedollarcnt, bedrooms_3.taxvaluedollarcnt)

# Because the p value is so small, we reject the null hypothesis. We think there is a significant difference in the average taxvaluedollarcnt of properties with 1 bedroom and properties with 3 bedrooms.
# Let's look at those averages:

bedrooms_1.taxvaluedollarcnt.mean()
bedrooms_3.taxvaluedollarcnt.mean()

"""
# T Test - Is the average taxvaluedollarncnt different for one bathroom vs. three bathroom properties?
# H[0]: The average taxvaluedollarcnt of properties with one bathroom or three bathrooms is the same.
"""

bathrooms_1 = train_scaled[train_scaled['bathroomcnt'] == 1]
bathrooms_3 = train_scaled[train_scaled['bathroomcnt'] == 3]

test_results = ttest_ind(bathrooms_1.taxvaluedollarcnt, bathrooms_3.taxvaluedollarcnt)

# Because the p value is so small, we reject the null hypothesis. We think there is a significant difference in the average taxvaluedollarcnt of properties with 1 bathroom and properties with 3 bedrooms.
# Let's look at those averages:

bathrooms_1.taxvaluedollarcnt.mean()
bathrooms_3.taxvaluedollarcnt.mean()

"""
# Pearson R - Are bathrooms and bedrooms linearly correlated, and what is the strength of that correlation?
# H[0]: There is not a linear correlation between number of bathrooms and number of bedrooms for a property.
"""

# We can pass the two series that contain the values we are looking at to the pearsonr function from scipy's stats module.
test_results_pearsonr = pearsonr(train_scaled.bathroomcnt, train_scaled.bedroomcnt)

"""
# because pearsonr is 0.0 we reject the null hypothesis that there is no linear relationship. The test also tells us the r^2 value of .645
"""

"""
# Pearson R - Are bathrooms and calculatedfinished square feet linearly correlated, and what is the strength of that correlation?
# H[0]: There is not a linear correlation between number of bathrooms and calculatedfinished sqaure feet for a property.
"""

# We can pass the two series that contain the values we are looking at to the pearsonr function from scipy's stats module.
test_results_pearsonr = pearsonr(train_scaled.bathroomcnt, train_scaled.calculatedfinishedsquarefeet)

"""
# because pearsonr is 0.0 we reject the null hypothesis that there is no linear relationship. The test also tells us the r ^2 value of .852 that this the strength of the relationship.
"""

"""
Take aways from exploration:
Calculated finished square feet is the most correlated to our dependent (target variable) taxvaluedollarcnt. From our hypothesis that our presumption that calculated finished square feet will influence taxvalue dollar count in a positive direction (overall taxvaluedollarcnt increases when calculated finished squarefeet increases)

We see that calculated finished square feet correlates with both bedroom, and bathroom count. This makes sense because the amount of liveable square fee will drive the amount of bedrooms, and bathrooms on a property. This finding is going to influence the feature selection. It gives the direction that you could combine bedroom, and bathroom count into one feature. Then use this combined room count feature in the selection process. 

Don't like that independent variables are correlated with each other so in feature selection process and modeling would want to see which of those give us better prediction values.
"""

"""
*** Feature Selection ***

Goal: Create a dataframe(s) with the features to be used to build your model.

"""

# Perform feature selection using RFE. Use features.py file function optimal_number_of_features to perform recursive feature elimination that will tell us the number of features to use to predict our target variable
number_of_features, score = features.optimal_number_of_features(X_train_scaled, y_train_scaled)

number_of_features, score

# Use recursive feature elimination. Use features.py file function optimal features to find out which features should be used
selected_features_rfe, X_train_rfe, X_test_rfe = features.optimal_features(X_train_scaled, X_test_scaled, y_train_scaled, number_of_features)

selected_features_rfe

"""
*** Modeling & Evaluation ***
Goal: develop a regression model that performs better than using overall average taxvaluedollarcnt as a baseline.
"""

predictions_train, predictions_test = model.modeling_function(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)

x = X_train_scaled[['bedroomcnt','bathroomcnt']]
y = y_train_scaled[['taxvaluedollarcnt']]

model.plot_residuals(x,y)
