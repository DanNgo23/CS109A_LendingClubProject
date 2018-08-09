
# Preparing the Data

## A Description of the Data 

For this project statement, we will use 8 datasets. The first 7 datasets are accepted loans from the fiscal quarters in [2016 and 2017](https://github.com/Polkaguy/LendingClubData/tree/master/Loan%20Data). Each has over 100,000 rows and all 4 were concantenated by row into a single dataset of accepted loans. Each featured the same columns, but not all columns were used. Since there are an abundance of columns, it would be to perform EDA if we selected only a subset of significant columns and focused on that subset. The 5th dataset was the rejected loan dataset from the Lending Club website that featured over 700,000 rows. You can see below in the "`acc_fields`" variable which columns were selected but the true subset is much less. The variables "`emp_length`" (years of employment), "`fico_range`" (FICO score) from the accepted loans dataset and "`Debt-to-Income Ratio`" from the rejected loans dataset are used for exploratory data analysis, considering that these are the variables that will serve as predictors in our baseline logistic regression model. 

The types of data in this column are objects (strings) and floats (64 bit) and they were read as such. There is discrete data (like "`fico_range`") continuous data (like "`interest rate`" and "`installment`"), and categorical data ("`purpose`" and "`addr_state`"). Some of this data had to be cleaned, as you'll see below. 

The process of cleaning and preparing the data for EDA and model building are as follows. The datasets from each fiscal quarter in 2016 and first 3 quarters in 2017 are concatenated into one dataset called **acc_data**, which has only a subset of columns (to save memory). The reason only the first 3 fiscal quarters of 2017 are used instead of all 4 was so that we can get an equal amount of rows for the combined accepted loans and the individual rejected loans dataset. The combined accepted loan data set (a combination of 7 datasets) is over 700,000 rows, equal to that of the rejected dataset. The rejected loan dataset from the LendingClub website was read and stored into **rej_data**. 

We see that home state, years of employment, and the first 3 zip code numbers are in both datasets. These are the first 3 columns that will be in the final concatenated dataset of all accepted and rejected loan applications. We also notice that **rej_data** has a "`Debt-to-Income Ratio`" column that **acc_data** does not have. Considering that this seems like a variable that could be a nice predictor in our prediction models, we would need to see if we can engineer this predictor in **acc_data**. And it turns out, that we can. By taking the loan amount that was requested (treating this as debt) and by taking the reported income, we can divide the loan amount by annual income to get our own "`Debt-to-Income Ratio`" in **acc_data**. Next, we see that **acc_data** has a lower and upper bound for borrower’s FICO score at loan origination. These two columns are averaged into one new column. For **rej_data**, FICO score was called "`Risk_Score`" in the dataset. Next, the FICO score columns in both datasets are renamed into "`FICO score`". This is now another common column within **acc_data** and **rej_data** and is thus another column that will be featured in the newly concatenated dataset featuring both accepted loan applications and rejected loan applications. With these 5 common columns- "`zip_code`", "`state`", "`emp_length`", "`Debt-to-Income-Ratio`", and "`FICO score`", we can now reduce **acc_data** and **rej_data** to **acc_piece** and **rej_piece** (these new datasets have the 5 common columns). Finally, we concatenate **acc_piece** and **rej_piece** by row to get our finalized dataset that we will be working with for EDA and machine learning. It is called **both_cats** (short for "both categories"). A binary variable was added to this new dataset to denote each row whether it was an accepted or rejected loan (this will be the reponse variable in our model). Using one-hot encoding, zip code column is turned into dummy variables for each unique value. We will add one more column: a *natural log* of the "`Debt-to-Income Ratio`" variable (the variable is heavily skewed). Any row with null values in any column are removed. There are over a million rows in total. About 700,000 of them are classified as accepted loans; about 700,000 are classified rejected loans. We have an equal amount of both classes represented.



```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import re
from scipy import stats
```


```python
acc_fields = ['id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment', 'grade', 
              'sub_grade', 'emp_title','emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'issue_d', 
              'loan_status', 'pymnt_plan', 'purpose', 'title', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs',
              'earliest_cr_line', 'fico_range_low', 'fico_range_high']
types = {'id': object, 'loan_amnt': 'float64', 'funded_amnt': 'float64', 'funded_amnt_inv': 'float64', 'term': object, 'int_rate': object, 'installment': 'float64', 'grade': object, 
              'sub_grade': object, 'emp_title': object,'emp_length': object, 'home_ownership': object, 'annual_inc': 'float64', 'verification_status': object, 'issue_d': object, 
              'loan_status': object, 'pymnt_plan': object, 'purpose': object, 'title': object, 'zip_code': object, 'addr_state': object, 'dti': 'float64', 'delinq_2yrs': 'float64',
              'earliest_cr_line': object, 'fico_range_low': 'float64', 'fico_range_high': 'float64'}

acc_data1 = pd.read_csv("LoanStats_securev1_2016Q1.csv", 
                        header=1, usecols=acc_fields, dtype=types)
acc_data2 = pd.read_csv("LoanStats_securev1_2016Q2.csv", 
                        header=1, usecols=acc_fields, dtype=types)
acc_data3 = pd.read_csv("LoanStats_securev1_2016Q3.csv", 
                        header=1, usecols=acc_fields, dtype=types)
acc_data4 = pd.read_csv("LoanStats_securev1_2016Q4.csv", 
                        header=1, usecols=acc_fields, dtype=types)
acc_data5 = pd.read_csv("LoanStats_securev1_2017Q1.csv", 
                        header=1, usecols=acc_fields, dtype=types)
acc_data6 = pd.read_csv("LoanStats_securev1_2017Q2.csv", 
                        header=1, usecols=acc_fields, dtype=types)
acc_data7 = pd.read_csv("LoanStats_securev1_2017Q3.csv", 
                        header=1, usecols=acc_fields, dtype=types)
#acc_data8 = pd.read_csv("LoanStats_securev1_2017Q4.csv", 
#                        header=1, usecols=acc_fields, dtype=types)
acc_data = pd.concat([acc_data1, acc_data2, acc_data3, acc_data4, 
                      acc_data5, acc_data6, acc_data7])
acc_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>...</th>
      <th>pymnt_plan</th>
      <th>purpose</th>
      <th>title</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>earliest_cr_line</th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75828813</td>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>60 months</td>
      <td>6.97%</td>
      <td>237.45</td>
      <td>A</td>
      <td>A3</td>
      <td>Teacher</td>
      <td>...</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>210xx</td>
      <td>MD</td>
      <td>13.00</td>
      <td>0.0</td>
      <td>Apr-2003</td>
      <td>720.0</td>
      <td>724.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75973524</td>
      <td>8000.0</td>
      <td>8000.0</td>
      <td>8000.0</td>
      <td>36 months</td>
      <td>11.99%</td>
      <td>265.68</td>
      <td>C</td>
      <td>C1</td>
      <td>Boat Captain</td>
      <td>...</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>325xx</td>
      <td>FL</td>
      <td>13.68</td>
      <td>0.0</td>
      <td>May-2009</td>
      <td>685.0</td>
      <td>689.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>76091670</td>
      <td>8200.0</td>
      <td>8200.0</td>
      <td>8200.0</td>
      <td>36 months</td>
      <td>11.47%</td>
      <td>270.29</td>
      <td>B</td>
      <td>B5</td>
      <td>Registered Nurse</td>
      <td>...</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>391xx</td>
      <td>MS</td>
      <td>12.74</td>
      <td>0.0</td>
      <td>Jun-2006</td>
      <td>675.0</td>
      <td>679.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75114281</td>
      <td>14000.0</td>
      <td>14000.0</td>
      <td>14000.0</td>
      <td>36 months</td>
      <td>10.75%</td>
      <td>456.69</td>
      <td>B</td>
      <td>B4</td>
      <td>Owner</td>
      <td>...</td>
      <td>n</td>
      <td>credit_card</td>
      <td>NaN</td>
      <td>777xx</td>
      <td>TX</td>
      <td>15.68</td>
      <td>0.0</td>
      <td>Dec-2003</td>
      <td>680.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75276111</td>
      <td>15000.0</td>
      <td>15000.0</td>
      <td>15000.0</td>
      <td>36 months</td>
      <td>9.16%</td>
      <td>478.12</td>
      <td>B</td>
      <td>B2</td>
      <td>mainteance</td>
      <td>...</td>
      <td>n</td>
      <td>home_improvement</td>
      <td>Home improvement</td>
      <td>285xx</td>
      <td>NC</td>
      <td>12.50</td>
      <td>0.0</td>
      <td>Dec-2003</td>
      <td>690.0</td>
      <td>694.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
acc_data.columns
```




    Index(['id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
           'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
           'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
           'loan_status', 'pymnt_plan', 'purpose', 'title', 'zip_code',
           'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
           'fico_range_low', 'fico_range_high'],
          dtype='object')




```python
rej_data = pd.read_csv("RejectStatsA.csv", header=1)
rej_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amount Requested</th>
      <th>Application Date</th>
      <th>Loan Title</th>
      <th>Risk_Score</th>
      <th>Debt-To-Income Ratio</th>
      <th>Zip Code</th>
      <th>State</th>
      <th>Employment Length</th>
      <th>Policy Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000.0</td>
      <td>2007-05-26</td>
      <td>Wedding Covered but No Honeymoon</td>
      <td>693.0</td>
      <td>10%</td>
      <td>481xx</td>
      <td>NM</td>
      <td>4 years</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000.0</td>
      <td>2007-05-26</td>
      <td>Consolidating Debt</td>
      <td>703.0</td>
      <td>10%</td>
      <td>010xx</td>
      <td>MA</td>
      <td>&lt; 1 year</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11000.0</td>
      <td>2007-05-27</td>
      <td>Want to consolidate my debt</td>
      <td>715.0</td>
      <td>10%</td>
      <td>212xx</td>
      <td>MD</td>
      <td>1 year</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6000.0</td>
      <td>2007-05-27</td>
      <td>waksman</td>
      <td>698.0</td>
      <td>38.64%</td>
      <td>017xx</td>
      <td>MA</td>
      <td>&lt; 1 year</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1500.0</td>
      <td>2007-05-27</td>
      <td>mdrigo</td>
      <td>509.0</td>
      <td>9.43%</td>
      <td>209xx</td>
      <td>MD</td>
      <td>&lt; 1 year</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
rej_data.columns
```




    Index(['Amount Requested', 'Application Date', 'Loan Title', 'Risk_Score',
           'Debt-To-Income Ratio', 'Zip Code', 'State', 'Employment Length',
           'Policy Code'],
          dtype='object')




```python
rej_data.shape
```




    (755491, 9)




```python
debt_to_inc = (acc_data['loan_amnt'] / acc_data['annual_inc']) * 100
avg_fico = (acc_data['fico_range_low']+acc_data['fico_range_high'])/2
accepted = [1] * len(acc_data)
rejected = [0] * len(rej_data)
```


```python
rej_piece = rej_data.loc[:,('Zip Code', 'Employment Length', 
                            'Debt-To-Income Ratio', 'Risk_Score')]
acc_piece = acc_data.loc[:,('zip_code', 'emp_length')]

acc_piece['Debt-To-Income Ratio'] = debt_to_inc
acc_piece['FICO score'] = avg_fico

acc_piece['accepted'] = accepted
rej_piece['accepted'] = rejected

acc_piece.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zip_code</th>
      <th>emp_length</th>
      <th>Debt-To-Income Ratio</th>
      <th>FICO score</th>
      <th>accepted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>210xx</td>
      <td>8 years</td>
      <td>22.222222</td>
      <td>722.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>325xx</td>
      <td>&lt; 1 year</td>
      <td>10.000000</td>
      <td>687.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>391xx</td>
      <td>10+ years</td>
      <td>11.714286</td>
      <td>677.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>777xx</td>
      <td>10+ years</td>
      <td>17.500000</td>
      <td>682.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>285xx</td>
      <td>2 years</td>
      <td>27.272727</td>
      <td>692.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
rej_piece.rename(columns={'Zip Code': 'zip_code', 
                          'Employment Length': 'emp_length', 
                          'Risk_Score': 'FICO score'}, inplace=True)
rej_piece['Debt-To-Income Ratio'] = \
rej_piece['Debt-To-Income Ratio'].apply(lambda x: float(x[:-1]))
rej_piece.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zip_code</th>
      <th>emp_length</th>
      <th>Debt-To-Income Ratio</th>
      <th>FICO score</th>
      <th>accepted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>481xx</td>
      <td>4 years</td>
      <td>10.00</td>
      <td>693.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>010xx</td>
      <td>&lt; 1 year</td>
      <td>10.00</td>
      <td>703.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>212xx</td>
      <td>1 year</td>
      <td>10.00</td>
      <td>715.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>017xx</td>
      <td>&lt; 1 year</td>
      <td>38.64</td>
      <td>698.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>209xx</td>
      <td>&lt; 1 year</td>
      <td>9.43</td>
      <td>509.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We will take only the number from emp_length. This will be a float variable. Someone with years of employment < 1 will be reclassified to 0.5. Someone with 10+ years will just be 10. 


```python
# clean up the emp_length column
def change_years(years):
    if type(years) is str:
        if years == '< 1 year':
            return 0.5
        elif years == '10+ years':
            return 10
        else:
            return int(years[0])
    else:
        return np.nan
```


```python
acc_piece = acc_piece.dropna(axis=0)
acc_piece.shape
```




    (708974, 5)




```python
rej_piece = rej_piece.dropna(axis=0)
rej_piece.shape
```




    (723534, 5)




```python
both_cats = pd.concat([acc_piece, rej_piece])
#both_cats = both_cats.dropna(axis=0)
# filter out those under 0 and those near infinity
both_cats = both_cats[(both_cats['Debt-To-Income Ratio'] >= 0) & (both_cats['Debt-To-Income Ratio'] < math.inf)]
# Debt-To-Income Ratio is heavily skewed, which is why we take the natural log of that column
both_cats['log_DtIR'] = np.log(both_cats['Debt-To-Income Ratio']+1)
# convert the years of employment column to numeric
both_cats['emp_length'] = both_cats['emp_length'].apply(change_years)
# get dummy variables for zip code
zip_dummies = pd.get_dummies(both_cats.zip_code, drop_first=True)
both_cats = pd.concat([both_cats, zip_dummies], axis=1)
both_cats = both_cats.drop(['Debt-To-Income Ratio', 'zip_code'], axis=1)
both_cats = both_cats.dropna(axis=0)
both_cats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emp_length</th>
      <th>FICO score</th>
      <th>accepted</th>
      <th>log_DtIR</th>
      <th>001xx</th>
      <th>006xx</th>
      <th>007xx</th>
      <th>008xx</th>
      <th>009xx</th>
      <th>010xx</th>
      <th>...</th>
      <th>990xx</th>
      <th>991xx</th>
      <th>992xx</th>
      <th>993xx</th>
      <th>994xx</th>
      <th>995xx</th>
      <th>996xx</th>
      <th>997xx</th>
      <th>998xx</th>
      <th>999xx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.0</td>
      <td>722.0</td>
      <td>1</td>
      <td>3.145110</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.5</td>
      <td>687.0</td>
      <td>1</td>
      <td>2.397895</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>677.0</td>
      <td>1</td>
      <td>2.542726</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>682.0</td>
      <td>1</td>
      <td>2.917771</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>692.0</td>
      <td>1</td>
      <td>3.341898</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 971 columns</p>
</div>




```python
both_cats['accepted'].value_counts()
```




    0    719663
    1    708961
    Name: accepted, dtype: int64



The dataset is complete and we check to make sure there are no null values. 


```python
both_cats.isnull().sum()
```




    emp_length    0
    FICO score    0
    accepted      0
    log_DtIR      0
    001xx         0
    006xx         0
    007xx         0
    008xx         0
    009xx         0
    010xx         0
    011xx         0
    012xx         0
    013xx         0
    014xx         0
    015xx         0
    016xx         0
    017xx         0
    018xx         0
    019xx         0
    020xx         0
    021xx         0
    022xx         0
    023xx         0
    024xx         0
    025xx         0
    026xx         0
    027xx         0
    028xx         0
    029xx         0
    030xx         0
                 ..
    970xx         0
    971xx         0
    972xx         0
    973xx         0
    974xx         0
    975xx         0
    976xx         0
    977xx         0
    978xx         0
    979xx         0
    980xx         0
    981xx         0
    982xx         0
    983xx         0
    984xx         0
    985xx         0
    986xx         0
    987xx         0
    988xx         0
    989xx         0
    990xx         0
    991xx         0
    992xx         0
    993xx         0
    994xx         0
    995xx         0
    996xx         0
    997xx         0
    998xx         0
    999xx         0
    Length: 971, dtype: int64



Below are summary statistics of the numerical variables.


```python
columns = ['emp_length', 'FICO score', 'log_DtIR']
both_cats[columns].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emp_length</th>
      <th>FICO score</th>
      <th>log_DtIR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.428624e+06</td>
      <td>1.428624e+06</td>
      <td>1.428624e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.763940e+00</td>
      <td>6.454719e+02</td>
      <td>2.764179e+00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.850484e+00</td>
      <td>1.364309e+02</td>
      <td>1.073144e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000e-01</td>
      <td>6.430000e+02</td>
      <td>2.342936e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000e+00</td>
      <td>6.770000e+02</td>
      <td>2.948472e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.000000e+00</td>
      <td>7.020000e+02</td>
      <td>3.404223e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+01</td>
      <td>8.500000e+02</td>
      <td>1.772753e+01</td>
    </tr>
  </tbody>
</table>
</div>



None of the variables correlate strongly with each other:


```python
both_cats[['emp_length', 'FICO score', 'log_DtIR', 'accepted']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emp_length</th>
      <th>FICO score</th>
      <th>log_DtIR</th>
      <th>accepted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>emp_length</th>
      <td>1.000000</td>
      <td>0.276814</td>
      <td>0.075099</td>
      <td>0.578204</td>
    </tr>
    <tr>
      <th>FICO score</th>
      <td>0.276814</td>
      <td>1.000000</td>
      <td>0.416666</td>
      <td>0.385470</td>
    </tr>
    <tr>
      <th>log_DtIR</th>
      <td>0.075099</td>
      <td>0.416666</td>
      <td>1.000000</td>
      <td>0.137106</td>
    </tr>
    <tr>
      <th>accepted</th>
      <td>0.578204</td>
      <td>0.385470</td>
      <td>0.137106</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### The data is now ready!
