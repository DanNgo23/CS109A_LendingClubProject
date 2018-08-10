
# Conclusions

To predict default of a loan application, we used Lending Club’s accepted loan data for analysis. Because the loans are funded, we can know the performance of each loan and do supervised learning. We used 2012-2013 loan data, because these
loans are old enough (longer than 60 month) so we don’t have to drop a great number of loans that are still outstanding.

The raw data from Lending Club’s website contains a lot of empty columns. We did preliminary data cleaning by deleting those empty columns. We then do the following stage of data cleaning/engineering (the procedure yields 184,567 rows).

First, we try to convert text to numeric information, which is easier to model. We did the conversion below:

• Column `emp_length`: This is a categorical field representing applicant’s employment length. We converted it to numerical.

• Column `earliest_cr_line`: This is a date field representing the date of the applicant’s earliest credit line. We convert the date to number of months ago.

• Column `loan_status`: This is our response variable. It is a categorical field including values of 'Fully Paid', 'Charged off', 'Default' and 'Current'. We define a successful loan (not default) as ‘Fully Paid’ and gives this field a value 0 . We define ‘Charged off’ and ‘Default’ as default loan, and give it a value 1. We discard rows with ‘Current’ value in this field.

• Column `Sub_grade`: This is a categorical field representing application’s creditworthiness with values A1,A2,…A5, B1, B2,…B5, …… G1,G2,…G5, with A1 being the best grade. We assigned A1 = 1, A2 = 2, …,G5 = 35.

• Convert other categorical variables that are not inherently ordered into dummy variables. Columns converted to dummies are : purpose (debt consolidation, home improvement…), term (36month, 60month), home owner status, state of address, initial list status, and income source verified.

• Deleted ‘ad-hoc’ columns that are part of the payment status of the loan. Such as total payment received, total interest received, total principal received, late fees. Such information is not available at the time when investors make funding decision, thus should not be included in predictors.

We then go on to eliminate highly collinear features for they will make the model behave weirdly. In this stage, we wanted to eliminate variables that have a higher than 0.8 correlation coefficient to other variables. We achieved elimination of collinear variables by calculating the correlation matrix among variables.

Variables are included unless-

1 - variable is very highly correlated with others

2 - variable looks like it represents information collected after loan disbursement

3 - variable is categorical that cannot be usefully converted

The above procedure yields 106 features.

Our next step is missing value imputation. Most of the variables in our analysis have missing values. The columns `mths_since_last_delinq`, `mths_since_last_record`, `mo_sin_old_il_acct`, and `mths_since_recent_revol_delinq` are scarcely populated but important predictors, so we do not want to just discard these columns. Missing values in those columns mean ‘not applicable’ (e.g. The applicant does not have any delinquency, or does not have any installment account’). So we impute the missing values with a large number. We use 600 (months) for all the missing values in those columns to represent absence of those events. Other imputations are zero for event counts or 100 for percentage utilization.

After the above procedure, we have an all-numeric matrix with no missing values. Because the input variables differ a lot in scale, we first use *sklearn.preprocessing* to adjust values from different scale to a nominally common scale. After normalization, each column has a mean of 0 and standard deviation of 1.

The final step before modeling is splitting the data into train and test sets. We use *sklearn.train_test_split* to split the dataset into train dataset and test dataset, with test dataset being 20% the size of the entire dataset.

In modeling, we did a non-normalized logistic model for diagnostic analysis. Then we also did logistic regression with normalization and regularization, random forest, linear discriminant analysis (LDA), quadratic discriminant analysis (QDA), and Adaboost. We attempted to do K nearest neighbors (KNN) but could not fit the model within reasonable amount of time.

Our baseline model is a normalized logistic regression with regularization. We used 5 fold cross-validation to search for the regularization parameter C. The best C searched is 0.0001. The train and test accuracy are both about 84%. Most of the
coefficients are close to 0, meaning there are no features that have particularly high explanatory power on loan default. Under 5% significance level, we have only 37 significant features out of the whole 106 feature set.

We have only 11 features that have a coefficient absolute value greater than 0.05. Those features are loan amount(+), sub_grade(+) ,annual income(-) , debt to income ratio(+), inquiry in last_6mths(+), total current balance of all accounts(-), account open in past 24 months(-) , Total open to buy on bankcard(-) , small business (+), 60 months(+).

Higher interest rate, worse sub-grade, high debt-to-income ratio, more account open in past 24 months, and longer loan term (60 months) increase the probability of default, while higher income decreases the probability of default. The result is
intuitive: higher interest rate increases the burden of borrower and is also equivalent to weaker creditworthiness, both of which increase the risk of default. Worse sub-grade represents higher default risk. Higher annual income means higher ability for debt payment. Higher debt-to-income ratio and more accounts opening in past 24 months both signal the borrower’s aggressiveness in using debt, which increase the default risk. A shorter loan has lower risk. Other variables seem to have lower explanatory power on default. The most significant features are sub-grade (0.22), and term (0.19). Sub-grade is the aggregation of other information (higher sub-grade value means worse credit worthiness), and loan term affect default risk
through directly increasing the duration of the loan.

Using only the 37 significant features to fit the logistic regression, we get comparably high training and test accuracy of 84%.

Other models we fit are random forest, LDA, QDA, KNN (attempted), Adaboost, Neural Network (NN), and logistic model with 6 components. Random Forest (max depth = 5), LDA, Adaboost, Neural Network and 6 components logistic models all get
training and test accuracy about 84%. QDA gets training accuracy of 58% and test accuracy of about 57.8%. We attempted to fit a KNN model using 5 fold of cross-validation to search the value k, but could not finish the fitting within reasonable
amount of time (20+ hours). The search of nearest neighbors takes very long because of the big size and high dimensionality of our dataset.

Performance : Logistic = PCA Logistic = NN = Random Forest = Adaboost = LDA > QDA

Fitting time : KNN > Logistic = PCA Logistic = NN > Adaboost > QDA > LDA = Random Forest

Though the models appear to have high accuracies, they are not good at determining which applications will default. For example, the logistic model only successfully predicts 25 out of 6000 default cases. The issue is the default rate of the whole population is low (about 16%). By simply predicting 'not default' for all applications, the model achieves 84% accuracy.

Prior default rate = 16% (the whole population)
Average predicted default rate of the non-default class = 15%
Average predicted default rate of the default class = 21%

As we can see from the visualization of 2D principal component analysis (PCA), though the two components only explain less than 10% variance, gives us an idea that the default and non-default classes are not well separated.

Our analysis found the input variables have weak predictive power on the occurrence of default. It is note worthy that the dataset includes the loans that got funded. This means the samples in the whole population have quite homogeneous
profiles so they all got funded. Two samples may have identical feature values and both got funded, but one defaults and one does not.(Such as the overlapping areas in the 2D PCA visualization). We can see that the average predicted default rate of the default group is less than 21%. While the average predicted default rate of the non-default group is 15%, the mean default rate of the two classes are close.

Although none of the models is good at predicting default, the probability we get from logistic regression can be used to build our investment strategy. Our investment strategy is not based on predicting default (classification) but instead on using the predicted default probability to calculate the expected return of a loan investment.

#### Investment Strategy

**Our investment strategy is to invest in any loan that has an expected value that is higher than our cost of fund, and to fund a smaller amount to greater number of loans.** Detail and derivation of our strategy are showed below.

The average loss of a default loan is about 37% of the principle.
The average default happens in the 18th month of a 36-month loan.
The average default happens in the 26th month of a 60-month loan.

The expected value of a 36-month loan = (1 + monthly-int)^18 * (1-default-rate) + (1-37%) * default-rate

The expected value of a 60-month loan = (1 + monthly-int)^26 * (1-default-rate) + (1-37%) * default-rate

We will use our baseline logistic regression model to predict the default rate of each application. And our investment strategy will be to **invest in any loan that has an expected value that is higher than (1 + cost of funding)**

For example, if our cost of fund is 7% (our opportunity cost), for a 36-month, 14% loan with 22% of default rate, will we fund this loan? The answer is no.

The loan has an expected value of
[1+(14%/12)^18] * (1-22%) + (1-37%) * 22% = 110%

The expected duration of the loan is:
(1-22%) * 36 + 22% * 18 = 32 (months)

So our cost of funding is (1 + 7%/12)^32 = 120.45%

The loan’s expected value is lower than our cost of funding so we will not fund this loan.

The variables driving our decision will be the **term** of loan (so we know the expected duration of the loan), the **interest rate** (so we know how much interest we can expect to receive), and the **default rate predicted by the model**. One
external factor is the cost of fund. Each investor has a different cost of fund.

This strategy expects a higher portfolio return than the investor’s cost of fund.

In finance, diversification is the process of allocating capital in a way that reduces the exposure to any one particular asset or risk. To achieve diversification, it is better for an investor to fund smaller amounts to greater number loans. The
diversification strategy lowers the exposure to specific loans.
