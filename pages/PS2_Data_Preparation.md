
## Project Goal 2 - Data Preparation

#### Lending club data was downloaded from the [website](https://www.lendingclub.com/info/download-data.action). We downloaded the loan data, which are the applications that got funded, so we can do supervised learning, knowing which loans default in the end. 
#### For analysis, we use 2012-2013 data because the loans are old enough so we know the final performances. We then read the csv file into ipython for the next stage of data cleaning/engineering. The procedure yields 184,567 rows.


```python
import numpy as np
import pandas as pd

import os
os.chdir("C:\\Users\\stuar\\OneDrive\\Documents\\cs109a\\project")

raw_loanData = pd.read_csv("from website\\LoanStats3b.csv",header=1,skipfooter=2)

```

    C:\Users\stuar\Anaconda3\lib\site-packages\ipykernel_launcher.py:7: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support skipfooter; you can avoid this warning by specifying engine='python'.
      import sys


First, we try to convert text and to numeric information, which is easier to model. We did the conversion below:<br>
•	Column ‘emp_length’ : This is a categorical field representing applicant’s employment length. We converted it to numerical.<br>
•	Column ‘earliest_cr_line: This is a date field representing the date of the applicant’s earliest credit line. We convert the date to number of months ago.<br>
•	Column ‘loan_status’ : This is our response variable. It is a categorical field including values of ‘Fully Paid’, ‘Charged off’, Default’ and ‘Current’… We define a successful loan(not default)  as ‘Fully Paid’ and gives this field a value 0 . We define ‘Charged off’ and ‘Default’ as default loan, and give it a value 1. We discard rows with ‘Current’ value in this field.<br>
•	Column ‘Sub_grade’ : This is a categorical field representing application’s creditworthiness with values A1,A2,…A5, B1, B2,…B5, …… G1,G2,…G5, with A1 being the best grade. We assigned A1 = 1, A2 = 2, …,G5 = 35.<br>
•	Column ‘total_pymnt’ and ‘loan_amnt’: We engineering these two columns to make a new column ‘loss’ by doing the operation:
[loss] = max(1 ['total_pymnt']/['loan_amnt'],0)
The [loss] column represents the % of loss if the loan default.
This column is not a predictor but the a response variable for our analysis later. <br>
•	Convert other categorical variables that are not inherently ordered into dummy variables. Columns converted to dummies are : purpose(debt consolidation, home improvement…), term(36month, 60month), home owner status, state of address, and initial list status.<br>
•	Deleted ‘ad-hoc’ columns that are part of the payment status of the loan. Such as total payment received, total interest received, total principal received. The above information is not available at the time when investors make funding decision, thus should not be included in predictors. 



```python
from datetime import datetime

# clean up variables that hae unsuitable formats
raw_loanData['el2'] = pd.to_numeric(raw_loanData['emp_length'].str.slice(0,2),errors='coerce').fillna(0)
raw_loanData['earliest_cr_line2'] = datetime.today() -  pd.to_datetime(raw_loanData['earliest_cr_line'])
raw_loanData['int_rate'] = pd.to_numeric(raw_loanData['int_rate'].replace({'\%': ''}, regex=True))
raw_loanData['revol_util'] = pd.to_numeric(raw_loanData['revol_util'].replace({'\%': ''}, regex=True))

# Convert earliest credit line from date to number of months ago
raw_loanData['earliest_cr_line2'] = ((raw_loanData['earliest_cr_line2'] / np.timedelta64(1, 'M'))).astype(int)

#change 'sub_grade' column to numerical
grade_dic = {'A1':1, 'A2':2,'A3':3,'A4':4,'A5':5,\
             'B1':6, 'B2':7,'B3':8,'B4':9,'B5':10,\
             'C1':11, 'C2':12,'C3':13,'C4':14,'C5':15,\
             'D1':16, 'D2':17,'D3':18,'D4':19,'D5':20,\
             'E1':21, 'E2':22,'E3':23,'E4':24,'E5':25,\
             'F1':26, 'F2':27,'F3':28,'F4':29,'F5':30,\
             'G1':31, 'G2':32,'G3':33,'G4':34,'G5':35}
raw_loanData = raw_loanData.replace({"sub_grade": grade_dic})

#get_dummies
purpose_dummies = pd.get_dummies(raw_loanData['purpose'],drop_first=True)
term_dummies = pd.get_dummies(raw_loanData['term'],drop_first=True)
home_owner_dummies = pd.get_dummies(raw_loanData['home_ownership'],drop_first=True)
state_dummies = pd.get_dummies(raw_loanData['addr_state'],drop_first=True)
initial_status_dummies = pd.get_dummies(raw_loanData['initial_list_status'],drop_first=True)
verification_status_dummies = pd.get_dummies(raw_loanData['verification_status'],drop_first=True)
#if we also made dummies for zip_code (3-digit) there would be 839 */

# join raw data and dummies
dfs = [raw_loanData, purpose_dummies,term_dummies,home_owner_dummies,state_dummies,verification_status_dummies]
#model_data = loan_sf.join(purpose_dummies)
joined_df = pd.concat(dfs, axis=1)

# define success and failure based on the loan_status
loan_success = joined_df.loc[joined_df['loan_status'] == "Fully Paid"]
loan_failure = pd.concat([joined_df.loc[joined_df['loan_status'] == "Charged Off"]
                              ,joined_df.loc[joined_df['loan_status'] == "Default"]])
loan_success['default'] = 0
loan_failure['default'] = 1
loan_sf = pd.concat([loan_success,loan_failure]).reset_index()

print(raw_loanData.shape)
print(loan_success.shape)
print(loan_failure.shape)
print(loan_sf.shape)
```

    C:\Users\stuar\Anaconda3\lib\site-packages\ipykernel_launcher.py:34: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy


    (188181, 147)
    (155036, 215)
    (29531, 215)
    (184567, 216)


All provided variables are included unless affected by one of these:<br>
   1 - variable is blank<br>
   2 - variable is very highly correlated with others (you'll see I brought in the heatmap and correlation stuff from HW3)<br>
   3 - variable looks like it represents information collected after loan disbursement<br> 
   4 - variable is a categorical that can't be usefully converted<br>

## Missing value imputation

Most of the variables in our analysis have missing values. The columns `mths_since_last_delinq`, `mths_since_last_record`, `mo_sin_old_il_acct`, `mths_since_recent_revol_delinq` are scarcely populated. Missing values in those columns mean `not applicable` (e.g. The applicant does not have any delinquency, or does not have any installment account’). So we impute the missing values with a large number. We use 600 (months) for all the missing values in those columns. For other missing values in other columns, we use either zero or 100% (as a utilization percentage) based on similar reasoning, representing our best understanding of the data.



```python
# drop unwanted columns, either: 
# - already converted to dummy or numeric
# - categorical variables with values too diffuse for useful dummy sets
# - always blank
# - inherently dependent on other variables, or very highly correlated
# - represent information collected after loan disbursement, so not available for prediction at lending time
final_df = loan_sf.drop(['id','member_id','index','emp_length','home_ownership','term','issue_d','purpose','addr_state',
    'earliest_cr_line','initial_list_status','loan_status','verification_status','emp_title',
    'url','desc','title','pymnt_plan','zip_code','grade','last_pymnt_d','next_pymnt_d',
    'last_credit_pull_d','application_type','hardship_flag','hardship_type','hardship_reason',              
    'hardship_status','hardship_start_date','hardship_end_date','payment_plan_start_date',                      
    'hardship_loan_status','disbursement_method','debt_settlement_flag','debt_settlement_flag_date',                             
    'settlement_status','settlement_date','annual_inc_joint','dti_joint',
    'verification_status_joint','open_acc_6m','open_act_il','open_il_12m',
    'open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m',
    'open_rv_24m','max_bal_bc','all_util','inq_fi','total_cu_tl','inq_last_12m',
    'pct_tl_nvr_dlq','revol_bal_joint','sec_app_earliest_cr_line','sec_app_inq_last_6mths',
    'sec_app_earliest_cr_line','sec_app_inq_last_6mths','sec_app_mort_acc','sec_app_open_acc',
    'sec_app_revol_util','sec_app_open_act_il','sec_app_num_rev_accts',
    'sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med',
    'sec_app_mths_since_last_major_derog','deferral_term','hardship_amount','hardship_length',
    'hardship_dpd','orig_projected_additional_accrued_interest','hardship_payoff_balance_amount',
    'hardship_last_payment_amount','settlement_amount','settlement_percentage','settlement_term',
    'funded_amnt','funded_amnt_inv','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv',
    'total_rec_int','total_rec_late_fee','total_rec_prncp','policy_code','recoveries',
    'collection_recovery_fee','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl','installment','int_rate','tot_hi_cred_lim','num_actv_bc_tl','num_actv_rev_tl',
    'num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_sats','avg_cur_bal','total_bc_limit',
    'percent_bc_gt_75','last_pymnt_amnt'], axis = 1)

# Missing values in columns ['mths_since_last_delinq','mths_since_last_record','mo_sin_old_il_acct','mths_since_recent_revol_delinq'] 
# means not applicable, use 50*12 = 600(months) -> never happened
fill_with_600_list = [
    'mths_since_last_delinq','mths_since_last_record','mo_sin_old_il_acct','mths_since_recent_revol_delinq',
    'mths_since_last_major_derog','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq']

# missing values for details about loan history interpreted as the absence of such events
fill_with_0_list = [
    'tot_coll_amt','tot_cur_bal','total_rev_hi_lim','acc_open_past_24mths',
    'bc_open_to_buy','mort_acc','num_accts_ever_120_pd','num_rev_accts','num_rev_tl_bal_gt_0',
    'num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m',
    'total_bal_ex_mort','total_il_high_credit_limit']

fill_with_100_list = ['bc_util','revol_util']

final_df[fill_with_600_list] = final_df[fill_with_600_list].fillna(600)
final_df[fill_with_0_list] = final_df[fill_with_0_list].fillna(0)
final_df[fill_with_100_list] = final_df[fill_with_100_list].fillna(100)

#eliminate states with very few observations 
final_df = final_df.drop(['IA','ID','MS','NE'], axis = 1)

print(final_df.shape)
```

    (184567, 107)

