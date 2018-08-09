# CS109A Final Project Project: 
## LendingClub Investments

### By Dan Ngo (Group #7)

[sample link](pages/CSCIE3Project3a.html) | [sample link 2](pages/Graph+Theory-+Dijkstra%27s+Algorithm.md)


#### About LendingClub

LendingClub is the first peer-to-peer lending network to register with the Securities and Exchange Commission (SEC). The LendingClub platform connects borrowers to investors and facilitates the payment of loans ranging from $1,000 to $40,000. The details of how this marketplace works are available on [LendingClub’s website](https://www.lendingclub.com/). The basic idea is that borrowers apply for a loan on the LendingClub Marketplace. Investors can review the loan applications, along with risk analysis provided by LendingClub, to determine how much of the loan request they are willing to fund. When the loan is fully funded, the borrower will receive the loan. Investors can also choose to have loans selected manually or automatically via a specified strategy. Loan details such as grade or subgrade, loan purpose, interest rate, and borrower information can all be used to evaluate a loan for possible investment.

As part of this marketplace, LendingClub makes a significant amount of the loan data that is available to investors available online. Investors can make their own investment decisions and strategies based on the information provided by LendingClub.
As with other types of loan and investment platforms, Lending Club is the site of possible discrimination or unfair lending practices. Note that Lending Club is an "Equal Housing Lender" (logo displayed in the bottom of each page of the website) which means that ["the bank makes loans without regard to race, color, religion, national origin, sex, handicap, or familial status."](https://www.fdic.gov/regulations/laws/rules/2000-6000.html) Such policies in the past have failed to stop discrimination, most publicly against loan applicants of color, especially African American applicants. Racial discrimination has continued in new forms following the federal Fair Housing Act; investors have adapted their approach to fit within the guidelines of the law while maintaining results [with damaging and immoral effects on communities of color](https://www.revealnews.org/article/for-people-of-color-banks-are-shutting-the-door-to-homeownership/). Other systems do not explicitly discriminate against minorities but enable practices such as refusing loans from specific zip codes that in effect harm minority applicants.


#### Project Goal

With this in mind, we hope to address the issue of discrimination the best way we can, using data from accepted and rejected loan applications. Many applicants have their loan applications turned down. Selected applicants have their loan applications approved. Many, if not most, of those approved will have their loans adequately funded. We will devote our time and research more towards seeking the differences between those approved and those rejected, rather than analyzing the quality of investments in funding loans for approved applications. A LendingClub investor can receive all the advising strategies in the world, but there needs to be a level of sufficient cognizance about the patterns within approving or denying applicants. Additionally, investors connected through the LendingClub platform should always consider whether they should really participate in the facilitation of loans through the LendingClub platform- where loans are approved and many more are denied based on practices that may appear ostensibly altruistic yet seem hazily unethical and/or discriminatory. This is what we hope to explore.

Based on only the common data between the accepted and rejected loan applications datasets, can we classify whether a loan application will be rejected or not? Originally, we hoped to build a predictive model that incorporates text descriptions from the application, zip code information, FICO score, years of employment, and debt to income ratio, to estimate classification accuracy, sensitivity (true positive rate: how accurate are we at predicting the loans that ended upbeing approved), and specificity (true negative rate: how accurate we are at predicting the loans that ended being rejected). However, as you will see in our early exploration of the data, the data is limited, in the sense that there is plenty of data on applications where the loans were approved but that same type of data does not appear in the data on applications where the loans were rejected. All applications initially feature the same type of information when being completed and being submitted. Yet, there is some missing information in the dataset of accepted loans and much more missing information in the dataset of rejected loans. We will explain a potential rationale and the implications of this near the end. Nonetheless, we could only use data that was common between both datasets. We could not use text data in our model, something we originally hoped and planned to do. There are only 4 columns of data that were common: state of residence, the first 3 digits of an applicant's zip code (more on this near the end), FICO score, and years of employment. Debt to income ratio appears in the rejected loans dataset but not the accepted (one of the rare piece of information included in the rejected loans dataset but not the accepted). We were able to calculate this ourselves for the accepted dataset by dividing loan amount column by the income amount column, using the loan amount requested as a proxy for debt. We then will filter these 5 common columns from both datasets and add an additional binary column that states whether the loan application was rejected/accepted. This new column will be the response variable. Finally, we merge the two datasets by concatenating the rows. We're building a binary classifier, and will experiment with 7 different classification models: logistic regression, linear discriminant analysis, quadratic discriminant analysis, K-nearest neighbors, random forests, boosting, and stacking.

#### Data Resources

[Loan Data](https://github.com/Polkaguy/LendingClubData)

[Declined Loan Data](https://www.lendingclub.com/info/download-data.action)
