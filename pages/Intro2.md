# Project Goal 2 – Lending Profitability

Our second project goal is to analyze profitability from the point of view of lenders on the platform, and
see if we can develop strategies that can maximize it. The factors that determine profitability are the
Probability of Default, the Loss Given Default, and the return that can be earned from interest. While it
could be possible to build a complete model of profitability that integrates all of these aspects, that
would be a very complicated model. Instead we have focused on only one key aspect in the models
that we have built – the Probability of Default.

The “accepted loan” dataset from the LendingClub website contains all loans that have been disbursed,
and records their loan status that indicates whether they have been successfully repaid. For the loans in
the “rejected loan” dataset, we cannot know whether or not they would have succeeded if they had
been accepted, so that data will not feature in this analysis. Many loans are still current and not yet
matured, but if we use the data from the 2012-13 table we find that almost all of the loans can be
classed as either success or failure. The dataset contains a large number of potential predictors (~145
columns), some of which require data cleaning in order to be used. There are also many missing values
that need to be dealt with, and categorical variable that may potentially be used after encoding as
dummies.

We have proceeded to run classification type models for predicting whether or not the loans will
default. We have tried many of the model type that have been covered in the class, such as Logistic
Regression, Linear and Quadratic Discriminant Analysis, Random Forest, and Neural Networks. We
evaluate the models by looking at the rate of correct predictions, and whether they can separate
defaults from non-defaults in a way that can help an investor select better loans to lend to.
Early in the project, we were looking into selecting specific variables to use based on conceptual
theories about what characteristics of loans would indicate higher default probability of default.
However, after exploring the data, it turned out that there did not seem to be any specific strong
relationships that could be garnered in that way. Thus, we shifted to using an approach where we feed in
all of the available variables into the models, and/or having the features chosen by the software based
on modeling results.
