# Credit-Default-Risk
I used the Home Credit Default Risk Dataset to predict whether an applicant will repay a loan, using a wide range of application, credit, and behavioral data. 


I started with a baseline model: The Decision Tree. I needed to tune the depth, minimum
samples, and regularization parameters of the Decision Tree. I decided to work with two
different baseline models to see how pre-pruning parameters affected the model’s overall
performance. The original baseline fit perfectly with the training data which 1 at first thought
was a good thing but it is actually not. Having a training accuracy of 1 means that there was a
high likelihood of overfitting, meaning that the model just memorized the training data, and it
lacks the ability to accurately predict outcomes on new data. For baseline2, I added ccp_alpha
which is a cost complexity pruning to control tree size as well as adding more minimum samples
to the nodes to split, as well as limiting the depth of the tree to 3. There was less overfitting as
the training accuracy was now .91 and the test accuracy was also .91 so it was improved.
Additionally, Chat GPT helped me plot the learning curve to show how training and validation
scores for this baseline model. The x-axis shows the tree depth, and the y-axis is the accuracy
which measures the performance of the model. Below is what the graph of the training accuracy
and the validation accuracy looks like for different values of depth.


Then I moved to building both a Gradient Boost model as well as an XG Boost model.
Boosting trains models sequentially so that each new model focuses on the errors of the previous
ones. Gradient Boosting minimizes the loss function by adding weak learners in the direction of
the negative gradient. XG Boost is an enhanced version of Gradient Boost that adds
improvements to the Gradient Boost model in terms of speed and regularization capabilities.
Chat GPT helped me build a Gradient Boost model that includes Randomized Search CV for
hyperparameter tuning of the model. It has 2-fold stratified cross-validation which splits our data
into 2 parts and ensures each fold preserves the class distributions. Originally, I tried running this
code on the entire training dataset and it took over 20 minutes to run. This turned into a big
problem for me, so chat GPT advised to take the first 40,000 samples of the data because the
original data was just too large. I am aware of that there are certain pitfalls to cutting up the data
into smaller sections and it risks missing the best parameters needed.


Using the best-found parameters from the Randomized Search CV, I was able to build the best
gradient boosted model for the subsection of the data I took. The data has high accuracy scores
and for high precision scores for Default(0) but has poor precision for Default(1) which means it
almost never predicts Default (which very well could be the case based on how rare Default
was). There is very poor recall for the minority class and this model has high bias (Underfitting
the data due to small subsection). This model struggles with the Default group and has very
weak precision. This model relies heavily on predictors like EXT_SOURCE_2 as well as
DAYS_BIRTH which is the age of the borrower (trends in younger borrowers’ default more than
older borrowers). Once again, I repeated this process using Randomized Search CV on my XG
Boost, but I learned from my mistakes made in the Gradient Boosted Model and balanced the
weights of the classes for XG Boost.
