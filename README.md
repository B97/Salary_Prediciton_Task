# Salary_Prediciton_Task


Based on a survey data asking professionals about their salaries, we want to build a Supervised ML model that can predict a candidate's salary based on the following 4 features: Gender, Age, Years of Experience, Country and Job Title.

Due to the tight deadline, we focused on getting the project done, by taking the necessary steps. Below, we explain the approach, and provide as well ideas for further improvements, and how we could have handled the projects had the deadline being more reasonable.

The apporach we followed for that sake is the following:

#### 1- 
We proceeded to an exploratory data analysis, to understand better the features, the value ranges and the basic statistics

#### 2- 
We dropped useless columns for our modeling, and proceeded to convert categorical features to numerical. We also had to do some data cleaning, and consolidate some country names. Nonetheless, we don't claim to be exhaustive here, since due to tight deadline, we couldn't dive deep and do a in-depth cleaning. Had we had enough time, we should have investigated more the categorical features, to check the imbalance of certain categories and try to fix that by some data augmentation technique.

#### 3- 
The Job Title, which is a required input for the ML model, was problematic, since it's a text data, and cannot be considered as categroical, and concerted to numerical by encoding, since the number of disctinct categories was very high (around 14k, and the total number of rows in our data set was 26k). Also, most of the job titles appears only a couple of times, which will makes theese appear either in train or test, but never both. This will not help the model generalize for unseen job category. 

#### 4-
##### 4.1 
We decided on a first attempt to just drop the Job Title column, and proceed to build the regression ML, using GradientBoosting. We didn't have enough to work on the model selection and hyperparameter tuning. We just went with GBDT, since it's well-known that for that kind of dataset, it yields good results. We reported the RMSE for the test set, and we didn't have time to investigate other algorithms and other metrics, such as the R2.
Had we had enough time, we would've considered Stratified Cross validation, and Hyperpararamter tuning using GridSearch or Bayesian Approaches.


##### 4.2 
On a second time, and in order to answer to the question, and include Job Title as required, we decided to proceed with encoding the job titles with fixed length embedding, using pre-trained models, using NLTK library in Python. We didn't proceed to any in-depth analysis of the Job Titles, otherwise, we could have come up with other ideas of conversion to a new kind of feature, like doing a job classification... to consolidate the wide variety of Jobs appearing in the original dataset. Thererfore, we just relied on pre-trained model, to compute the average embedding vector of each Job Title after some basic text processing. We then treated the embeddings (whose dimension is 300) as feature inputs for a new GBDT model. Again, we just trained one model, without any fine-tuninig. We report the test RMSE, and we noticed that RMSE in this case drops by more than 75%.


