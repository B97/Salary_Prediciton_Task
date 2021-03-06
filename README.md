# Salary_Prediciton_Task

(Remark: if some outputs cells are not visible in the notebook, please refer to the google colab link: https://colab.research.google.com/github/B97/Salary_Prediciton_Task/blob/main/ML_model.ipynb)

Based on a survey data asking professionals about their salaries, we want to build a Supervised ML model that can predict a candidate's salary based on the following features: Gender, Age, Years of Experience, Country and Job Title.

Due to the tight deadline, we focused on getting the project done, by taking only the necessary steps. Below, we explain the approach, and we also provide ideas for further improvements, and how we could have handled the project had the deadline been reasonable.

The apporach we followed for that sake is the following:

#### 1- 
We proceeded to an exploratory data analysis, to understand better the features, the value ranges and the basic statistics

#### 2- 
We dropped useless columns for our modeling, and proceeded to convert categorical features to numerical. We also had to do some data cleaning, and consolidate some country names. Nonetheless, we don't claim to be exhaustive here, since due to tight deadline, we couldn't dive deep and do a in-depth cleaning. Had we had enough time, we should have investigated more the categorical features, to check the imbalance of certain categories and try to fix that by some data augmentation techniques.

#### 3- 
The Job Title, which is a required input for the ML model, was problematic, since it's a text data, and cannot be considered as categroical, and not easily converted to numerical by encoding, since the number of disctinct categories is very high (around 14k, and the total number of rows in our data set was 26k). Also, most of the job titles appear only a couple of times, which will makes them appear either in train or test, but unlikely in both. This will not help the model generalize for unseen job category. 

#### 4-
##### 4.1 
We decided on a first attempt to just drop the Job Title column, and proceed to build the regression ML, using GradientBoosting. We didn't have enough to work on the model selection and hyperparameter tuning. We just went with GBDT, since it's well-known that for that kind of dataset, it yields good results. We reported the RMSE for the test set, and we didn't have time to investigate other algorithms and other metrics, such as the R2.
Had we had enough time, we would have considered Stratified Cross validation, and Hyperpararamter tuning using GridSearch or Bayesian Approaches.


##### 4.2 
On a second time, and in order to answer to the question, and include Job Title as required, we decided to proceed with encoding the job titles with fixed length embedding, using pre-trained models, using NLTK library in Python. We didn't proceed to any in-depth analysis of the Job Titles, otherwise, we could have come up with other ideas of conversion to a new kind of feature, like doing a job classification... to consolidate the wide variety of Jobs appearing in the original dataset. Thererfore, we just relied on pre-trained model, to compute the average embedding vector of each Job Title after some basic text processing. We then treated the embeddings (whose dimension is 300) as feature inputs for a new GBDT model. Again, we just trained one model, without any fine-tuninig. We report the test RMSE, and we noticed that RMSE in this case drops by more than 75%.


### Further Path:

Once the model is trained, validated and tested on an unseen dataset, then we can move to deployment stage. Since we went with a Tree-based approach (with the GBDT), we can save our model on PMML file. We can then use containers (Docker, Kubernetes..) along with building a user-friendly interface, where customers of the interface, can have some fields to enter: Job Title, age, years of exp, Country. The model running in the backend will run the inference and returns the predicted salary of the candidate! 
