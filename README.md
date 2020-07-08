# Churn Prediction Project Prototype
This project is a Cloudera Machine Learning 
([CML](https://www.cloudera.com/products/machine-learning.html)) **Applied Machine Learning 
Project Prototype**. It has all the code and data needed to deploy an end-to-end machine 
learning project in a running CML instance.

## Project Overview
This project builds the telco churn with model interpretability project discussed in more 
detail [this blog post](https://blog.cloudera.com/visual-model-interpretability-for-telco-churn-in-cloudera-data-science-workbench/). 
The initial idea and code comes from the FFL Interpretability report which is now freely 
available and you can read the full report [here](https://ff06-2020.fastforwardlabs.com/)

![table_view](images/table_view.png)

The goal is to build a classifier model using Logistic Regression to predict the churn 
probability for a group of customers from a telecoms company. On top that, the model 
can then be interpreted using [LIME](https://github.com/marcotcr/lime). Both the Logistic 
Regression and LIME models are then deployed using CML's real-time model deployment 
capability and finally a basic flask based web application is deployed that will let 
you interact with the real-time model to see which factors in the data have the most 
influence on the churn probability.

By following the notebooks in this project, you will understand how to perform similar 
classification tasks on CML as well as how to use the platform's major features to your 
advantage. These features include **streamlined model experimentation**, 
**point-and-click model deployment**, and **ML app hosting**.

We will focus our attention on working within CML, using all it has to offer, while
glossing over the details that are simply standard data science.
We trust that you are familiar with typical data science workflows
and do not need detailed explanations of the code.
Notes that are *specific to CML* will be emphasized in **block quotes**.

## CDSW Inteface

Before start create the project and run the labs, let's explore CDSW interface:

![ml_create_project_1](images/cdsw_interface_1.png)

You have on the left hand panel:  

* **Projects** - where you create data science projects  
* **Jobs** - Run and schedule jobs and add dependencies  
* **Sessions** - Python, Scala or R sessions  
* **Experiments** - batch experiments  
* **Models** - build, deploy, and manage models as REST APIs to serve predictions  
* **Applications** - deploy long-running applications  
* **Settings** - User, Hadoop Authentication, SSH Keys and permission settings   

## Create the Project
In the main CDSW main page, click `New Project`:

![ml_create_project_1](images/ml_create_project_1.png)

```
Pooject Name:           Telco Churn
Project Visibility:     Private
Initial Setup -> Git:   https://github.com/campossalex/CMLTelcoChurn
```

Click `Create Project`.  

You will see something like this:  

![telco_churn_project_overview](images/telco_churn_project_overview.png)

Each project has following structure:

* **Models** - build, deploy, and manage models as REST APIs to serve predictions  
* **Jobs** - Run and schedule jobs and add dependencies  
* **Files** - Project files  

Now it is all set to start labs, but before, let's understand how to open a Workbench session.

From the project main page, click _Open Workbench_ button to.

![telco_churn_project_overview](images/launch_workbench_session_1.png)

Then you can select the options to open a new workbench session:

![telco_churn_project_overview](images/launch_workbench_session_2.png)

1. On the far left is a file browser (note the little ‘refresh’ icon at the top:  )
2. In the middle is an editor open for files.
3. On the right is a Session Start tile. You can select following options:  
    - **Editor**: use a native _Workbench_ or _Jupyter_ editor.
    - **Engine Kernel**: you can select Python 2, Python 3, Scala or R.
    - Engine Profile: memory and cpu configuration for the session. This environments 
  is configured with three profiles: 1vCPU/2GiB, 1vCPU/4GiB and 2vCPU/16GiB.  
4. By now, select a _Workbench_ editor, _Python 3_ engine and _1 vCPU/2GiB_ profile.
5. Click _Launch Session_ button.
6.  This will startup a Python engine and the right hand side will become two tiles. 
The top one is an output tile and the bottom one (with a red, then green left hand border) is a shell input window.

    ![telco_churn_project_overview](images/run_file_1.png)
 
7. From the left panel, select `0_bootstrap.py` file. This will open the file in the middle panel.

    ![telco_churn_project_overview](images/run_file_2.png)

8. You have two options to execute code. One is select the lines from the file and click from the top menu
_Run -> Run Line(s)_, or you can run all the lines by click _Run -> Run All_. In this case, run all the lines.
 
    ![telco_churn_project_overview](images/run_file_3.png)

9. The cursor on the left should turn to red and, after a few seconds, you should see output like this:

    ![telco_churn_project_overview](images/run_file_4.png)

10. What’s happening here is that the code in the file is being executed in the console (did you notice the left hand edge turned red?) and now you can see the output in the right hand screen. 
11. Stop this session. The Stop button is either on the top menu bar (when there’s sufficient room for it):

    ![telco_churn_project_overview](images/run_file_5.png)

12. Or it’s in the the Session drop down (when there’s little room for buttons):

    ![telco_churn_project_overview](images/run_file_6.png)
    
13. You can go back to the main project page by clicking _Project_ button located at the top menu.

What we have done executing `0_bootstrap.py` is uploading the data used in the project to `$STORAGE/datalake/data/churn/`. 
The original file comes as part of this git repo in the `raw` folder

## Labs
Go through each of the steps manually to build and understand how the project 
works, follow the steps below. There is a lot more detail and explanation/comments in each 
of the files/notebooks so its worth looking into those. Follow the steps below and you 
will end up with a running application.

### 1 Ingest Data
This script will read in the data csv from the file uploaded to the s3 bucket setup 
during the bootstrap and create a managed table in Hive. This is all done using Spark.

Open the file  `1_data_ingest.py` in a Workbench session: Python 3, 1vCPU/2GiB. Run all the lines.

### 2 Explore Data
This is a Jupyter Notebook that does some basic data exploration and visualization. It 
is to show how this would be part of the data science workflow.

![data](https://raw.githubusercontent.com/fletchjeff/cml_churn_demo_mlops/master/images/data.png)

Open a Jupyter Notebook session (rather than a Workbench): Python 3, 1vCPU/2GiB and 
open the `2_data_exploration.ipynb` file. 

At the top of the page click **Cells > Run All**.

### 3 Model Building
This is also a Jupyter Notebook to show the process of selecting and building the model 
to predict churn. It also shows more details on how the LIME model is created and a bit 
more on what LIME is actually doing.

Open a Jupyter Notebook session (rather than a work bench): python3, 1 CPU, 2 GB and 
open the `	3_model_building.ipynb` file. 

At the top of the page click **Cells > Run All**.

### 4 Model Training
A model pre-trained is saved with the repo has been and placed in the `models` directory. 
If you want to retrain the model, open the `4_train_models.py` file in a workbench  session: 
python3 1 vCPU, 2 GiB and run the file. The newly model will be saved in the models directory 
named `telco_linear`. 

There are 2 other ways of running the model training process

***1. Jobs***

The **[Jobs](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html)**
feature allows for adhoc, recurring and depend jobs to run specific scripts. To run this model 
training process as a job, create a new job by going to the Project window and clicking _Jobs >
New Job_ and entering the following settings:
* **Name** : Train Mdoel
* **Script** : 4_train_models.py
* **Arguments** : _Leave blank_
* **Kernel** : Python 3
* **Schedule** : Manual
* **Engine Profile** : 1 vCPU / 2 GiB
The rest can be left as is. Once the job has been created, click **Run** to start a manual 
run for that job.

***2. Experiments***

The other option is running an **[Experiment](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-running-an-experiment.html)**. Experiments run immediately and are used for testing different parameters in a model training process. In this instance it would be use for hyperparameter optimisation. To run an experiment, from the Project window click Experiments > Run Experiment with the following settings.
* **Script** : 4_train_models.py
* **Arguments** : 5 lbfgs 100 _(these the cv, solver and max_iter parameters to be passed to 
LogisticRegressionCV() function)
* **Kernel** : Python 3
* **Engine Profile** : 1 vCPU / 2 GiB

Click **Start Run** and the expriment will be sheduled to build and run. Once the Run is 
completed you can view the outputs that are tracked with the experiment using the 
`cdsw.track_metrics` function. It's worth reading through the code to get a sense of what 
all is going on.


### 5 Serve Model
The **[Models](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-creating-and-deploying-a-model.html)** 
is used top deploy a machine learning model into production for real-time prediction. To 
deploy the model trailed in the previous step, from  to the Project page, click **Models > New
Model** and create a new model with the following details:

* **Name**: Explainer
* **Description**: Explain customer churn prediction
* **File**: 5_model_serve_explainer.py
* **Function**: explain
* **Input**: 
```
{
	"StreamingTV": "No",
	"MonthlyCharges": 70.35,
	"PhoneService": "No",
	"PaperlessBilling": "No",
	"Partner": "No",
	"OnlineBackup": "No",
	"gender": "Female",
	"Contract": "Month-to-month",
	"TotalCharges": 1397.475,
	"StreamingMovies": "No",
	"DeviceProtection": "No",
	"PaymentMethod": "Bank transfer (automatic)",
	"tenure": 29,
	"Dependents": "No",
	"OnlineSecurity": "No",
	"MultipleLines": "No",
	"InternetService": "DSL",
	"SeniorCitizen": "No",
	"TechSupport": "No"
}
```
* **Kernel**: Python 3
* **Engine Profile**: 1vCPU / 2 GiB Memory

Leave the rest unchanged. Click **Deploy Model** and the model will go through the build 
process and deploy a REST endpoint. Once the model is deployed, you can test it is working 
from the model Model Overview page.

_**Note: This is important**_

Once the model is deployed, you must disable the additional model authentication feature. In the model settings page, untick **Enable Authentication**.

![disable_auth](images/disable_auth.png)

### 6 Deploy Application
The next step is to deploy the Flask application. The **[Applications](https://docs.cloudera.com/machine-learning/cloud/applications/topics/ml-applications.html)** feature is still quite new for CML. For this project it is used to deploy a web based application that interacts with the underlying model created in the previous step.

_**Note: This next step is important**_

_In the deployed model from step 5, go to **Model > Settings** and make a note (i.e. copy) the 
"Access Key". It will look something like this (ie. mukd9sit7tacnfq2phhn3whc4unq1f38)_

_From the Project level click on "Open Workbench" (note you don't actually have to Launch a 
session) in order to edit a file. Select the flask/single_view.html file and paste the Access 
Key in at line 19._

`        const accessKey = "mp3ebluylxh4yn5h9xurh1r0430y76ca";`

_Save the file (if it has not auto saved already) and go back to the Project._

From the Go to the **Applications** section and select "New Application" with the following:
* **Name**: Churn Analysis App
* **Subdomain**: churn-app _(note: this needs to be unique, so if you've done this before, 
pick a more random subdomain name)_
* **Script**: 6_application.py
* **Kernel**: Python 3
* **Engine Profile**: 1vCPU / 2 GiB Memory


After the Application deploys, click on the blue-arrow next to the name. The initial view is a 
table of randomly selected from the dataset. This shows a global view of which features are 
most important for the predictor model. The reds show incresed importance for preditcting a 
cusomter that will churn and the blues for for customers that will not.

![table_view](images/table_view.png)

Clicking on any single row will show a "local" interpreted model for that particular data point 
instance. Here you can see how adjusting any one of the features will change the instance's 
churn prediction.


![single_view_1](images/single_view_1.png)

Changing the InternetService to DSL lowers the probablity of churn. *Note: this does not mean 
that changing the Internet Service to DSL cause the probability to go down, this is just what 
the model would predict for a customer with those data points*


![single_view_2](images/single_view_2.png)
