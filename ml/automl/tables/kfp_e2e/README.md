# Tutorial: AutoML Tables: end-to-end workflow on Cloud AI Platform Pipelines


## Introduction

[AutoML Tables][1] lets you automatically build, analyze, and deploy state-of-the-art machine learning models using your own structured data. 

A number of new AutoML Tables features have been released recently.  These include: an improved Python [client library][2], the ability to obtain [explanations][3] for your online predictions ++and batch?++, the ability to [export your model and serve it in a container][4] anywhere, and the ability to view model search progress and final model hyperparameters [in Cloud Logging][5].

This tutorial gives a tour of some of these new features via a [Cloud AI Platform Pipelines][6] example, that shows end-to-end management of an AutoML Tables workflow. 

The example pipeline [creates a _dataset_][7], [imports][8] data into the dataset from a [BigQuery][9] _view_, and [trains][10] a custom model on that data. Then, it fetches [evaluation][11] and metrics information about the trained model, and based on specified criteria about model quality, uses that information to automatically determine whether to [deploy][12] the model for online prediction.   Once the model is deployed, you can make prediction requests, and optionally obtain prediction [explanations][13] as well as the prediction result.
In addition, the example shows how to scalably **_serve_** your ￼exported trained model￼ from your Cloud AI Platform Pipelines installation for prediction requests.

You can manage all the parts of this workflow from the [Tables UI][14] as well, or programmatically via a [notebook][15] or script.  But specifying this process as a workflow has some advantages: the workflow becomes reliable and repeatable, and Pipelines makes it easy to monitor the results and schedule recurring runs.
For example, if your dataset is updated regularly—say once a day— you could schedule a workflow to run daily, each day building a model that trains on an updated dataset.
(With a bit more work, you could also set up event-based triggering pipeline runs, for example [when new data is added][16] to a [Google Cloud Storage][17] bucket.)

### About the example dataset and scenario

The [Cloud Public Datasets Program][18] makes available public datasets that are useful for experimenting with machine learning. For our examples, we’ll use data that is essentially a join of two public datasets stored in [BigQuery][19]: [London Bike rentals][20] and [NOAA weather data][21], with some additional processing to clean up outliers and derive additional GIS and day-of-week fields.  Using this dataset, we’ll build a regression model to predict the _duration_ of a bike rental based on information about the start and end rental stations, the day of the week, the weather on that day, and other data. If we were running a bike rental company, we could use these predictions—and their [explanations][22]—to help us anticipate demand and even plan how to stock each location.
While we’re using bike and weather data here, you can use AutoML Tables for tasks as varied as asset valuations, fraud detection, credit risk analysis, customer retention prediction, analyzing item layouts in stores, and many more.

## Using Cloud AI Platform Pipelines or Kubeflow Pipelines to orchestrate a Tables workflow

You can run this example via a [Cloud AI Platform Pipelines][23] installation, or via [Kubeflow Pipelines][24] on a [Kubeflow on GKE][25] installation.  [Cloud AI Platform Pipelines][26] was recently launched in Beta.

Slightly different variants of the pipeline specification are required depending upon which you’re using. (It would be possible to run the example on other Kubeflow installations too, but that would require additional credentials setup not covered in this tutorial).

### Install a Cloud AI Platform Pipelines cluster

You can create an AI Platform Pipelines installation with a few clicks.  Access AI Platform Pipelines by visiting the [AI Platform Panel][27] in the [Cloud Console][28]. 

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/sA17BykJuzF.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/sA17BykJuzF.png" width="90%"/></a>
<figcaption><br/><i>Create a new Pipelines instance.</i></figcaption>
</figure>

See the [documentation][29] for more detail.

(You can also do this installation [from the command line][30] onto an existing GKE cluster if you prefer. If you do, for consistency with the UI installation, create the GKE cluster with `--scopes cloud-platform`).

### Or, install Kubeflow to use Kubeflow Pipelines

You can also run this example from a [Kubeflow][31] installation. For the example to work out of the box, you’ll need a Kubeflow on [GKE][32] installation, set up to use [IAP][33].  An easy way to do this is via the Kubeflow [‘click to deploy’ web app][34], or you can follow the command-line instructions [here][35].

### Upload and run the Tables end-to-end Pipeline

Once a Pipelines installation is running, we can upload the example AutoML Tables pipeline. 
Click on **Pipelines** in the left nav bar of the Pipelines Dashboard.  Click on **Upload Pipeline**. 

- For Cloud AI Platform Pipelines, upload [this file][36].  This archive points to the compiled version of [this pipeline][37], specified and compiled using the [Kubeflow Pipelines SDK][38].  
- For Kubeflow Pipelines on a Kubeflow installation, upload [this file][39].  This archive points to the compiled version of [this pipeline][40].

The difference between the two pipelines relates to how GCP authentication is handled.  For the Kubeflow pipeline, we’ve added `.apply(gcp.use_gcp_secret('user-gcp-sa'))` annotations to the pipeline steps. This tells the pipeline to use the mounted _secret_—set up during the installation process— that provides GCP account credentials. 
With the Cloud AI Platform Pipelines installation, the GKE cluster nodes have been set up to use the `cloud-platform` scope.

++does the sdk now support handling this? and it so does it apply to KF installations? probably not++

The uploaded pipeline graph will look similar to this:

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-17%20at%204.27.41%20PM.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-17%20at%204.27.41%20PM.png" width="40%"/></a>
<figcaption><br/><i>The uploaded Tables 'end-to-end' pipeline.</i></figcaption>
</figure>

Click the **+Create Run** button to run the pipeline.  You will need to fill in some pipeline parameters.
Specifically, replace `YOUR_PROJECT_HERE` with the name of your project; replace `YOUR_DATASET_NAME` with the name you want to give your new dataset (make it unique, and use letters, numbers and underscores up to 32 characters); and replace `YOUR_BUCKET_NAME` with the name of a [GCS bucket][41].  This bucket should be in the same _region_ as that specified by the `gcp_region` parameter. E.g., if you keep the default `us-central1` region, your bucket should also be a _regional_ (not multi-regional) bucket in the `us-central1` region. ++double check that this is necessary.++

 If you want to schedule a recurrent set of runs, you can do that instead.  If your data is in [BigQuery][42]— as is the case for this example pipeline— and has a temporal aspect, you could define a _view_ to reflect that, e.g. to return data from a window over the last `N` days or hours.  Then, the AutoML pipeline could specify ingestion of data from that view, grabbing an updated data window each time the pipeline is run, and building a new model based on that updated window.

## The steps executed by the pipeline

The example pipeline [creates a _dataset_][43], [imports][44] data into the dataset from a [BigQuery][45] _view_, and [trains][46] a custom model on that data. Then, it fetches [evaluation][47] and metrics information about the trained model, and based on specified criteria about model quality, uses that information to automatically determine whether to [deploy][48] the model for online prediction. We’ll take a closer look at each of the pipeline steps, and how they’re implemented.

### Create a Tables dataset and adjust its schema

This pipeline creates a new Tables _dataset_, and ingests data from a [BigQuery][49] table for the “bikes and weather” dataset described above. These actions are implemented by the first two steps in the pipeline  (the `automl-create-dataset-for-tables` and `automl-import-data-for-tables` steps).

While we’re not showing it in this example, AutoML Tables supports ingestion from BigQuery _views_ as well as tables.  This can be an easy way to do _feature engineering_: leverage BigQuery’s rich set of functions and operators to clean and transform your data before you ingest it.

When the data is ingested, AutoML Tables infers the _data type_ for each field (column).  In some cases, those inferred types may not be what you want.  For example, for our “bikes and weather” dataset, several ID fields (like the station IDs) are set by default to be numeric, but we want them treated as categorical when we train our model.  In addition, we want to treat the `loc_cross` strings as categorical rather than text.
We make these adjustments programmatically, by defining a pipeline parameter that specifies the schema changes.
Then, in the `automl-set-dataset-schema` pipeline step, for each schema adjustment , we call `update_column_spec`:

```python
client.update_column_spec(
          dataset_display_name=dataset_display_name,
          column_spec_display_name=column_spec_display_name,
          type_code=type_code,
          nullable=nullable
      )
```

Before we can train the model, we also need to specify the _target_ column— what we want our model to predict.  In this case, we’ll train the model to predict rental _duration_.  This is a numeric value, so we’ll train a [regression][50] model.

```python
client.set_target_column(
              dataset_display_name=dataset_display_name,
              column_spec_display_name=target_column_spec_name
          )
```

### Train a custom model on the dataset

Once the dataset is defined and its schema set properly, the pipeline will train the model.  This happens in the `automl-create-model-for-tables` pipeline step. Via pipeline parameters, we can specify the training budget, the _optimization objective_ (if de the default), and can additionally specify which columns to include or exclude from the model inputs.  

You may want to specify a non-default optimization objective depending upon the characteristics of your dataset.  [This table][51] describes the available optimization objectives and when you might want to use them.
For example, if you were training a classification model using an imbalanced dataset, you might want to specify use of AUC PR (`MAXIMIZE_AU_PRC`), which optimizes results for predictions for the less common class.

```python
client.create_model(
    model_display_name,
    train_budget_milli_node_hours=train_budget_milli_node_hours,
    dataset_display_name=dataset_display_name,
    optimization_objective=optimization_objective,
    include_column_spec_names=include_column_spec_names,
    exclude_column_spec_names=exclude_column_spec_names,
  )
```

### View model search information via Cloud Logging

You can view details about an AutoML Tables model [via Cloud Logging][52]. Using Logging, you can see the final model hyperparameters as well as the hyperparameters and object values used during model training and tuning.

An easy way to access these logs is to go to the [AutoML Tables page][53] in the Cloud Console.  Select the Models tab in the left navigation pane and click on the model you’re interested in.  Click the “Model” link to see the final hyperparameter logs.  To see the tuning trial hyperparameters, click the “Trials” link.

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-23%20at%202.20.46%20PM.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-23%20at%202.20.46%20PM.png" width="30%"/></a>
<figcaption><br/><i>View a model's search logs from its evaluation information.</i></figcaption>
</figure>

For example, here is a look at the Trials logs a custom model trained on the “bikes and weather” dataset, with one of the entries expanded in the logs:

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-23%20at%202.23.00%20PM.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-23%20at%202.23.00%20PM.png" width="90%"/></a>
<figcaption><br/><i>The 'Trials' logs for a "bikes and weather" model</i></figcaption>
</figure>



### Custom model evaluation 

Once your custom model has finished training, the pipeline moves on to its next step: model evaluation. We can access evaluation metrics via the API.  We’ll use this information to decide whether or not to deploy the model. 

These actions are factored into two steps. The process of fetching the evaluation information can be a general-purpose component (pipeline step) used in many situations; and then we’ll follow that with a more special-purpose step, that analyzes that information and uses it to decide whether or not to deploy the trained model.  

In the first of these pipeline steps— the `automl-eval-tables-model` step— we’ll retrieve the evaluation and _global feature importance_ information.

```python
model = client.get_model(model_display_name=model_display_name)
feat_list = [(column.feature_importance, column.column_display_name)
        for column in model.tables_model_metadata.tables_model_column_info]
evals = list(client.list_model_evaluations(model_display_name=model_display_name))

```

AutoML Tables automatically computes global feature importance for a trained model. This shows, across the evaluation set, the average absolute attribution each feature receives. Higher values mean the feature generally has greater influence on the model’s predictions.
This information is useful for debugging and improving your model. If a feature’s contribution is negligible—if it has a low value—you can simplify the model by excluding it from future training. 
The pipeline step renders the global feature importance data as part of the pipeline run’s output:

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-23%20at%201.22.42%20PM.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-23%20at%201.22.42%20PM.png" width="50%"/></a>
<figcaption><br/><i>Global feature importance for the model inputs, rendered by a Kubeflow Pipeline step.</i></figcaption>
</figure>


For our example, based on the graphic above, we might try training a model without including bike_id.

In the following pipeline step— the `automl-eval-metrics` step— the evaluation output from the previous step is grabbed as input, and parsed to extract metrics that we’ll use in conjunction with pipeline parameters to decide whether or not to deploy the model. One of the pipeline input parameters allows specification of metric thresholds. In this example, we’re training a regression model, and we’re specifying a `mean_absolute_error` (MAE) value as a threshold in the pipeline input parameters:
```python
{"mean_absolute_error": 450}
```

The pipeline step compares the model evaluation information to the given threshold constraints. In this case, if the MAE is \< `450`, the model will not be deployed. The pipeline step outputs that decision, and displays the evaluation information it’s using as part of the pipeline run’s output:

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-23%20at%202.07.21%20PM.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-23%20at%202.07.21%20PM.png" width="25%"/></a>
<figcaption><br/><i>Information about a model's evaluation, rendered by a Kubeflow Pipeline step.</i></figcaption>
</figure>


### (Conditional) model deployment

You can _deploy_ any of your custom Tables models to make them accessible for online prediction requests. 
The pipeline code uses a _conditional test_ to determine whether or not to run the step that deploys the model, based on the output of the evaluation step described above:

```python
with dsl.Condition(eval_metrics.outputs['deploy'] == True):
  deploy_model = deploy_model_op( ... )
```

Only if the model meets the given criteria, will the deployment step (called `automl-deploy-tables-model`) be run, and the model be deployed automatically as part of the pipeline run:
```python
response = client.deploy_model(model_display_name=model_display_name)
```

You can always deploy a model later if you like.

### Putting it together: The full pipeline execution

The figure below shows the result of a pipeline run.  In this case, the conditional step was executed— based on the model evaluation metrics— and the trained model was deployed.  
Via the UI, you can view outputs, logs for each step, run artifacts and lineage information, and more.  See [this post][54] for more detail.

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-17%20at%204.28.32%20PM.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/automl/tables_e2e/Screen%20Shot%202020-03-17%20at%204.28.32%20PM.png" width="40%"/></a>
<figcaption><br/><i>Execution of a pipeline run. You can view outputs, logs for each step, run artifacts and lineage information, and more.</i></figcaption>
</figure>

## Getting explanations about your model’s predictions

Once a model is deployed, you can request predictions from that model.  You can additionally request _explanations for local feature importance_: a score showing how much (and in which direction) each feature influenced the prediction for a single example.  See [this blog post][55] for more information on how those values are calculated.

Here is a [notebook example][56] of how to request a prediction and its explanation using the Python client libraries.

```python
from google.cloud import automl_v1beta1 as automl
client = automl.TablesClient(project=PROJECT_ID, region=REGION)
 
response = client.predict(
    model_display_name=model_display_name,
    inputs=inputs,
    feature_importance=True,
)
```

The prediction response will have a structure like [this][57]. (The notebook above shows how to visualize the local feature importance results using `matplotlib`.)

It’s easy to explore local feature importance through the Cloud Console’s [AutoML Tables UI ][58]as well. After you deploy a model, go to the **TEST & USE** tab of the Tables panel, select **ONLINE PREDICTION**, enter the field values for the prediction, and then check the **Generate feature importance** box at the bottom of the page. The result will show the feature importance values as well as the prediction.  This [blog post][59] gives some examples of how these explanations can be used to find potential issues with your data or help you better understand your problem domain.

## The AutoML Tables UI in the Cloud Console

With this example we’ve focused on how you can automate a Tables workflow using Kubeflow pipelines and the Python client libraries.

All of the pipeline steps can also be accomplished via the [AutoML Tables UI][60] in the Cloud Console, including many useful visualizations, and other functionality not implemented by this example pipeline— such as the ability to export the model’s test set and prediction results to BigQuery for further analysis.

## Export the trained model and serve it on a GKE cluster

Recently, Tables launched a feature to let you export your full custom model, packaged so that you can serve it via a Docker container. This lets you serve your models anywhere that you can run a container.  For example, [this blog post][61] walks through the steps to serve the exported model using [Cloud Run][62].  
+++[http://amygdala.github.io/automl/ml/2019/12/05/automl\_tables\_export.html][63]++

Similarly, you can serve your exported model from any [GKE][64] cluster, including the cluster created for an AI Platform Pipelines installation. Follow the instructions in the blog post above to create your container.  Then, you can create a Kubernetes _deployment_  and _service_ to serve your model, by instantiating [this template][65].
Once the service is deployed, you can send it prediction requests.  The sample’s [README][66] walks through this process in more detail. 

## A deeper dive into the pipeline code

See the sample’s [README][67] for a more detailed walkthrough of the pipeline code.  The updated [Python client library][68] makes it very straightforward to build the Pipelines components that support each stage of the workflow.

## Summary and what’s next

\<…\>

[1]:	https://cloud.google.com/automl-tables/docs/
[2]:	https://googleapis.dev/python/automl/latest/gapic/v1beta1/tables.html
[3]:	xxx
[4]:	http://amygdala.github.io/automl/ml/2019/12/05/automl_tables_export.html
[5]:	https://cloud.google.com/automl-tables/docs/logging
[6]:	%20https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-ai-platform-pipelines
[7]:	xxx
[8]:	xxx
[9]:	xxx
[10]:	xxx
[11]:	xxx
[12]:	xxx
[13]:	xxx
[14]:	xxx
[15]:	xxx
[16]:	xxx
[17]:	xxx
[18]:	https://cloud.google.com/bigquery/public-data/
[19]:	https://cloud.google.com/bigquery/
[20]:	https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=london_bicycles&page=dataset
[21]:	https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=noaa_gsod&page=dataset
[22]:	xxx
[23]:	https://cloud.google.com/ai-platform/pipelines/docs
[24]:	xxx
[25]:	xxx
[26]:	https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-ai-platform-pipelines
[27]:	https://console.cloud.google.com/ai-platform/pipelines/clusters
[28]:	https://console.cloud.google.com
[29]:	xxx
[30]:	https://github.com/kubeflow/pipelines/tree/master/manifests/gcp_marketplace
[31]:	https://www.kubeflow.org/
[32]:	https://cloud.google.com/kubernetes-engine
[33]:	https://cloud.google.com/iap
[34]:	https://deploy.kubeflow.cloud/#/deploy
[35]:	https://www.kubeflow.org/docs/gke/deploy/deploy-cli/
[36]:	xxx
[37]:	xxx
[38]:	xxx
[39]:	xxx
[40]:	xxx
[41]:	xxx
[42]:	xxx
[43]:	xxx
[44]:	xxx
[45]:	xxx
[46]:	xxx
[47]:	xxx
[48]:	xxx
[49]:	xxx
[50]:	xxx
[51]:	https://cloud.google.com/automl-tables/docs/train#opt-obj
[52]:	https://cloud.google.com/automl-tables/docs/logging
[53]:	https://console.cloud.google.com/automl-tables
[54]:	https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-ai-platform-pipelines
[55]:	https://cloud.google.com/blog/products/ai-machine-learning/explaining-model-predictions-structured-data
[56]:	https://github.com/amygdala/code-snippets/blob/master/ml/automl/tables/xai/automl_tables_xai.ipynb
[57]:	https://gist.github.com/amygdala/c96d45bdf694737d77d91597ca3ef1f0
[58]:	https://console.cloud.google.com/automl-tables
[59]:	https://cloud.google.com/blog/products/ai-machine-learning/explaining-model-predictions-structured-data
[60]:	https://console.cloud.google.com/automl-tables
[61]:	xxx
[62]:	xxx
[63]:	http://amygdala.github.io/automl/ml/2019/12/05/automl_tables_export.html
[64]:	xxx
[65]:	https://github.com/amygdala/code-snippets/blob/tables_e2e/ml/automl/tables/kfp_e2e/deploy_model_for_tables/model_serve_template.yaml
[66]:	xxx
[67]:	xxx
[68]:	https://googleapis.dev/python/automl/latest/gapic/v1beta1/tables.html