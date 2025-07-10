# agentops-example-project
This project comes with example Agent code to train, validate and deploy a regression model to predict NYC taxi fares.
If you're a data scientist just getting started with this repo for a brand new Agent project, we recommend 
adapting the provided example code to your Agent problem. Then making and 
testing Agent code changes on Databricks or your local machine.

The "Getting Started" docs can be found at https://docs.databricks.com/dev-tools/bundles/mlops-stacks.html.

## Table of contents
* [Code structure](#code-structure): structure of this project.

* [Iterating on Agent code](#iterating-on-Agent-code): making and testing Agent code changes on Databricks or your local machine.
* [Next steps](#next-steps)

This directory contains an Agent project based on the default
[Databricks MLOps Stacks](https://github.com/databricks/mlops-stacks),
defining a production-grade ML pipeline for automated retraining and batch inference of an ML model on tabular data.

## Code structure
This project contains the following components:

| Component                  | Description                                                                                                                                                                                                                                                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Agent Code                    | Example Agent project code, with unit tested Python modules and notebooks                                                                                                                                                                                                                                                                                  |
| Agent Resources as Code | Agent pipeline resources (data preparation and agent development jobs with schedules, etc) configured and deployed through [Databricks CLI bundles](https://docs.databricks.com/dev-tools/cli/bundle-cli.html)                                                                                              |

contained in the following files:

```
agentops-example-project        <- Root directory. Both monorepo and polyrepo are supported.
│
├── agent-example-project       <- Contains python code, notebooks and Agent resources related to one Agent project. 
│   │
│   ├── requirements.txt        <- Specifies Python dependencies for ML code (for example: model training, batch inference).
│   │
│   ├── databricks.yml          <- databricks.yml is the root bundle file for the ML project that can be loaded by Databricks CLI bundles. It defines the bundle name, workspace URL and resource config component to be included.
│   │
│   ├── data_preparation        <- Retrieves, stores, cleans, and vectorizes source data that is then ingested into a Vector Search index.
│   │   │
│   │   ├── data_ingestion                             <- Databricks Documentation scraping retrieval and storage.
│   │   │
│   │   ├── data_preprocessing                         <- Documentation cleansing and vectorization.
│   │   │
│   │   ├── vector_search                              <- Vector Search and index creation and ingestion.
│   │
│   │
│   ├── agent_development       <- Creates, registers, and evaluates the agent.
│   │   │
│   │   ├── agent                                      <- LangGraph Agent creation.
│   │   │
│   │   ├── agent_evaluation                           <- Databricks Agent llm-as-a-judge evaluation.
│   │
│   ├── agent_deployment        <- Deploys agent serving and contains a Databricks Apps front end interface.
│   │   │
│   │   ├── chat_interface_deployment                  <- Databricks App front end interface for end users.
│   │   │
│   │   ├── model_serving                              <- Model serving endpoint for the Agent.
│   │
│   │
│   ├── tests                   <- Tests for the Agent project.
│   │
│   ├── resources               <- Agent resource (Agent jobs, MLflow models) config definitions expressed as code, across dev/staging/prod/test.
│       │
│       ├── data-preparation-resource.yml              <- Agent resource config definition for data preparation and vectorization.
│       │
│       ├── agent-resource-workflow-resource.yml       <- Agent resource config definition for agent development, evaluation, and deployment.
│       │
│       ├── app-deployment-resource.yml                <- Agent resource config definition for launching the Databricks App frontend.
│       │
│       ├── ml-artifacts-resource.yml                  <- Agent resource config definition for model and experiment.
```


## Iterating on Agent code

### Deploy Agent code and resources to dev workspace using Bundles

Refer to [Local development and dev workspace](./resources/README.md#local-development-and-dev-workspace)
to use Databricks CLI bundles to deploy ML code together with ML resource configs to dev workspace.

This will allow you to develop locally and use Databricks CLI bundles to deploy to your dev workspace to test out code and config changes.

### Develop on Databricks using Databricks Repos

#### Prerequisites
You'll need:
* Access to run commands on a cluster running Databricks Runtime ML version 11.0 or above in your dev Databricks workspace
* To set up [Databricks Repos](https://docs.databricks.com/repos/index.html): see instructions below

#### Configuring Databricks Repos
To use Repos, [set up git integration](https://docs.databricks.com/repos/repos-setup.html) in your dev workspace.

If the current project has already been pushed to a hosted Git repo, follow the
[UI workflow](https://docs.databricks.com/repos/git-operations-with-repos.html#add-a-repo-connected-to-a-remote-repo)
to clone it into your dev workspace and iterate. 

Otherwise, e.g. if iterating on Agent code for a new project, follow the steps below:
* Follow the [UI workflow](https://docs.databricks.com/repos/git-operations-with-repos.html#add-a-repo-connected-to-a-remote-repo)
  for creating a repo, but uncheck the "Create repo by cloning a Git repository" checkbox.
* Install the `dbx` CLI via `pip install --upgrade dbx`
* Run `databricks configure --profile mlops-example-project-dev --token --host <your-dev-workspace-url>`, passing the URL of your dev workspace.
  This should prompt you to enter an API token
* [Create a personal access token](https://docs.databricks.com/dev-tools/auth/pat.html)
  in your dev workspace and paste it into the prompt from the previous step
* From within the root directory of the current project, use the [dbx sync](https://dbx.readthedocs.io/en/latest/guides/python/devloop/mixed/#using-dbx-sync-repo-for-local-to-repo-synchronization) tool to copy code files from your local machine into the Repo by running
  `dbx sync repo --profile mlops-example-project-dev --source . --dest-repo your-repo-name`, where `your-repo-name` should be the last segment of the full repo name (`/Repos/username/your-repo-name`)



## Next Steps

When you're satisfied with initial Agent experimentation (e.g. validated that a model with reasonable performance can be trained on your dataset) and ready to deploy production training/inference pipelines, ask your ops team to set up CI/CD for the current Agent project if they haven't already. CI/CD can be set up as part of the


For AI Agent Ops Stacks initialization, even if it was skipped in this case, or this project can be added to a repo setup with CI/CD already, following the directions under "Setting up CI/CD" in the repo root directory README.

To add CI/CD to this repo:
 1. Run `databricks bundle init mlops-stacks` via the Databricks CLI
 2. Select the option to only initialize `CICD_Only`
 3. Provide the root directory of this project and answer the subsequent prompts

More details can be found on the homepage [MLOps Stacks README](https://github.com/databricks/mlops-stacks/blob/main/README.md).
