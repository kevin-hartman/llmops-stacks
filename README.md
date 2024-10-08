# Databricks LLMOps Stacks

> **_NOTE:_**  This feature is in [private preview](https://docs.databricks.com/release-notes/release-types.html).

This repo provides a customizable stack for starting new GenAI projects
on Databricks that follow production best-practices out of the box.

One of the largest current gaps in an organization’s ability to utilize the advantages of LLMs is the infrastructure and processes around how LLMs are deployed and managed. We hear overwhelming feedback from Data Scientists and stakeholders alike about the slow time to productionize, uncertainty in best practices, and a lack of transparency in the vision of the final product.

We have a plan to address these concerns by providing a valuable template that demystifies the LLMOps process.

## Vision

We want to make an open source Databricks Asset Bundle (DAB) that deploys an easily understandable and readily adaptable end-to-end LLMOps pipeline. 

It should be turnkey and immediately deployable in any Databricks workspace. It should include a working example of a RAG process and app, and surface a user interface automatically for immediate use by users with minimal development overhead. 

The pipeline should conform to development, management, and governance best practices out of the box.

## Key Components

Corpus ingestion, cleaning, vectorization, and storage in Mosaic AI Vector Search
RAG Pipeline for returning results from the Mosaic AI Vector Search
RAG Automated Evaluation using both mlflow.evaluate and llm-as-a-judge
Prod model Champion/Challenger comparison and analysis
RAG Agent Serving using Databricks Model Serving Endpoints
RAG Pipeline integration testing
RAG Pipeline validation testing
RAG Pipeline Unit Testing
Chat Interface that is readily deployable for immediate use by stakeholders
RAG Performance testing with results stored in a delta table
RAG Model Drift testing with results stored in a delta table
RAG Human Evaluation feedback mechanism with results stored in a delta table
Model Monitoring Dashboard surfacing the testing and evaluation results
CI/CD process for promotion through dev/staging/pro
DAB project adaptation
Important Items to Keep in Mind

All pertinent artifacts and metrics must be logged in MLFlow.
Unity Catalog must be used for all relevant components of the pipeline.
Everything must be developed for serverless compute and workloads unless we find that serverless is incompatible with certain processes.
All artifacts will be used to create a Databricks Asset Bundle and must be developed with that deployment paradigm in mind.
The Front-end GUI must include the option for user feedback and store the results of the user testing.
The deployed example must be easy to follow and adapt for business use cases.

## Future State

Once the initial bundle is made available to all users, we can continue to flesh out the bundle with more advanced use cases:

Instruction Fine-tuning (Prompt Engineering) examples using Mosaic AI Model Training
Model Pre-training (Fine-tuning) using Mosaic AI Model Training
Advanced Chaining
Synthetic data generation
Advanced Prompt Engineering examples

## Inspiration

Here are some samples for inspiration, some of which might be a bit out of date. These are meant to serve as directional guidance but not necessarily line-for-line templates for our ultimate solution:

[Databricks Gen AI Cookbook](https://ai-cookbook.io/index-2.html#)
[Databricks Gen AI Product Documentation](https://docs.databricks.com/en/generative-ai/generative-ai.html)
[DBDemos Rag Chatbot Example](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot)
[MLFlow Tracing for Agents](https://docs.databricks.com/en/mlflow/mlflow-tracing.html#enable-inference-tables-to-collect-traces)
[MLOps Stacks](https://docs.databricks.com/en/machine-learning/mlops/mlops-stacks.html)

## Main Components
Using Databricks LLMOps Stacks, AI engineers can quickly get started iterating on ML code for new projects while ops engineers set up CI/CD and ML resources
management, with an easy transition to production. You can also use LLMOps Stacks as a building block in automation for creating new data science projects with production-grade CI/CD pre-configured. More information can be found at https://docs.databricks.com/en/dev-tools/bundles/LLMOps-stacks.html.

The default stack in this repo includes three modular components: 

| Component                   | Description                                                                                                                                                           | Why it's useful                                                                                                                                                                         |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ML Code](template/{{.input_root_dir}}/{{template%20`project_name_alphanumeric_underscore`%20.}}/)                     | Example ML project structure ([training](template/{{.input_root_dir}}/{{template%20`project_name_alphanumeric_underscore`%20.}}/training) and [batch inference](template/{{.input_root_dir}}/{{template%20`project_name_alphanumeric_underscore`%20.}}/deployment/batch_inference), etc), with unit tested Python modules and notebooks                                                                                           | Quickly iterate on ML problems, without worrying about refactoring your code into tested modules for productionization later on.                                                        |
| [ML Resources as Code](template/{{.input_root_dir}}/{{template%20`project_name_alphanumeric_underscore`%20.}}/resources) | ML pipeline resources ([training](template/{{.input_root_dir}}/{{template%20`project_name_alphanumeric_underscore`%20.}}/resources/model-workflow-resource.yml.tmpl) and [batch inference](template/{{.input_root_dir}}/{{template%20`project_name_alphanumeric_underscore`%20.}}/resources/batch-inference-workflow-resource.yml.tmpl) jobs, etc) defined through [databricks CLI bundles](https://docs.databricks.com/dev-tools/cli/bundle-cli.html)    | Govern, audit, and deploy changes to your ML resources (e.g. "use a larger instance type for automated model retraining") through pull requests, rather than adhoc changes made via UI. |
| CI/CD([GitHub Actions](template/{{.input_root_dir}}/.github/) or [Azure DevOps](template/{{.input_root_dir}}/.azure/))                       | [GitHub Actions](https://docs.github.com/en/actions) or [Azure DevOps](https://azure.microsoft.com/en-us/products/devops) workflows to test and deploy ML code and resources | Ship ML code faster and with confidence: ensure all production changes are performed through automation and that only tested code is deployed to prod                                   |

See the [FAQ](#FAQ) for questions on common use cases.

## ML pipeline structure and development loops

An ML solution comprises data, code, and models. These resources need to be developed, validated (staging), and deployed (production). In this repository, we use the notion of dev, staging, and prod to represent the execution
environments of each stage. 

An instantiated project from LLMOps Stacks contains an ML pipeline with CI/CD workflows to test and deploy automated model training and batch inference jobs across your dev, staging, and prod Databricks workspaces. 

<img src="https://github.com/databricks/mlops-stacks/blob/main/doc-images/mlops-stack-summary.png">
<img src="https://docs.databricks.com/en/_images/llmops-rag-3p.png">

Data scientists can iterate on ML code and file pull requests (PRs). This will trigger unit tests and integration tests in an isolated staging Databricks workspace. Model training and batch inference jobs in staging will immediately update to run the latest code when a PR is merged into main. After merging a PR into main, you can cut a new release branch as part of your regularly scheduled release process to promote ML code changes to production.

### Develop ML pipelines
https://github.com/databricks/LLMOps-stacks/assets/87999496/00eed790-70f4-4428-9f18-71771051f92a


### Create a PR and CI
https://github.com/databricks/LLMOps-stacks/assets/87999496/f5b3c82d-77a5-4ee5-85f5-8f00b026ae05


### Merge the PR and deploy to Staging
https://github.com/databricks/LLMOps-stacks/assets/87999496/7239e4d0-2327-4d30-91cc-5e7f8328ef73

https://github.com/databricks/LLMOps-stacks/assets/87999496/013c0d32-c283-494b-8c3f-2a9a60366207


### Deploy to Prod
https://github.com/databricks/LLMOps-stacks/assets/87999496/0d220d55-465e-4a69-bd83-1e66ad2e8464


[See this page](Pipeline.md) for detailed description and diagrams of the ML pipeline structure defined in the default stack. 

## LLMOps changes to MLOps production architecture
This section highlights the major changes to the MLOps reference architecture for LLMOps applications.

### Model hub
LLM applications often use existing, pretrained models selected from an internal or external model hub. The model can be used as-is or fine-tuned.

Databricks includes a selection of high-quality, pre-trained foundation models in Unity Catalog and in Databricks Marketplace. You can use these pre-trained models to access state-of-the-art AI capabilities, saving you the time and expense of building your own custom models. For details, see [Pre-trained models in Unity Catalog and Marketplace](https://docs.databricks.com/en/generative-ai/pretrained-models.html). 

### Vector database
Some LLM applications use vector databases for fast similarity searches, for example to provide context or domain knowledge in LLM queries. Databricks provides an integrated vector search functionality that lets you use any Delta table in Unity Catalog as a vector database. The vector search index automatically syncs with the Delta table. For details, see [Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html).

You can create a model artifact that encapsulates the logic to retrieve information from a vector database and provides the returned data as context to the LLM. You can then log the model using the MLflow LangChain or PyFunc model flavor.

### Fine-tune LLM
Because LLM models are expensive and time-consuming to create from scratch, LLM applications often fine-tune an existing model to improve its performance in a particular scenario. In the reference architecture, fine-tuning and model deployment are represented as distinct Databricks Jobs. Validating a fine-tuned model before deploying is often a manual process.

Databricks provides Mosaic AI Model Training, which lets you use your own data to customize an existing LLM to optimize its performance for your specific application. For details, see [Mosaic AI Model Training](https://docs.databricks.com/en/large-language-models/foundation-model-training/index.html).

### Model serving
In the RAG using a third-party API scenario, an important architectural change is that the LLM pipeline makes external API calls, from the Model Serving endpoint to internal or third-party LLM APIs. This adds complexity, potential latency, and additional credential management.

Databricks provides Mosaic AI Model Serving, which provides a unified interface to deploy, govern, and query AI models. For details, see [Mosaic AI Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html).

### Human feedback in monitoring and evaluation
Human feedback loops are essential in most LLM applications. Human feedback should be managed like other data, ideally incorporated into monitoring based on near real-time streaming.

The Mosaic AI Agent Framework review app helps you gather feedback from human reviewers. For details, see [Get feedback about the quality of an agentic application](https://docs.databricks.com/en/generative-ai/agent-evaluation/human-evaluation.html).


## Using LLMOps Stacks

### Prerequisites
 - Python 3.8+
 - [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/databricks-cli.html) >= v0.221.0

[Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/databricks-cli.html) contains [Databricks asset bundle templates](https://docs.databricks.com/en/dev-tools/bundles/templates.html) for the purpose of project creation.

Please follow [the instruction](https://docs.databricks.com/en/dev-tools/cli/databricks-cli-ref.html#install-the-cli) to install and set up databricks CLI. Releases of databricks CLI can be found in the [releases section](https://github.com/databricks/cli/releases) of databricks/cli repository.

[Databricks asset bundles](https://docs.databricks.com/en/dev-tools/bundles/index.html) and [Databricks asset bundle templates](https://docs.databricks.com/en/dev-tools/bundles/templates.html) are in public preview.


### Start a new project

To create a new project, run:

    databricks bundle init LLMOps-stacks

This will prompt for parameters for initialization. Some of these parameters are required to get started:
 * ``input_setup_cicd_and_project`` : If both CI/CD and the project should be set up, or only one of them. 
   * ``CICD_and_Project`` - Setup both CI/CD and project, the default option.
   * ``Project_Only`` - Setup project only, easiest for Data Scientists to get started with.
   * ``CICD_Only`` - Setup CI/CD only, likely for monorepo setups or setting up CI/CD on an already initialized project.
   We expect Data Scientists to specify ``Project_Only`` to get 
   started in a development capacity, and when ready to move the project to Staging/Production, CI/CD can be set up. We expect that step to be done by Machine Learning Engineers (MLEs) who can specify ``CICD_Only`` during initialization and use the provided workflow to setup CI/CD for one or more projects.
 * ``input_root_dir``: name of the root directory. When initializing with ``CICD_and_Project``, this field will automatically be set to ``input_project_name``.
 * ``input_cloud``: Cloud provider you use with Databricks (AWS, Azure, or GCP).

Others must be correctly specified for CI/CD to work:
 * ``input_cicd_platform`` : CI/CD platform of choice (GitHub Actions or GitHub Actions for GitHub Enterprise Servers or Azure DevOps)
 * ``input_databricks_staging_workspace_host``: URL of staging Databricks workspace, used to run CI tests on PRs and preview config changes before they're deployed to production.
   We encourage granting data scientists working on the current ML project non-admin (read) access to this workspace,
   to enable them to view and debug CI test results
 * ``input_databricks_prod_workspace_host``: URL of production Databricks workspace. We encourage granting data scientists working on the current ML project non-admin (read) access to this workspace,
   to enable them to view production job status and see job logs to debug failures.
 * ``input_default_branch``: Name of the default branch, where the prod and staging ML resources are deployed from and the latest ML code is staged.
 * ``input_release_branch``: Name of the release branch. The production jobs (model training, batch inference) defined in this
    repo pull ML code from this branch.

Or used for project initialization:
 * ``input_project_name``: name of the current project
 * ``input_read_user_group``: User group name to give READ permissions to for project resources (ML jobs, integration test job runs, and machine learning resources). A group with this name must exist in both the staging and prod workspaces. Defaults to "users", which grants read permission to all users in the staging/prod workspaces. You can specify a custom group name e.g. to restrict read permissions to members of the team working on the current ML project.
  * ``input_include_models_in_unity_catalog``: If selected, models will be registered to [Unity Catalog](https://docs.databricks.com/en/mlflow/models-in-uc.html#models-in-unity-catalog). Models will be registered under a three-level namespace of `<catalog>.<schema_name>.<model_name>`, according the the target environment in which the model registration code is executed. Thus, if model registration code runs in the `prod` environment, the model will be registered to the `prod` catalog under the namespace `<prod>.<schema>.<model_name>`. This assumes that the respective catalogs exist in Unity Catalog (e.g. `dev`, `staging` and `prod` catalogs). Target environment names, and catalogs to be used are defined in the Databricks bundles files, and can be updated as needed.
 * ``input_schema_name``: If using [Models in Unity Catalog](https://docs.databricks.com/en/mlflow/models-in-uc.html#models-in-unity-catalog), specify the name of the schema under which the models should be registered, but we recommend keeping the name the same as the project name. We default to using the same `schema_name` across catalogs, thus this schema must exist in each catalog used. For example, the training pipeline when executed in the staging environment will register the model to `staging.<schema_name>.<model_name>`, whereas the same pipeline executed in the prod environment will register the mode to `prod.<schema_name>.<model_name>`. Also, be sure that the service principals in each respective environment have the right permissions to access this schema, which would be `USE_CATALOG`, `USE_SCHEMA`, `MODIFY`, `CREATE_MODEL`, and `CREATE_TABLE`.
 * ``input_unity_catalog_read_user_group``: If using [Models in Unity Catalog](https://docs.databricks.com/en/mlflow/models-in-uc.html#models-in-unity-catalog), define the name of the user group to grant `EXECUTE` (read & use model) privileges for the registered model. Defaults to "account users".
 * ``input_include_feature_store``: If selected, will provide [Databricks Feature Store](https://docs.databricks.com/machine-learning/feature-store/index.html) stack components including: project structure and sample feature Python modules, feature engineering notebooks, ML resource configs to provision and manage Feature Store jobs, and automated integration tests covering feature engineering and training.
 * ``input_include_mlflow_recipes``: If selected, will provide [MLflow Recipes](https://mlflow.org/docs/latest/recipes.html) stack components, dividing the training pipeline into configurable steps and profiles.

See the generated ``README.md`` for next steps!

## Customize LLMOps Stacks
Your organization can use the default stack as is or customize it as needed, e.g. to add/remove components or
adapt individual components to fit your organization's best practices. See the
[stack customization guide](stack-customization.md) for more details.

## FAQ

### Do I need separate dev/staging/prod workspaces to use LLMOps Stacks?
We recommend using separate dev/staging/prod Databricks workspaces for stronger
isolation between environments. For example, Databricks REST API rate limits
are applied per-workspace, so if using [Databricks Model Serving](https://docs.databricks.com/applications/mlflow/model-serving.html),
using separate workspaces can help prevent high load in staging from DOSing your
production model serving endpoints.

However, you can create a single workspace stack, by supplying the same workspace URL for
`input_databricks_staging_workspace_host` and `input_databricks_prod_workspace_host`. If you go this route, we
recommend using different service principals to manage staging vs prod resources,
to ensure that CI workloads run in staging cannot interfere with production resources.

### I have an existing ML project. Can I productionize it using LLMOps Stacks?
Yes. Currently, you can instantiate a new project and copy relevant components
into your existing project to productionize it. LLMOps Stacks is modularized, so
you can e.g. copy just the GitHub Actions workflows under `.github` or ML resource configs
 under ``{{.input_root_dir}}/{{template `project_name_alphanumeric_underscore` .}}/resources`` 
and ``{{.input_root_dir}}/{{template `project_name_alphanumeric_underscore` .}}/databricks.yml`` into your existing project.

### Can I adopt individual components of LLMOps Stacks?
For this use case, we recommend instantiating via [Databricks asset bundle templates](https://docs.databricks.com/en/dev-tools/bundles/templates.html) 
and copying the relevant subdirectories. For example, all ML resource configs
are defined under ``{{.input_root_dir}}/{{template `project_name_alphanumeric_underscore` .}}/resources``
and ``{{.input_root_dir}}/{{template `project_name_alphanumeric_underscore` .}}/databricks.yml``, while CI/CD is defined e.g. under `.github`
if using GitHub Actions, or under `.azure` if using Azure DevOps.

### Can I customize my LLMOps Stack?
Yes. We provide the default stack in this repo as a production-friendly starting point for LLMOps.
However, in many cases you may need to customize the stack to match your organization's
best practices. See [the stack customization guide](stack-customization.md)
for details on how to do this.

### Does the LLMOps Stacks cover data (ETL) pipelines?

Since LLMOps Stacks is based on [databricks CLI bundles](https://docs.databricks.com/dev-tools/cli/bundle-commands.html),
it's not limited only to ML workflows and resources - it works for resources across the Databricks Lakehouse. For instance, while the existing ML
code samples contain feature engineering, training, model validation, deployment and batch inference workflows,
you can use it for Delta Live Tables pipelines as well.

### How can I provide feedback?

Please provide feedback (bug reports, feature requests, etc) via GitHub issues.

## Contributing

We welcome community contributions. For substantial changes, we ask that you first file a GitHub issue to facilitate
discussion, before opening a pull request.

LLMOps Stacks is implemented as a [Databricks asset bundle template](https://docs.databricks.com/en/dev-tools/bundles/templates.html)
that generates new projects given user-supplied parameters. Parametrized project code can be found under
the `{{.input_root_dir}}` directory.

### Installing development requirements

To run tests, install [actionlint](https://github.com/rhysd/actionlint),
[databricks CLI](https://docs.databricks.com/dev-tools/cli/databricks-cli.html), [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm), and
[act](https://github.com/nektos/act), then install the Python
dependencies listed in `dev-requirements.txt`:

    pip install -r dev-requirements.txt

### Running the tests
**NOTE**: This section is for open-source developers contributing to the default stack
in this repo.  If you are working on an ML project using the stack (e.g. if you ran `databricks bundle init`
to start a new project), see the `README.md` within your generated
project directory for detailed instructions on how to make and test changes.

Run unit tests:

```
pytest tests
```

Run all tests (unit and slower integration tests):

```
pytest tests --large
```

Run integration tests only:

```
pytest tests --large-only
```

### Previewing changes
When making changes to LLMOps Stacks, it can be convenient to see how those changes affect
a generated new ML project. To do this, you can create an example
project from your local checkout of the repo, and inspect its contents/run tests within
the project.

We provide example project configs for Azure (using both GitHub and Azure DevOps), AWS (using GitHub), and GCP (using GitHub) under `tests/example-project-configs`.
To create an example Azure project, using Azure DevOps as the CI/CD platform, run the following from the desired parent directory
of the example project:

```
# Note: update LLMOps_STACKS_PATH to the path to your local checkout of the LLMOps Stacks repo
LLMOps_STACKS_PATH=~/LLMOps-stacks
databricks bundle init "$LLMOps_STACKS_PATH" --config-file "$LLMOps_STACKS_PATH/tests/example-project-configs/azure/azure-devops.json"
```

To create an example AWS project, using GitHub Actions for CI/CD, run:
```
# Note: update LLMOps_STACKS_PATH to the path to your local checkout of the LLMOps Stacks repo
LLMOps_STACKS_PATH=~/LLMOps-stacks
databricks bundle init "$LLMOps_STACKS_PATH" --config-file "$LLMOps_STACKS_PATH/tests/example-project-configs/aws/aws-github.json"
```

To create an example GCP project, using GitHub Actions for CI/CD, run:
```
# Note: update LLMOps_STACKS_PATH to the path to your local checkout of the LLMOps Stacks repo
LLMOps_STACKS_PATH=~/LLMOps-stacks
databricks bundle init "$LLMOps_STACKS_PATH" --config-file "$LLMOps_STACKS_PATH/tests/example-project-configs/gcp/gcp-github.json"
```
