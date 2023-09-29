# dineof-ee-gcp
Example running the DINEOF on GCP services using data from Earth Engine 

# cloud-run-dineof

README for the source code pertaining to DINEOF process using Cloud Run.

This subdirectory hosts the code and setup for executing the
DINEOF processing, including extracting data from Earth Engine, running DINEOF,
saving COGs to Cloud Storage, and indexing the COGs on Earth Engine as a COG 
backed asset collection.

The subdirectory within here, `src`, hosts all of the source code and main program
for the processing.

## Setup 

Artifact Registry is used to host the Docker images used to deploy the processing
job to Cloud Run. If the target repository does not exist, create a new repository:

```
REGION=us-central1
gcloud artifacts repositories create dineof-process \
    --location=$REGION \
    --repository-format=docker \
    --description="Repository for hosting the Docker images with the DINEOF binary and data transfer service" \
    --async
```

## Build the Docker images

Next, the Docker image needs to be built. This is done using Cloud Build. To 
submit job to build the Docker image and store in the registry repo just created:

```
gcloud builds submit --config cloudbuild.yaml
```

## Deploy the DINEOF job to Cloud Run

To deploy a processing job to gap fill a satellite data product, run the 
following command:

```
gcloud beta run jobs create dineof-proc-$(date '+%Y%m%d%H%M%S') \
  --image ${REGION}-docker.pkg.dev/sfwmd-sandbox/dineof-process/dineof-runner \
  --max-retries=0 \
  --cpu=2 \
  --memory=4Gi \
  --task-timeout=420m \
  --env-vars-file=job_envs.yaml \
  --execute-now
```

This will submit the processing job to the Cloud Run Jobs service and begin 
executing immediately. It should be noted that this uses the [Preview 
(i.e. beta) Job execution](https://cloud.google.com/run/docs/configuring/task-timeout#long-task-timeout)
 with 24hr execution time because DINEOF is a long running process that can take
 longer than 1 hr with the GA execution environment.

The arguments in the command are as follows:

* `image`: Docker image with command to run
* `max-retries`: number of times to attempt in case of error
* `cpu`: number of CPUs for the machine running the job
* `memory`: amount of RAM for the machine running the job
* `task-timeout`: time limit that the job task can run (maximum is 24hr)
* `service-account`: service account used to make authenticated calls (defaults 
to the default comput SA if one if not provided)
* `env-vars-file`: YAML file containing the enviromental variables for the job 
execution. This is where parameters are passed for different jobs
* `execute-now`: flag that starts the job immediately after being submitted. 
Otherwise it requires manual trigger to execute.

The `job_envs.yaml` file is where job specific parameters are changed. For 
example, if the DINEOF process would like to be run on the CICyano data or SST 
data. Each time the parameters change a new Cloud Run Job will need to be 
submitted or the environmental variables for the job will need to be updated 
before execution.