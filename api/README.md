# ASF Heatpump Installer Chatbot API

## Structure

The API app is structured in this way so that RAG and chain related components are separated from API development and deployment.
This is so that dev dependencies can be separated into a standalone application, and allow the deployment environment to be stringently managed with better practices than in a dev or testing environment.

The API app uses `poetry`, and dependencies should eventually be separated from development of RAG and chain components, i.e. only add dependencies from `requirements.txt` that are relevant to deployment.

## Installation

To install with `poetry`, deactivate conda environments then run:

```bash
poetry env use 3.10
poetry install
```

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

## Running locally with langchain-cli for dev/ testing

With `langchain-cli` installed, we can run the server.py app script by specifying the relative module path from the `/app` folder.

First `cd` into `/app` then run:

```shell
langchain app serve --app app.server:app
```

## Adding packages

```bash
# adding packages from
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add $PROJECT_NAME

# adding custom GitHub repo packages
langchain app add --repo $OWNER/$REPO
# or with whole git string (supports other git providers):
# langchain app add git+https://github.com/hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
langchain app add $PROJECT_NAME --api_path=/my/custom/path/rag
```

Note: you remove packages by their api path

```bash
langchain app remove my/custom/path/rag
```

## Setup LangSmith (Optional, not available)

LangSmith will help us trace, monitor and debug LangChain applications.
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/).
If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Launch LangServe

```bash
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

## Deployment (Local testing)

docker-compose configs are also included. This allows deployment of the vector database and API as a single service, and makes use of a `.env` at the root of the repo. See `.env.template` for template example of required environment variables.

For the first time running run the following command to build and deploy locally.

```shell
docker compose up --build
```

Add `-d` to launch in daemon (background) mode.

To stop the service, find the container group in Docker dashboard and stop it there. You can also run the below command:

```shell
docker compose stop
```

To tear down the service:

```shell
docker compose down
```

## Deployment (Cloud/Prod)

1. Login to a cloud container registry (i.e. Elastic Container Registry)

2. Tag the container with the URL corresponding to the cloud registry

   - Examples: `docker.registry.url/image_name:tag` or `docker.registry.url/asf-hpi-chatbot:api-latest-prod`
   - Note that the part after colon is a version related tag

3. In the cloud instance, install dependencies, login to cloud container registry, and pull the container an example image url `docker.registry.url/asf-hpi-chatbot:api-latest-prod`

4. Setup reverse proxies (Caddy or Nginx)/ Route53, as well as other network config that is relevant

5. Add the relevant `.env` details, copy contents of `docker-compose-prod.yml` to the same folder, and run `docker compose -f docker-compose-prod.yml --build` or without the `--build` argument. Similar to the local testing deployment, but pointing directly to the production docker compose config file.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```
