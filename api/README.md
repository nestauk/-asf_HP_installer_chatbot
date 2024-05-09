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
