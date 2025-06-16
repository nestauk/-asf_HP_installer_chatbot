# AWS Deployment on bare EC2 instance with Docker Compose

This document outlines the steps to deploy the Installer Chatbot API on an AWS EC2 instance using Docker Compose. It will first walk you through the local deployment of the API, then how to push the Docker image to AWS Elastic Container Registry (ECR), and finally how to deploy it on an EC2 instance.

> **⚠️ DISCLAIMER:**
>
> This document is for reference only and is not intended to be a comprehensive guide.
>
> It is designed to be flexible and can be adapted to fit your specific requirements.
>
> **We will not be responsible for any issues that arise from following these steps, as they are provided as a guide.**
>
> It is important to test and validate your deployment in a controlled environment before using it in production.

## Things to note:

- It is recommended to use an EC2.medium instance (2vCPU, 4GB RAM) or larger.
- It is recommended to use Ubuntu AMIs.
- The instance should have at least 10GB of storage, this is for OS dependencies, Docker, Docker Compose, and the API Docker image (~2.79GB). You will need more storage if you plan to store large datasets or documents. As reference, Nesta's deployment uses `50GB` of `gp3` storage.
- You may find that setting up a Swap file to be more cost efficient than using a larger instance type for more RAM, especially if you are running multiple applications on the same instance, or have a large collection of PDF documents. This is not covered in this document, but you can find more information on how to set up a swap file [here](https://repost.aws/knowledge-center/ec2-memory-swap-file).
- Please set up an Elastic Container Registry (ECR) repository to store the Docker image.
- Ensure that the security group allows inbound traffic on port 80 (HTTP) and port 443 (HTTPS) if you plan to use SSL.
- You will need to set up your own reverse proxy or load balancer (e.g., Nginx or Caddy) if you want to serve the API over HTTP/HTTPS. See [section 3.8](#38optional-setup-reverse-proxy-with-caddy) for instructions on using Caddy.

## Contents

- [1. Local Deployment](#1-local-deployment)
- [2. Push the Docker Image to ECR](#2-push-the-docker-image-to-ecr)
- [3. Deploy on AWS EC2 Instance](#3-deploy-on-aws-ec2-instance)
  - [3.1. Launch an EC2 Instance](#31-launch-an-ec2-instance)
  - [3.2. Install Docker and Docker Compose](#32-install-docker-and-docker-compose)
  - [3.3. Install AWS](#33-install-aws)
  - [3.4. Pull the Docker Image from ECR](#34-pull-the-docker-image-from-ecr)
  - [3.5. Copy the Docker Compose File into the EC2 Instance](#35-copy-the-docker-compose-file-into-the-ec2-instance)
  - [3.6. Create the Documents Directory](#36-create-the-documents-directory)
  - [3.7. Create a TMUX session to run the API in the background](#37-create-a-tmux-session-to-run-the-api-in-the-background)
  - [3.8. Run the Docker Compose Command](#38-run-the-docker-compose-command)
  - [3.9. Setup Reverse Proxy with Caddy (Optional)](#39-setup-reverse-proxy-with-caddy-optional)
  - [3.10. Route53 Configuration (Optional)](#310-route53-configuration-optional)
  - [3.11. Access the API](#311-access-the-api)
  - [3.12. Firewall Configuration](#312-firewall-configuration)
- [4. Document Ingestion](#4-document-ingestion)
  - [4.1.1. Notes on Document Ingestion](#411-notes-on-document-ingestion)
  - [4.1.2. Explore document collections in the Qdrant dashboard](#412-explore-document-collections-in-the-qdrant-dashboard)
- [5. Additional Resources](#5-additional-resources)
- [6. Troubleshooting](#6-troubleshooting)
- [7. Conclusion](#7-conclusion)
- [8. License](#8-license)
- [9. Additional Notes](#9-additional-notes)

## 1. Local Deployment

As a first step, the Chatbot API image needs to be built. The following command can also be used in the [cloud deployment steps](#3-deploy-on-aws-ec2-instance) below to Deploy the API on an EC2 instance.

```bash
# RUN THIS COMMAND IN THE /api DIRECTORY
DOCKER_REGISTRY=<YOUR-AWS-ACCOUNT-NUMBER>.dkr.ecr.eu-west-2.amazonaws.com STAGE=prod DOCUMENTS_PATH=../inputs/data/ docker compose -f docker-compose.yml up --build
```

This command builds the Docker image and starts the container. Make sure to replace `<YOUR-AWS-ACCOUNT-NUMBER>.dkr.ecr.eu-west-2.amazonaws.com` with your actual ECR repository URL.

From the above command, there are some environment variables that you can see is used in Line 33 and Line 45-46 of `docker-compose.yml`,

```yaml
# Line 33
image: ${DOCKER_REGISTRY}/asf-hpi-chatbot:api-latest-${STAGE}
```

```yaml
# Lines 45-46
volumes:
  - ${DOCUMENTS_PATH}:/documents:ro
```

Where:

- `DOCKER_REGISTRY` is the URL of your ECR registry.
- `STAGE` is the deployment stage, e.g., `prod`, `dev`, etc.
- `DOCUMENTS_PATH` is the path to the documents directory that contains the data files (.pdf files) for the API. This is relative to the `/api` directory where the `docker-compose.yml` file is located. You can set this to the path where your documents are stored locally, such as `../inputs/data/`.

These can be modified to fit your requirements. The image naming is a convention and is trivial to change.

### 1.1. Access the API Locally

Once the Docker container is running, you can access the API at `http://0.0.0.0:8000`. You should see the Swagger documentation page for the API, which allows you to interact with the API endpoints.

### 1.2. Stopping the API

To stop the API, you can use `Ctrl + C` in the terminal where the Docker container is running. Then run the following command to remove the Docker compose services and clean up the resources:

```bash
docker compose -f docker-compose.yml down
```

## 2. Push the Docker Image to ECR

After building the Docker image, you need to push it to your ECR repository. First, authenticate Docker to your ECR registry:

```bash
aws ecr get-login-password --region <YOUR_REGION> | docker login --username AWS --password-stdin <YOUR-AWS-ACCOUNT-NUMBER>.dkr.ecr.eu-west-2.amazonaws.com
```

Then, push the Docker image:

```bash
docker push <YOUR-AWS-ACCOUNT-NUMBER>.dkr.ecr.eu-west-2.amazonaws.com/asf-hpi-chatbot:api-latest-${STAGE}
```

## 3. Deploy on AWS EC2 Instance

### 3.1. Launch an EC2 Instance

1. Go to the AWS Management Console and navigate to the EC2 service.
2. Click on "Launch Instance".
3. Choose an Ubuntu AMI (e.g., Ubuntu Server 20.04 LTS).
4. Select an instance type (e.g., `t3a.medium` or larger).
5. Configure the instance details, ensuring you have at least 10GB of storage.
6. Configure the security group to allow inbound traffic on port 80 (HTTP) and port 443 (HTTPS).
7. Review and launch the instance, ensuring you have a key pair for SSH access.

### 3.2. Install Docker and Docker Compose

The following is adapted from the [official Docker documentation](https://docs.docker.com/engine/install/ubuntu/) for installing Docker and Docker Compose on Ubuntu.

First, SSH into your EC2 instance using the key pair you created during the instance launch:

```bash
ssh -L 6333:127.0.0.1:6333 -L 6334:127.0.0.1:6334 -i path_to/asf-hpi-chatbot.pem ubuntu@ec2-XX-XXX-XX-XX.compute.amazonaws.com
```

> NOTE: XX-XXX-XX-XX is the public IP of the EC2 instance.

Next, run the following commands to install Docker and Docker Compose:

```bash
# Update the package index
sudo apt-get update && sudo apt-get upgrade -y

# Remove conflicting legacy packages
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Setup Docker `apt` repository
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker Engine, CLI, and Containerd
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start Docker service
sudo service docker start

# Create the docker group.
sudo groupadd docker

# Add your user to the docker group.
sudo usermod -aG docker $USER

# Configure Docker to start on boot with systemd
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

### 3.3. Install AWS

Install the AWS CLI on your EC2 instance to interact with AWS services:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Ensure that the instance has the necessary IAM role with permissions to access ECR. You can attach a policy like `AmazonEC2ContainerRegistryFullAccess` to the instance role.

### 3.4. Pull the Docker Image from ECR

Now that Docker, Docker Compose, and AWS CLI are installed and the necessary IAM role with permissions to access ECR is assigned, you can pull the Docker image from ECR:

```bash
# First, within your EC2 instance, authenticate Docker for access to your ECR registry
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin <YOUR-AWS-ACCOUNT-NUMBER>.dkr.ecr.eu-west-2.amazonaws.com

# Then pull the Docker image
docker pull <YOUR-AWS-ACCOUNT-NUMBER>.dkr.ecr.eu-west-2.amazonaws.com/asf-hpi-chatbot:api-latest-${STAGE}
```

### 3.5. Copy the Docker Compose File into the EC2 Instance

You can use `scp` to copy the `docker-compose.yml` file from your local machine to the EC2 instance:

```bash
scp -i /path/to/your-key.pem docker-compose.yml ubuntu@<YOUR_EC2_PUBLIC_IP>:/home/ubuntu/
```

### 3.6. Create the Documents Directory

Create a directory on the EC2 instance to hold the documents that the API will use. This should match the `DOCUMENTS_PATH` you set according to the `docker-compose.yml` file. For ease as example, create a directory named `pdf` in the same location as the `docker-compose.yml` file:

```bash
mkdir -p ./pdf
```

### 3.7. Create a TMUX session to run the API in the background

You can use `tmux` to run the API in the background. If `tmux` is not installed, you can install it using:

```bash
sudo apt-get install tmux
```

Then, create a new `tmux` session:

```bash
tmux new -s app
```

Once inside the `tmux` session, you can run the Docker Compose command to start the API. This allows you to detach from the session and keep the API running in the background.

### 3.8. Run the Docker Compose Command

SSH into your EC2 instance:

```bash
ssh -i /path/to/your-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

Then navigate to the directory where you copied the `docker-compose.yml` file and run the same Docker Compose command as before:

```bash
# RUN THIS COMMAND IN WHERE THE docker-compose.yml FILE IS LOCATED
DOCKER_REGISTRY=<YOUR-AWS-ACCOUNT-NUMBER>.dkr.ecr.eu-west-2.amazonaws.com STAGE=prod DOCUMENTS_PATH=./pdf/ docker compose -f docker-compose.yml up
```

The API will be accessible on port 8000 of your EC2 instance.

You can exit the `tmux` session by pressing `Ctrl + b`, then `d`. This will keep the API running in a background terminal session.

### 3.9. Setup Reverse Proxy with Caddy (Optional)

If you want to serve the API over HTTP/HTTPS, you can set up a reverse proxy using Caddy. Caddy is a web server that automatically obtains and renews SSL certificates.

This step is essential for production if:

- You want HTTPS
- You have a custom domain
- You want a clean, secure, public-facing API

The choice of reverse proxy is up to you, but Caddy is recommended for its simplicity and automatic SSL certificate management.

To install Caddy, follow these steps:

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# Create a Caddyfile
touch Caddyfile
```

Edit the `Caddyfile` to include the following configuration:

```caddyfile
# Caddyfile
your-domain.com {
    reverse_proxy localhost:8000 {
        header_up Host {http.reverse_proxy.upstream.hostport}
        header_up X-Real-IP {http.reverse_proxy.upstream.remote_addr}
        header_up X-Forwarded-For {http.reverse_proxy.upstream.remote_addr}
        header_up X-Forwarded-Port {http.reverse_proxy.upstream.remote_port}
        header_up X-Forwarded-Proto {http.reverse_proxy.upstream.scheme}
    }
}
```

Make sure to replace `your-domain.com` with your actual domain name.

After creating the `Caddyfile`, you can start Caddy:

```bash
sudo systemctl start caddy
```

You can check the status of Caddy to ensure it is running:

```bash
sudo systemctl status caddy
```

If you want Caddy to start automatically on boot, enable it with:

```bash
sudo systemctl enable caddy
```

You may need to restart the Caddy service after making changes to the `Caddyfile`:

```bash
sudo caddy stop && sudo caddy start
```

### 3.10. Route53 Configuration (Optional)

If you want to use a custom domain name for your API, you can set up a Route53 hosted zone and create an A record that points to your EC2 instance's public IP address.

1. Go to the AWS Management Console and navigate to Route53.
2. Create a new hosted zone for your domain name.
3. Create an A record that points to your EC2 instance's public IP address.
4. If you are using Caddy as a reverse proxy, ensure that the domain name in the `Caddyfile` (previous section) matches the A record you created in Route53.

For more information on how to set up Route53, refer to the [AWS Route53 documentation](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/Welcome.html).

### 3.11. Access the API

You can now access the API at `http://<YOUR_EC2_PUBLIC_IP>:8000` or `https://your-domain.com` if you set up the reverse proxy with Caddy.

You should be redirected to the API Swagger documentation page, where you can test interactions with the API endpoints.

### 3.12. Firewall Configuration

If you have a firewall enabled on your EC2 instance, make sure to allow inbound traffic on ports:

- `:22` for SSH connections using a key pair file
- `:8000` (or the port you configured for the API)
- `:6333` for Qdrant HTTP endpoints and dashboard
- `:6334` for Qdrant gRPC endpoints
- `:80` for HTTP
- `:443` for HTTPS

You may want to restrict access to these ports to specific IP addresses or ranges for security reasons.

This can be adjusted through the EC2 security group settings in the AWS Management Console.

## 4. Document Ingestion

Copy your pdf files into the `pdf` directory you created earlier on the EC2 instance. The API will automatically index these documents when it starts up.

You can also run the following command to manually trigger the indexing of documents into a `test` collection:

```bash
DOCKER_REGISTRY=<YOUR-AWS-ACCOUNT-NUMBER>.dkr.ecr.eu-west-2.amazonaws.com STAGE=prod DOCUMENTS_PATH=./pdf/ docker compose run app chatbot init_vdb --url=your-domain.com:6334 --doc_directory=/documents/ --collection_name=test --max_workers 4
```

### 4.1.1. Notes on Document Ingestion

- The `chatbot init_vdb` command initializes the vector database with the documents in the specified directory.
- The `--url` parameter should point to the URL of your API, which can be your EC2 public IP or domain name. Make sure to replace `your-domain.com` with your actual domain name or the public IP of your EC2 instance.
- The `--doc_directory` parameter should point to the volume mounted path where your documents are stored, which is `/documents/` in this case. See Line 46 of the `docker-compose.yml` file. Where `:/documents:ro` is the volume mount path with Read-Only (`ro`) access.
- When `--collection_name` is not set, the default collection name is `chatbot-onboarding`.
- The `--max_workers` parameter can be adjusted based on the number of CPU cores available on your EC2 instance. This can speed up the indexing process with parallelism.
- To index documents into an existing Docker Compose service stack, you will need to modify the `docker-compose.yml` file to point to the stack network as an external network. This allows the container command to communicate with the `app` service. For example:

```yaml
version: "3.8"
services:
  app:
    networks:
      - my-network
networks:
  my-network:
    external: true
```

### 4.1.2. Explore document collections in the Qdrant dashboard

You can explore the document collections in the Qdrant dashboard by accessing the Qdrant web interface. On a local deployment, this is typically available at `http://0.0.0.0:6333/dashboard`. Please refer to the [Qdrant Web UI documentation](https://qdrant.tech/documentation/web-ui/) for more details on how to use the dashboard.

Some things you can do in the Qdrant dashboard:

- View the collections and their metadata.
- Explore the indexed documents and their embeddings in a cluster view.
- Perform searches and queries on the indexed documents.
- Manage collections, including creating, deleting, and updating them.

## 5. Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [AWS ECR Documentation](https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/index.html)
- [Caddy Documentation](https://caddyserver.com/docs/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)
- [AWS IAM Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)
- [AWS Route53 Documentation](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/Welcome.html)

## 6. Troubleshooting

If you encounter issues during the deployment, here are some common troubleshooting steps:

- **Docker Issues**: Ensure that Docker is running and that you have the correct permissions to run Docker commands. You may need to log out and log back in after adding your user to the `docker` group.
- **ECR Authentication**: If you have issues pulling the Docker image from ECR, ensure that your AWS CLI is configured with the correct credentials and that the IAM role attached to your EC2 instance has the necessary permissions to access ECR.
- **Network Issues**: Ensure that your security group allows inbound traffic on the necessary ports (80, 443, 8000) and that your EC2 instance has a public IP address or is accessible through a domain name.
- **Caddy Configuration**: If you are using Caddy as a reverse proxy, ensure that the `Caddyfile` is correctly configured and that Caddy is running. You can check the Caddy logs for any errors.
- **Document Ingestion**: If the document ingestion fails, check the logs for any errors related to file permissions or missing files. Ensure that the `DOCUMENTS_PATH` is correctly set and that the files are accessible by the Docker container.
- **Qdrant Issues**: If you have issues with Qdrant, check the Qdrant logs for any errors. Ensure that the Qdrant service is running and that the API is correctly configured to connect to it.
- **API Access**: If you cannot access the API, check the security group settings and ensure that the API is running on the correct port. You can also check the Docker logs for any errors related to the API service.
- **SSL Issues**: If you are using SSL with Caddy and encounter issues, check the Caddy logs for any errors related to certificate generation or renewal. Ensure that your domain name is correctly configured and that DNS records point to your EC2 instance.
- **Performance Issues**: If the API is slow or unresponsive, consider increasing the instance size or optimizing the document indexing process. You can also monitor the resource usage (CPU, memory) of the EC2 instance to identify bottlenecks.

## 7. Conclusion

This document provides a comprehensive guide to deploying the Installer Chatbot API on an AWS EC2 instance using Docker Compose. By following these steps, you can set up a scalable and efficient deployment of the API, allowing you to leverage the power of AWS for your applications. If you have any questions or need further assistance, feel free to reach out to the community or consult the additional resources provided.

## 8. License

This repository is licensed under the AGPL-3.0 license. You can find the full text of the license in the [LICENSE](../LICENSE) file in the root directory of this repository.

## 9. Additional Notes

These deployment steps is for reference only, designed to be flexible, and can be adapted to fit your specific requirements. Whether you are deploying the API for development, testing, or production use, the principles outlined here will help you get started with a robust deployment on AWS.

We will not be responsible for any issues that arise from following these steps, as they are provided as a guide. It is important to test and validate your deployment in a controlled environment before using it in production.

If you have any suggestions for improvements or additional features, please feel free to contribute to the project or open an issue on the GitHub repository. Your feedback is valuable and helps us improve the documentation and deployment process for everyone.
