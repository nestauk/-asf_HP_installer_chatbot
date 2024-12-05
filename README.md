# Heat Pump Installer Chatbot Prototype

This repository contains the **Heat Pump Installer Chatbot**, a prototype chatbot designed to assist heat pump installers with on-the-job queries via WhatsApp. Built with a focus on large language model (LLM) integration, the chatbot serves as a knowledge-enhancing tool, enabling installers to access relevant information and solutions in real time.

## Key Features

- **WhatsApp Integration**: Chatbot accessible directly from WhatsApp for seamless communication.
- **Powered by LLM**: Utilizes advanced language models to provide accurate and contextual responses.
- **Installer-Focused**: Designed with the needs of heat pump installers in mind, offering targeted insights and troubleshooting support.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **Conda**: For environment management. Install from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
- **Direnv**: For managing environment variables. Install it via your package manager ([direnv docs](https://direnv.net/)).
- **Make**: Required for automated tasks.

## Installation

Follow these steps to set up the chatbot on your local machine:

1. **Clone the repository**:
   ```
   git clone https://github.com/nestauk/-asf_HP_installer_chatbot.git
   cd -asf_HP_installer_chatbot
   ```

2. **Set up the environment**:
   - Ensure `direnv` is installed and enabled.
   - Run the following command to install and configure project dependencies:
     ```make install```
   This will:
   - Create a conda environment.
   - Install all necessary dependencies.
   - Set up pre-commit hooks for code formatting and linting.

3. **Activate the environment**:
   - Allow `direnv` to load the environment:
     ```direnv allow```
   - Activate the conda environment:
     ```conda activate hp_installer_chatbot```

4. **Run the application**:
   ```python app.py```

Now the chatbot should be up and running locally!

## Project Structure

-asf_HP_installer_chatbot/
├── app/                 # Core chatbot application code
├── data/                # Data assets and configurations
├── tests/               # Unit tests and test configurations
├── docs/                # Documentation and usage guides
├── .env.template        # Environment variable template
├── requirements.txt     # Python dependencies
├── Makefile             # Automation commands
└── README.md            # Project overview

## Usage

Once the chatbot is running, it connects to WhatsApp to handle user queries. Follow these steps to start using the bot:

1. **Deploy the Bot**:
   - Use the Twilio WhatsApp API to set up the connection.
   - Update the .env file with your Twilio API credentials and chatbot settings.

2. **Test the Chatbot**:
   - Send messages via WhatsApp to the configured bot number.
   - The chatbot will respond with relevant information or troubleshooting steps.

## Deployment

To deploy the chatbot on a server:

1. **Dockerize the Application**:
   - Build the Docker image:
     ```docker build -t hp_installer_chatbot .```
   - Run the container:
     ```docker run -p 8000:8000 --env-file .env hp_installer_chatbot```

2. **Set Up a Public Endpoint**:
   - Use tools like ngrok or a cloud platform to expose the bot’s endpoint to Twilio.

## Contributing

We welcome contributions! To get started:

1. Fork this repository and create your feature branch:
   ```git checkout -b feature/your-feature```

2. Commit your changes:
   ```git commit -m "Add your feature"```

3. Push to the branch:
   ```git push origin feature/your-feature```

4. Open a pull request.

For detailed guidelines, check out the CONTRIBUTING.md file in the repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
