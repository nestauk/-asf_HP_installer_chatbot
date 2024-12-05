# Heat Pump Installer Chatbot Prototype

This repository contains the **Heat Pump Installer Chatbot**, a prototype chatbot designed to assist heat pump installers with on-the-job queries via WhatsApp. Built with a focus on large language model (LLM) integration, the chatbot serves as a knowledge-enhancing tool, enabling installers to access relevant information and solutions in real time.

## Key Features

- **WhatsApp Integration**: Chatbot accessible directly from WhatsApp for seamless communication.
- **Powered by LLM**: Utilizes advanced language models to provide accurate and contextual responses.
- **Installer-Focused**: Designed with the needs of heat pump installers in mind, offering targeted insights and troubleshooting support.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.10+**
- **Docker**
- **poetry**

## Installation & Deployment

For detailed instructions on getting the app up and running, see the README in the `api` directory

## Project Structure

-asf_HP_installer_chatbot/
├── api/                         # Core chatbot application code
├── asf_hp_installer_chatbot/    # Pipeline code
├── rag/                         # Retrieval Augmented Generation (RAG) code
├── docs/                        # Documentation
├── .env.template                # Environment variable template
├── requirements.txt             # Python dependencies
├── Makefile                     # Automation commands
└── README.md                    # Project overview

## Usage

Once the chatbot is running, it connects to WhatsApp to handle user queries. Follow these steps to start using the bot:

1. **Deploy the Bot**:
   - Use the Twilio WhatsApp API to set up the connection.
   - Update the .env file with your Twilio API credentials and chatbot settings.

2. **Test the Chatbot**:
   - Send messages via WhatsApp to the configured bot number.
   - The chatbot will respond with relevant information or troubleshooting steps.

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
