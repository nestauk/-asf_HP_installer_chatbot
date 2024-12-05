# Heat Pump Installer Chatbot Prototype

This repository contains the **Heat Pump Installer Chatbot**, a prototype chatbot designed to assist heat pump installers with on-the-job queries via WhatsApp. Built with a focus on large language model (LLM) integration, the chatbot serves as a knowledge-enhancing tool, enabling installers to access relevant information and solutions in real time. We have also developed an easy-to-use Streamlit based frontend that sits on top of this chatbot, which can be found [here](https://github.com/nestauk/hpi_chatbot_frontend).

## Key Features

- **Powered by LLM**: Utilises advanced language models to provide accurate and contextual responses.
- **Installer-Focused**: Designed with the needs of heat pump installers in mind, offering targeted insights and troubleshooting support.
- **Frontend Integration**: Seamless integration with front ends, such as our Streamlit implementation [here](https://github.com/nestauk/hpi_chatbot_frontend). Also possible to connect to via methods such as Whatsapp.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.10+**
- **Docker**
- **poetry**

## Installation & Deployment

For detailed instructions on getting the app up and running, see the [README](api/README.md) in the `api` directory

## Project Structure

-asf_HP_installer_chatbot/
├── api/                         # Core chatbot application code
├── asf_hp_installer_chatbot/    # Archive development and proof of concept (PoC) code.
├── rag/                         # Retrieval Augmented Generation (RAG) code
├── docs/                        # Documentation
├── .env.template                # Environment variable template
├── requirements.txt             # Python dependencies
├── Makefile                     # Automation commands
└── README.md                    # Project overview

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
