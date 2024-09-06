import os, logging
from datetime import date
from fastapi import Response, Request
from twilio.twiml.messaging_response import MessagingResponse

# from twilio.request_validator import RequestValidator

from rag.chains import rag_chain_with_source  # , get_session_history
from rag.utils.callbacks import langfuse_handler_from_config
from app.utils.messaging import get_sender_hash

logger = logging.getLogger(__file__)


async def chat(request: Request) -> Response:
    """Interface for the Twilio WhatsApp webhook.

    Args:
        request (Request): The incoming webhook request from Twilio.

    Returns:
        Response: The response to the incoming webhook request.
    """
    form_data = await request.form()
    # TODO test and add form validation of request, will need to add a test for this
    # TODO refactor out whatsapp client to separate app
    # validator = RequestValidator(os.environ.get("TWILIO_AUTH_TOKEN"))
    # webhook_url = os.environ.get("WEBHOOK_URL")
    # twilio_signature = request.headers.get("X-Twilio-Signature")
    # validator.validate(webhook_url, form_data, twilio_signature)
    whatsapp_number = form_data["From"]
    body = form_data["Body"]
    user_id = get_sender_hash(whatsapp_number)
    chat_session_id = f"{user_id}_{date.today().isoformat()}"

    langfuse_handler = langfuse_handler_from_config(
        trace_name="whatsapp_chatbot_chain",
        user_id=user_id,
        session_id=date.today().isoformat(),  # different session_id from chains, langfuse specific
        version="0.1.0",
        release="0.1.0",
        tags=["dev", "api", "v1", "whatsapp"],
    )  # TODO move to env vars

    # chat with history is disabled for now
    chain_res = await rag_chain_with_source().ainvoke(
        body,
        config={
            "configurable": {"session_id": chat_session_id},
            "callbacks": [langfuse_handler],
        },
    )

    logging.info(f"Result: {chain_res}")
    # logging.info(f"History: {get_session_history(chat_session_id)}")

    response = MessagingResponse()
    response.message(chain_res["answer"])

    return Response(content=str(response), media_type="application/xml")
