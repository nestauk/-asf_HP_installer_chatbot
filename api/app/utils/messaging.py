import hashlib


def get_sender_hash(sender_number: str) -> str:
    """
    Generates and returns a SHA256 hash of the sender's number. This is done so as to anonymise the sender's number.

    Args:
        sender_number (str): The sender's number.

    Returns:
        str: A SHA256 hash of the sender's number.
    """
    sender_number = sender_number.replace("whatsapp:", "")
    return hashlib.sha256(sender_number.encode()).hexdigest()
