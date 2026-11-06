"""Choose whether this request needs the Team 2A eligibility service."""


def select_eligibility_client(client, enabled):
    if not enabled:
        return None
    if not client.eligibility_url:
        raise ValueError("2A filtering is unavailable. Set TEAM2A_MODE=disabled in the server environment until the Team 2A API is configured.")
    return client
