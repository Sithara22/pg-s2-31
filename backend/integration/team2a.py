"""Provisional Team 2A boundary. Replace mappings when their contract arrives."""

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Team2AError(RuntimeError):
    """The upstream feed or eligibility service could not be trusted."""


def _ids(value):
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in value
    ):
        raise Team2AError("Expected an array of non-empty EKO IDs.")
    if len(value) != len(set(value)):
        raise Team2AError("Duplicate EKO IDs in response.")
    return value


class Team2AClient:
    def __init__(self, bulk_url=None, eligibility_url=None, token=None, timeout=30):
        self.bulk_url = bulk_url or os.getenv("TEAM2A_BULK_URL")
        self.eligibility_url = eligibility_url or os.getenv("TEAM2A_ELIGIBILITY_URL")
        self.token = token if token is not None else os.getenv("TEAM2A_API_TOKEN")
        self.timeout = timeout

    def _request(self, url, payload=None):
        if not url:
            raise Team2AError("Team 2A endpoint is not configured.")
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise Team2AError("Team 2A request failed or returned invalid JSON.") from exc

    def fetch_bulk(self):
        return self._request(self.bulk_url)

    def eligible_ids(self, question, context):
        response = self._request(self.eligibility_url, {**context, "question": question})
        if not isinstance(response, dict) or set(response) != {"eko_ids"}:
            raise Team2AError("Expected an eligibility object containing only eko_ids.")
        return _ids(response["eko_ids"])


def map_bulk_snapshot(payload):
    """Validate a complete snapshot before allowing any index mutation.

    Provisional identity: ko_id is the immutable VERSION ID used for retrieval
    and citations. logical_ko_id identifies the object across versions.
    """
    if not isinstance(payload, dict) or set(payload) != {"complete", "ekos"} or payload["complete"] is not True:
        raise Team2AError("A confirmed complete bulk snapshot is required.")
    if not isinstance(payload["ekos"], list):
        raise Team2AError("ekos must be an array.")
    records = []
    logical_ids = set()
    all_version_ids = []
    strings = ("ko_id", "logical_ko_id", "title", "stage", "trade", "jurisdiction", "learner_level",
               "publication_status", "validation_status", "l1", "l2", "l3", "l4", "l5")
    flags = ("ai_indexable", "viewer_exposable", "superseded", "is_current")
    for item in payload["ekos"]:
        if not isinstance(item, dict) or any(not isinstance(item.get(key), str) for key in strings):
            raise Team2AError("EKO fields do not match the provisional schema.")
        if any(type(item.get(key)) is not bool for key in flags):
            raise Team2AError("Explicit boolean eligibility flags are required.")
        if type(item.get("version")) is not int or item["version"] < 1:
            raise Team2AError("A positive integer version is required.")
        if any(not item[key].strip() or item[key] != item[key].strip() for key in ("ko_id", "logical_ko_id", "title")):
            raise Team2AError("Identity and title fields must be non-empty and trimmed.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", item["ko_id"]):
            raise Team2AError("Version IDs must be safe citation identifiers.")
        all_version_ids.append(item["ko_id"])
        eligible = (item["publication_status"] == "published" and item["validation_status"] == "validated"
                    and item["ai_indexable"] and item["viewer_exposable"]
                    and item["is_current"] and not item["superseded"])
        if not eligible:
            continue
        if item["logical_ko_id"] in logical_ids:
            raise Team2AError("Multiple eligible versions of the same logical EKO.")
        logical_ids.add(item["logical_ko_id"])
        sections = [("Title", item["title"])] + [(f"L{i}", item[f"l{i}"]) for i in range(1, 6)]
        if not any(item[f"l{i}"].strip() for i in range(1, 6)):
            raise Team2AError("Eligible EKO has no knowledge text.")
        records.append({
            "ko_id": item["ko_id"],
            "searchable_text": "\n".join(f"{label}: {' '.join(value.split())}" for label, value in sections if value.strip()),
            "metadata": {key: item[key] for key in ("title", "stage", "trade", "jurisdiction", "learner_level", "version")},
        })
    _ids(all_version_ids)
    if not records:
        raise Team2AError("Empty eligible snapshot: refusing to erase the active index.")
    return records
