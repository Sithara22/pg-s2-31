import os

from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")

HF_MODEL = "Qwen/Qwen3-4B:featherless-ai"
GROQ_MODEL = "qwen/qwen3.8-27b"

LLM_MODEL = (
    GROQ_MODEL
    if LLM_PROVIDER == "groq"
    else HF_MODEL
)


def query_huggingface(model_input: dict, timeout: float = 120) -> dict:
    """Existing Hugging Face Qwen implementation."""

    token = os.environ["HF_TOKEN"]

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
        timeout=timeout,
        max_retries=1,
    )

    completion = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": model_input["system"]},
            {"role": "user", "content": model_input["user"]},
        ],
        temperature=0,
        max_tokens=1024,
    )

    usage = completion.usage
    answer = completion.choices[0].message.content or ""

    if not answer.strip():
        raise ValueError("Hugging Face returned an empty answer")

    return {
        "answer": answer.strip(),
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
    }


def query_groq(model_input: dict, timeout: float = 120) -> dict:
    """Generate RAG answers using Groq Qwen."""

    client = Groq(
        api_key=os.environ["GROQ_API_KEY"],
        timeout=timeout,
        max_retries=1,
    )

    completion = client.chat.completions.create(
        model=GROQ_MODEL,     
        messages=[
            {
                "role": "user",
                "content": (
                    model_input["system"]
                    + "\n\n"
                    + model_input["user"]
                    + "\n\n"
                    + "IMPORTANT: Return your response as "
                    "a valid JSON object only."
                ),
            },
        ],
        temperature=0.6,
        max_completion_tokens=2048,
        top_p=0.95,
        reasoning_effort="default",
        response_format={"type": "json_object"},
        stream=False,
    )

    usage = completion.usage
    answer = completion.choices[0].message.content or ""

    if not answer.strip():
        raise ValueError("Groq returned an empty answer")

    return {
        "answer": answer.strip(),
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
    }


def query_llm(model_input: dict, timeout: float = 120) -> dict:
    """Common entry point for both LLM providers."""

    if LLM_PROVIDER == "groq":
        return query_groq(model_input, timeout)

    if LLM_PROVIDER == "huggingface":
        return query_huggingface(model_input, timeout)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}"
    )