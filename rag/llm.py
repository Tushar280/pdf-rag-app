from typing import List
from rag.config import (
    LLM_BACKEND,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    LOCAL_LLM_MODEL_PATH,
    LOCAL_LLM_MODEL_TYPE,
    LOCAL_LLM_MAX_TOKENS,
    LOCAL_LLM_TEMPERATURE,
)
from rag.utils import logger
import os

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly using the provided context from PDFs.
If the answer is not present in the context, say you don't know.
Cite sources with (filename:page) wherever relevant.

Question:
{question}

Context:
{context}

Answer:"""


class LocalLLM:
    def __init__(self):
        from ctransformers import AutoModelForCausalLM
        if not LOCAL_LLM_MODEL_PATH or not os.path.exists(LOCAL_LLM_MODEL_PATH):
            raise ValueError(f"Local LLM model not found at: {LOCAL_LLM_MODEL_PATH}")
        logger.info(f"Loading local LLM from {LOCAL_LLM_MODEL_PATH}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_MODEL_PATH,
            model_type=LOCAL_LLM_MODEL_TYPE,
        )
        self.max_tokens = LOCAL_LLM_MAX_TOKENS
        self.temperature = LOCAL_LLM_TEMPERATURE

    def generate(self, prompt: str) -> str:
        out = self.llm(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return out.strip()


class OpenAILLM:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        from openai import OpenAI
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that accurately cites sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()



def get_llm():
    if LLM_BACKEND == "openai":
        return OpenAILLM()
    return LocalLLM()



def format_context(docs: List[dict]) -> str:
    blocks = []
    for d in docs:
        meta = f"({d.get('filename','unknown')}:{d.get('page','?')})"
        text = d.get("text", "")[:3000]
        blocks.append(f"{meta}\n{text}")
    return "\n\n---\n\n".join(blocks)


def make_prompt(question: str, docs: List[dict]) -> str:
    return PROMPT_TEMPLATE.format(question=question, context=format_context(docs))
