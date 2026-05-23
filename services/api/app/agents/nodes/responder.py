# services/api/app/agents/nodes/responder.py
from services.api.app.agents.state import AgentState
from services.api.app.clients.ray_llm import llm_client

LANGUAGE_INSTRUCTIONS = {
    "Chinese": "你必须用中文回答。无论问题是什么语言，始终用中文输出。如果文档中没有相关信息，用中文说明。",
    "English": "You must answer in English.",
    "German":  "Sie müssen auf Deutsch antworten.",
    "French":  "Vous devez répondre en français.",
    "Italian": "Devi rispondere in italiano.",
}

async def generate_node(state: AgentState) -> dict:
    query = state["current_query"]
    documents = state.get("documents", [])
    language = state.get("language", "Chinese")

    context_str = "\n\n".join(documents) if documents else ""
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["Chinese"])

    prompt = f"""You are a helpful Enterprise Assistant.

Language rule: {lang_instruction}

Context from knowledge base:
{context_str if context_str else "No documents found."}

User input: {query}

Rules:
1. Treat the user input as a question, even if it looks like a keyword or topic (e.g. "RAG" means "What is RAG?").
2. Answer based on the context above.
3. Cite sources using [Source: Filename] if available.
4. If the context does not contain enough information, say so in the required language.
5. Never ask the user to provide a question — always attempt to answer.
"""

    answer = await llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return {
        "messages": [{"role": "assistant", "content": answer}]
    }
