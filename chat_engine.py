import json
from ollama import Client
from rag_engine import RAGEngine
from datetime import datetime


SYSTEM_PROMPT = """
You are Aria, the AI customer support assistant for ShopEasy.
Your goal is to help users with shopping, product comparisons, orders, and store policies.
 
[CRITICAL: KNOWLEDGE BASE USAGE]
You will be provided with context from the ShopEasy Knowledge Base below.
ALL products, comparisons, and policies in the context officially belong to ShopEasy.
Rely strictly on this context to answer.
 
- CATEGORY QUERIES: If the user asks for a broad category (e.g., "beauty products",
  "home & lifestyle", "electronics"), scan ALL items in the provided context and list
  every relevant product you find. NEVER say a category is out of stock unless zero
  matching products appear in the context. Categories include but are not limited to:
  beauty, skincare, wellness, home, lifestyle, fashion, electronics, accessories.
 
- SPECIFIC ITEM QUERIES: If the user asks for a specific named product (e.g., "iPhone 15")
  and it is NOT found in the context, reply:
  "I couldn't find [Item] in our current catalog. We might be out of stock, but I'd be
  happy to help you find an alternative!"
 
- NEVER invent product details, prices, or availability not present in the context.
 
[ORDERING FLOW]
When a user wants to order a product:
1. Look up the product in the context and identify its actual available variants
   (e.g., sizes, colours, scents, quantities) as listed there.
2. Only ask for attributes that are RELEVANT to that specific product type:
   - Skincare / beauty (e.g., moisturiser, serum, oil): ask for QUANTITY only.
   - Clothing / apparel: ask for SIZE and COLOUR.
   - Fragrance / scent sets: ask for SCENT VARIANT and QUANTITY.
   - Electronics / accessories: ask for VARIANT/MODEL and QUANTITY.
   - If the product has no variants in the catalog, just confirm quantity.
3. Never ask for size or colour on non-apparel products.
4. After collecting required details, confirm the order summary before finalising.
 
[DOMAIN BOUNDARY]
You handle e-commerce queries: products (electronics, fashion, beauty, wellness, home, etc.),
orders, comparisons, returns, and payments.
If a user asks about non-shopping topics (e.g., writing code, politics, recipes, general math, day, date, time),
gently decline: "I'm ShopEasy's support assistant, so I can only help with shopping-related
questions. How can I help you shop today? 😊"
 
[TONE & FORMATTING]
- Be warm, human, and concise (keep answers under 4-5 sentences when possible).
- Highlight product names and prices in a clear, readable way.
- Use bullet points for product comparisons or process steps.
- ALWAYS end your response with a helpful follow-up question.
"""



class ChatEngine:
    """
    Core chatbot engine combining:
    - Ollama (local LLM, no API cost)
    - RAG (retrieves relevant context before each response)
    - Conversation history (maintains context across turns)
    """

    def __init__(self, model: str = "gemma3:4b", max_history: int = 10):
        self.model = model
        self.max_history = max_history  # sliding window for context
        self.rag = RAGEngine()
        self.ollama = Client()
        self.conversation_history = []
        print(f"🤖 ChatEngine ready with model: {model}")

    def _build_messages(self, user_message: str, context: str) -> list[dict]:
        """Build the message list for Ollama with system prompt + history + RAG context."""

        # System message with today's date injected
        system = SYSTEM_PROMPT.format(date=datetime.now().strftime("%d %B %Y"))

        # Augment system with retrieved context (RAG)
        augmented_system = system + f"""
---
RELEVANT SHOPEASY KNOWLEDGE BASE CONTEXT:
[CRITICAL INSTRUCTION: The following information is from the official ShopEasy catalog and policies. YOU MUST treat any product queries matching this context as ShopEasy-related and fully within your scope.]
{context}
---
Use the above context to give accurate answers. If context is insufficient, acknowledge it."""

        messages = [{"role": "system", "content": augmented_system}]

        # Add conversation history (sliding window — Current session)
        history_window = self.conversation_history[-(self.max_history * 2):]
        messages.extend(history_window)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def chat(self, user_message: str) -> dict:
        """
        Main chat method:
        1. Retrieve relevant context via RAG
        2. Build prompt with history + context
        3. Generate response via Ollama
        4. Save to conversation history
        5. Return response + metadata
        """

        # Step 1: Semantic retrieval
        retrieved_docs = self.rag.retrieve(user_message, top_k=3)
        context = self.rag.format_context(retrieved_docs)

        # Step 2: Build messages
        messages = self._build_messages(user_message, context)

        # Step 3: Generate response (Ollama — runs locally)
        response = self.ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": 0.7,      # balanced creativity
                "top_p": 0.9,
                "num_ctx": 4096,          # context window
            }
        )

        assistant_message = response["message"]["content"]

        # Step 4: Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        # Step 5: Return response + debug metadata
        return {
            "response": assistant_message,
            "retrieved_docs": retrieved_docs,
            "context_used": context,
            "history_length": len(self.conversation_history) // 2,
            "model": self.model
        }

    def stream_chat(self, user_message: str):
        """
        Stream chat method:
        1. Retrieve relevant context via RAG
        2. Build prompt with history + context
        3. Yield metadata block
        4. Generate response via Ollama token by token
        5. Save to conversation history
        6. Yield done signal
        """
        # Step 1: Semantic retrieval
        retrieved_docs = self.rag.retrieve(user_message, top_k=3)
        context = self.rag.format_context(retrieved_docs)

        # Step 2: Build messages
        messages = self._build_messages(user_message, context)

        # Step 3: Yield metadata
        metadata = {
            "type": "metadata",
            "sources": [
                {
                    "category": doc["category"],
                    "similarity": doc["similarity"]
                }
                for doc in retrieved_docs
            ],
            "history_turns": len(self.conversation_history) // 2,
            "model": self.model
        }
        yield json.dumps(metadata) + "\n"

        # Step 4: Stream response (Ollama — runs locally)
        assistant_message = ""
        try:
            for chunk in self.ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                },
                stream=True
            ):
                token = chunk["message"]["content"]
                assistant_message += token
                yield json.dumps({"type": "token", "content": token}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

        # Step 5: Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        # Step 6: Verify and signify completion
        yield json.dumps({"type": "done"}) + "\n"

    def reset_conversation(self):
        """Clear conversation history for a new session."""
        self.conversation_history = []
        return {"status": "Conversation reset successfully"}

    def get_history(self) -> list[dict]:
        """Return current conversation history."""
        return self.conversation_history


engine=ChatEngine()
print(engine.max_history)