import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from openai import embeddings
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app

#TODO:
# Provide System prompt. Goal is to explain LLM that in the user message will be provide rag context that is retrieved
# based on user question and user question and LLM need to answer to user based on provided context
SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user information.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
- Be conversational and helpful in your responses.
- When presenting user information, format it clearly and include relevant details.
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""



def format_user_document(user: dict[str, Any]) -> str:
    pairs = []
    for key, value in user.items():
        pairs.append(f"{key}: {value}")
    return "\n".join(pairs) + "\n\n"


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        user_client = UserClient()
        documents = []
        for user in user_client.search_users():
            documents.append(Document(format_user_document(user)))

        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        batches = (documents[i:i + batch_size] for i in  range(0, len(documents), batch_size))
        tasks = []
        for batch in batches:
            tasks.append(FAISS.afrom_documents(batch, self.embeddings))

        results = await asyncio.gather(*tasks)
        merged = None
        for result in results:
            if merged is None:
                merged = result
            else:
                merged.merge_from(result)
        return merged

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        similarities = self.vectorstore.similarity_search_with_relevance_scores(query=query, k=k, score=score)
        context_parts = []
        for doc, relevance_score in similarities:
            context_parts.append(doc.page_content)
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(augmented_prompt),
        ]
        return self.llm_client.invoke(messages).content


async def main():

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=API_KEY,
        model="text-embedding-3-small-1",
        dimensions = 384
    )
    llm_client = AzureChatOpenAI(
        api_key=API_KEY,
        model="gpt-4o",
        azure_endpoint=DIAL_URL,
        api_version=""
    )
    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            context = await rag.retrieve_context(user_question)
            augmentation = rag.augment_prompt(user_question, context)
            answer = rag.generate_answer(augmentation)
            print(answer)


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce