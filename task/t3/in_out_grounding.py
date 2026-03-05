import asyncio
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import BaseModel, Field

from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountians`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)


SYSTEM_PROMPT = """You are a RAG-powered assistant that searches users by hobbies, performs Named Entity Extraction (NEE), and provides a response in JSON format.

## Your Task:
Given a list of users and a query about hobbies, extract the relevant hobbies from the query, match them to users from the context, and group user IDs by hobby.

## Structure of User message:
- `RAG CONTEXT` — Retrieved documents relevant to the query. Format: list of `userId: userAbout` entries separated by `\\n`. Each entry contains the user's ID and their "about me" text.
- `USER QUESTION` — The user's actual question about finding people with specific hobbies.

## Instructions:
1. Identify the hobbies mentioned or implied in the `USER QUESTION`.
2. For each identified hobby, find all users in `RAG CONTEXT` whose "about me" text explicitly mentions that hobby or a closely related activity.
3. Group matching user IDs by hobby.
4. A single user may appear under multiple hobbies if their "about me" text matches more than one.
5. Only extract explicit values — do not infer or assume hobbies not mentioned in the user's "about me" text.
6. Answer ONLY based on the provided `RAG CONTEXT`. Do not fabricate user IDs or information.
7. If no relevant users are found in `RAG CONTEXT`, return an empty list for that hobby.

## Response Format:
{format_instructions}
"""


USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""

class UserHobby (BaseModel):
    hobby: str = Field(description="User hobby")
    user_ids: list[int] = Field(description="Group of user IDs that have this hobby")

class UserHobbies (BaseModel):
    hobbies: list[UserHobby] = Field(default_factory=list, description="List of user hobbies.")


def format_user_document(user: dict[str, Any]) -> Document:
    return Document(page_content=user['about_me'], id=user['id'])

class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None
        self.parser = PydanticOutputParser(pydantic_object=UserHobbies)
        self.user_client = UserClient()

    async def __aenter__(self):
        print("🔎 Loading all users...")
        user_client = UserClient()
        documents = []
        for user in user_client.search_users():
            documents.append(format_user_document(user))

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
            context_parts.append(f"{doc.id}: '{doc.page_content}'")
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    async def generate_answer(self, augmented_prompt: str) -> UserHobbies:
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        prompt = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=self.parser.get_format_instructions())
        user_hobbies: UserHobbies = await (prompt | self.llm_client | self.parser).ainvoke({})
        return user_hobbies

    async def output_grounding(self, user_hobbies: UserHobbies) -> dict[str, list[dict[str, Any]]]:
        grounded_users: dict[str, list[dict[str, Any]]] = {}
        for user_hobby in user_hobbies.hobbies:
            hobby_users = []
            for user_id in user_hobby.user_ids:
                user_info = await self.user_client.get_user(user_id)
                if user_info:
                    hobby_users.append(user_info)
                else:
                    print(f"⚠️ Warning: User ID {user_id} not found during grounding.")
            grounded_users[user_hobby.hobby] = hobby_users
        return grounded_users

    async def update_vectorstore(self):
        current_users = {user['id']: user for user in self.user_client.search_users()}
        existing_ids = set(int(id) for id in self.vectorstore.index_to_docstore_id.values())
        current_ids = set(current_users.keys())

        deleted_ids = existing_ids - current_ids
        if deleted_ids:
            print(f"🗑️ Removing deleted users: {deleted_ids}")
            self.vectorstore.delete(ids=list(deleted_ids))

        new_ids = current_ids - existing_ids
        if new_ids:
            print(f"➕ Adding new users: {new_ids}")
            new_documents = [format_user_document(current_users[user_id]) for user_id in new_ids]
            await self.vectorstore.aadd_documents(new_documents)

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
        print(" - I need people who love to go to mountians")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break

            await rag.update_vectorstore()
            context = await rag.retrieve_context(user_question)
            augmentation = rag.augment_prompt(user_question, context)
            answer = await rag.generate_answer(augmentation)
            grounded_answer = await rag.output_grounding(answer)
            print(grounded_answer)


asyncio.run(main())
