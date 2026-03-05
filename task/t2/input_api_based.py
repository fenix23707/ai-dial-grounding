from enum import StrEnum
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from openai import BaseModel
from pyasn1_modules.rfc2251 import SearchRequest
from pydantic import SecretStr, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `api_based_grounding.png` to see the flow of app

QUERY_ANALYSIS_PROMPT = """You are a query analysis system that extracts search parameters from user questions about users.

## Available Search Fields:
- **name**: User's first name (e.g., "John", "Mary")
- **surname**: User's last name (e.g., "Smith", "Johnson") 
- **email**: User's email address (e.g., "john@example.com")

## Instructions:
1. Analyze the user's question and identify what they're looking for
2. Extract specific search values mentioned in the query
3. Map them to the appropriate search fields
4. If multiple search criteria are mentioned, include all of them
5. Only extract explicit values - don't infer or assume values not mentioned

## Examples:
- "Who is John?" → name: "John"
- "Find users with surname Smith" → surname: "Smith" 
- "Look for john@example.com" → email: "john@example.com"
- "Find John Smith" → name: "John", surname: "Smith"
- "I need user emails that filled with hiking" → No clear search parameters (return empty list)

## Response Format:
{format_instructions}
"""

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

llm_client = AzureChatOpenAI(
    api_key=API_KEY,
    model="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_version=""
)

user_client = UserClient()

class SearchField (StrEnum):
    name = "name"
    surname = "surname"
    email = "email"

class SearchRequest (BaseModel):
    search_field: SearchField = Field(description="The field to search by.")
    search_value: str = Field(description="The value to search for in the specified field.")

class SearchRequests (BaseModel):
    search_request_parameters: list[SearchRequest] = Field(default_factory=list, description="List of search requests extracted from the user query.")

def retrieve_context(user_question: str) -> list[dict[str, Any]]:
    """Extract search parameters from user query and retrieve matching users."""

    parser = PydanticOutputParser(pydantic_object=SearchRequests)
    messages = [
        SystemMessagePromptTemplate.from_template(QUERY_ANALYSIS_PROMPT),
        HumanMessage(content=user_question)
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())
    search_requests: SearchRequests = (prompt | llm_client | parser).invoke({})
    if search_requests.search_request_parameters:
        requests_dict = {}
        for search_request in search_requests.search_request_parameters:
            requests_dict[search_request.search_field.value] = search_request.search_value

        print(requests_dict)
        return user_client.search_users(**requests_dict)

    else:
        print('No specific search parameters found!')
        return []

def join_context(context: list[dict[str, Any]]) -> str:
    result = ""
    for user in context:
        pairs = []
        for key, value in user.items():
            pairs.append(f"{key}: {value}")
        result += "User:\n  " + "\n  ".join(pairs) + "\n\n"
    return result

def augment_prompt(user_question: str, context: list[dict[str, Any]]) -> str:
    """Combine user query with retrieved context into a formatted prompt."""
    joined_context = join_context(context)
    augmented_prompt = USER_PROMPT.format(context=joined_context, query=user_question)
    print(augmented_prompt)
    return augmented_prompt


def generate_answer(augmented_prompt: str) -> str:
    """Generate final answer using the augmented prompt."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=augmented_prompt)
    ]

    return llm_client.invoke(input= messages).content


def main():
    print("Query samples:")
    print(" - I need user emails that filled with hiking and psychology")
    print(" - Who is John?")
    print(" - Find users with surname Adams")
    print(" - Do we have smbd with name John that love painting?")

    while True:
        user_question = input("> ").strip()
        if not user_question:
            continue

        context = retrieve_context(user_question)
        if context:
            augmented_prompt = augment_prompt(user_question, context)
            answer = generate_answer(augmented_prompt)
            print("Answer:\n" + answer)
        else:
            print("No relevant information found")


if __name__ == "__main__":
    main()


# The problems with API based Grounding approach are:
#   - We need a Pre-Step to figure out what field should be used for search (Takes time)
#   - Values for search should be correct (✅ John -> ❌ Jonh)
#   - Is not so flexible
# Benefits are:
#   - We fetch actual data (new users added and deleted every 5 minutes)
#   - Costs reduce