import os
import getpass
import json
import operator
# import joblib # No longer needed
from typing import List, Annotated
from typing_extensions import TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import END, StateGraph

# An agentic workflow that checks for hallucinations and self-corrects, uses LangGraph for a Graph Node workflow


ENABLE_CACHE = True 
CACHE_DIR = "./agentic-exp/cache"
VECTORSTORE_PERSIST_PATH = os.path.join(CACHE_DIR, "vectorstore.parquet")
VECTORSTORE_SERIALIZER = "parquet"


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter {var}: ")

_set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# --- LLM Init ---

#local_llm = "qwen3:4b" # Make sure Ollama is running and this model is pulled
#local_llm = "qwen3:32b" # Bigger model, demo purposes only
local_llm = "qwen3:30b" # Big MoE, faster responses, slightly worse than 32b quality, demo purposes only
# local_llm = "gemma3:27b"

try:
    
    llm = ChatOllama(model=local_llm, temperature=0, base_url="http://localhost:11434")
    llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json", base_url="http://localhost:11434")
    # Test connection
    print("Attempting to connect to Ollama...")
    llm.invoke("hello world :3")
    print("Ollama connection successful.")
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    exit()


# --- Search Tool ---
web_search_tool = TavilySearchResults(k=3)


# --- VectorStore Setup ---
print("VectorStore being set up...")
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://www.coingecko.com/en/coins/bitcoin",
    "https://www.coingecko.com/en/coins/ethereum",
]

# --- Init Embeddings ---
embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

# --- Load or Create Vectorstore ---
vectorstore = None
if ENABLE_CACHE:
    os.makedirs(CACHE_DIR, exist_ok=True) # Ensure cache directory exists
    # Check if the specific persist file exists
    if os.path.exists(VECTORSTORE_PERSIST_PATH):
        try:
            print(f"--- Loading cached vector store from {VECTORSTORE_PERSIST_PATH} ---")
            # Load by re-instantiating with the persist_path and serializer
            vectorstore = SKLearnVectorStore(
                embedding=embedding_model,
                persist_path=VECTORSTORE_PERSIST_PATH,
                serializer=VECTORSTORE_SERIALIZER
            )
            print("--- Cached vector store loaded successfully ---")
        except ImportError:
             print(f"--- Error loading cached vector store: Serializer '{VECTORSTORE_SERIALIZER}' requires extra dependencies. ---")
             vectorstore = None # Reset on error
        except Exception as e:
            print(f"--- Error loading cached vector store: {e}. Recomputing... ---")
            vectorstore = None # Reset on error
    else:
        print(f"--- Cache file {VECTORSTORE_PERSIST_PATH} not found. Will compute. ---")
        vectorstore = None # Ensure vectorstore is None if cache file missing

# If cache is disabled or loading failed/not found
if vectorstore is None:
    print("--- Computing or Recomputing Vector Store ---")
    try:
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
    except Exception as e:
        print(f"Error loading documents from URLs: {e}")
        print("Please check your internet connection and whether URLs are valid")
        exit()

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Adding to vectorstore
    try:
        print("--- Creating new vector store and computing embeddings ---")
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=embedding_model, # Use the initialized embedding model
            persist_path=VECTORSTORE_PERSIST_PATH if ENABLE_CACHE else None, # Pass path if caching
            serializer=VECTORSTORE_SERIALIZER if ENABLE_CACHE else 'json', # Pass serializer if caching
        )
        if ENABLE_CACHE:
            try:
                print(f"--- Persisting vector store to cache file: {VECTORSTORE_PERSIST_PATH} ---")
                # Explicitly call persist after creation
                vectorstore.persist()
                print("--- Vector store persisted successfully ---")
            except ImportError:
                 print(f"--- Error persisting vector store: Serializer '{VECTORSTORE_SERIALIZER}' requires extra dependencies. Please install 'pandas' and 'pyarrow'. ---")
                 print("--- Run: uv pip install pandas pyarrow ---")
                 # Continue without saving if dependencies missing
            except Exception as e:
                print(f"--- Error persisting vector store to cache: {e} ---")
        print("--- VECTORSTORE SETUP COMPLETE (New computation) ---")
    except ImportError:
        print(f"--- Error creating vector store: Serializer '{VECTORSTORE_SERIALIZER}' requires extra dependencies for caching. Please install 'pandas' and 'pyarrow' or set ENABLE_CACHE=False. ---")
        print("--- Run: uv pip install pandas pyarrow ---")
        exit()
    except Exception as e:
        print(f"Error setting up vector store with NomicEmbeddings: {e}")
        # (Keep existing Nomic-specific error messages)
        print("Ensure Nomic embedding dependencies are installed ('pip install langchain-nomic')")
        print("And that the local inference mode is set up correctly if required by Nomic.")
        exit()
else:
     print("--- VECTORSTORE SETUP COMPLETE (Loaded from cache) ---")


# --- Retriever --- # Now uses the loaded or newly created vectorstore
retriever = vectorstore.as_retriever(k=3)
# print("--- VECTORSTORE SETUP COMPLETE ---") # Removed redundant print


# --- Component Prompts ---


router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""


doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

doc_grader_prompt = """Here is the retrieved document: 

 {document} 

 Here is the user question: 

 {question}.

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""


rag_prompt = """You are an assistant for question-answering tasks.

Here is the context to use to answer the question:

{context}

Think carefully about the above context.

Now, review the user question:

{question}

Provide an answer to this questions using only the above context.

Use three sentences maximum and keep the answer concise.

Answer:"""


hallucination_grader_instructions = """

You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS.

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

hallucination_grader_prompt = """FACTS: 

 {documents} 

 STUDENT ANSWER: {generation}.

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

# Answer Grader
answer_grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score.

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

answer_grader_prompt = """QUESTION: 

 {question} 

 STUDENT ANSWER: {generation}.

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""


# --- Helper Function ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question: str  # User question
    generation: str | None  # LLM generation (can be None initially)
    web_search: str  # Binary decision 'Yes'/'No' to run web search
    documents: List[Document]  # List of retrieved documents
    loop_step: Annotated[int, operator.add] # Internal counter for cycles


def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents from vectorstore

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: New key added to state, documents, that contains retrieved documents
    """
    print("---Node: Retrieve document---")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} documents.")
    return {"documents": documents}


def generate(state: GraphState) -> GraphState:
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: New key added to state, generation, that contains LLM generation
    """
    print("---Node: Generate---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0) # Initialize loop_step if not present

    if not documents:
         print("---Node: Generate (No documents found, skipping generation)---")
         return {"generation": None, "loop_step": loop_step + 1} # Handle case with no documents

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    print("Generated answer.")
    return {"generation": generation.content, "loop_step": loop_step + 1}


def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: Filtered out irrelevant documents and updated web_search state
    """
    print("---Node: Grade Documents---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No" # Default to No web search until we find that document is irrelevant to query
    if not documents:
        print("---Grade: No documents to grade. Flagging for web search.")
        web_search = "Yes"
        return {"documents": [], "web_search": web_search}

    for d in documents:
        try:
            doc_grader_prompt_formatted = doc_grader_prompt.format(
                document=d.page_content, question=question
            )
            result = llm_json_mode.invoke(
                [SystemMessage(content=doc_grader_instructions)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )
            grade_data = json.loads(result.content)
            grade = grade_data.get("binary_score", "no").lower() 
            if grade == "yes":
                print("---GRADE: Document is relevant to query.---")
                filtered_docs.append(d)
            else:
                print("---GRADE: Document is not relevant to query.---")
                # default to web search if *any* doc is irrelevant
                web_search = "Yes"
        except json.JSONDecodeError:
            print(f"---GRADE: ERROR - Invalid JSON received from LLM: {result.content}")
            print("---GRADE: Assuming document was NOT RELEVANT due to error. Flagging for web search.")
            web_search = "Yes" # default to web search on error
        except Exception as e:
            print(f"---GRADE: ERROR - An unexpected error occurred during grading: {e}")
            print("---GRADE: Assuming document was NOT RELEVANT due to error. Flagging for web search.")
            web_search = "Yes" # default to web search on error

    if not filtered_docs and web_search == "No":
        # default to web search if invalid json or empty list
        print("---GRADE: All documents were filtered out or an issue occurred. Flagging for web search.")
        web_search = "Yes"


    return {"documents": filtered_docs, "web_search": web_search}


def web_search_node(state: GraphState) -> GraphState:
    """
    Web search based based on the question

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: Appended web results to documents list
    """
    print("---Node: WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", []) # Use state.get

    # Ensure documents is a list initially if it wasn't passed or was None
    if not isinstance(documents, list):
        print(f"Warning: 'documents' was not a list ({type(documents)}). Initializing as empty list.")
        documents = []

    try:
        print(f"Performing web search for: {question}")
        # The Tavily tool returns a list of search result dictionaries.
        search_results = web_search_tool.invoke({"query": question})
        # Add print statement to inspect raw output
        print(f"---RAW TAVILY OUTPUT: {search_results} (Type: {type(search_results)})---")

        if search_results and isinstance(search_results, list):
            # Filter out potential non-dictionary items and items without 'content'
            valid_contents = [str(d.get('content', '')) for d in search_results if isinstance(d, dict) and 'content' in d]
            if valid_contents:
                web_content = "\n\n".join(valid_contents)
                web_results_doc = Document(page_content=web_content, metadata={"source": "web_search"})
                documents.append(web_results_doc)
                print(f"Added {len(valid_contents)} web results.")
            else:
                print("---WEB SEARCH: No valid content found in results.---")
        else:
            print("---WEB SEARCH: No results found or unexpected format.---")

    except Exception as e:
        print(f"---WEB SEARCH: Error during web search: {e}")

    return {"documents": documents}


# --- Graph Edges ---

def route_question(state: GraphState) -> str:
    """
    Route question to web search or RAG

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Next node to call ('web_search_node' or 'retrieve')
    """
    print("---Edge: Routing Query---")
    question = state["question"]
    try:
        route_question_llm_call = llm_json_mode.invoke(
            [SystemMessage(content=router_instructions)]
            + [HumanMessage(content=question)]
        )
        route_data = json.loads(route_question_llm_call.content)
        source = route_data.get("datasource", "vectorstore").lower() # Default to vectorstore
        if source == "websearch":
            print("---ROUTE: Question requires WEB SEARCH ---")
            return "web_search_node" # Route to the web search node function
        elif source == "vectorstore":
            print("---ROUTE: Question requires VECTORSTORE (RAG) ---")
            return "retrieve"
        else:
            print(f"---ROUTE: Unexpected datasource '{source}'. Defaulting to VECTORSTORE.---")
            return "retrieve"
    except json.JSONDecodeError:
        print(f"---ROUTE: ERROR - Invalid JSON received from LLM: {route_question_llm_call.content}")
        print("---ROUTE: Defaulting to VECTORSTORE due to routing error.")
        return "retrieve"
    except Exception as e:
        print(f"---ROUTE: ERROR - An unexpected error occurred during routing: {e}")
        print("---ROUTE: Defaulting to VECTORSTORE due to routing error.")
        return "retrieve"


def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Binary decision for next node to call ('web_search_node' or 'generate')
    """
    print("---Edge: Decide to answer with web_search tool or relevant documents---")
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # If web search was flagged needed (e.g., irrelevant docs found)
        print("---DECISION: Web search needed based on document grades. Routing to WEB SEARCH.---")
        return "web_search_node" # Route to the web search node function
    elif not filtered_documents:
         # If no relevant documents were found after grading, but web_search wasn't explicitly 'Yes'
         print("---DECISION: No relevant documents found. Routing to WEB SEARCH.---")
         return "web_search_node"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: Relevant documents found. Proceeding to GENERATE.---")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """
    Determines whether the generation is grounded in the documents and answers question.
    Implements self-correction logic.

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Decision for next node call:
             - "useful" (generation is good, finish)
             - "not supported" (hallucination, regenerate)
             - "not useful" (doesn't answer question, add web search)
             - "max_retries" (too many loops, finish)
             - "__end__" (special value to end immediately, e.g. on error)
    """
    question = state["question"]
    documents = state["documents"]
    generation = state.get("generation") # Generation might be None if skipped
    loop_step = state.get("loop_step", 0)
    max_retries = 3 # Define max retries here or pass in state

    # Handle case where generation was skipped (e.g., no documents)
    if generation is None:
        print("---Grade: No generation produced (likely no documents). Adding web search.")
        # Decide if we should end or try web search
        if loop_step <= max_retries:
            return "web_search_node" # Try web search if no generation happened
        else:
             print("---Grade: Max retries reached even without generation. Ending.")
             return "__end__" # End if max retries hit

    # 1. Check Hallucinations
    print("---Hallucination Grading---")
    try:
        hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
            documents=format_docs(documents), generation=generation
        )
        result_hallucination = llm_json_mode.invoke(
            [SystemMessage(content=hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        grade_hallucination_data = json.loads(result_hallucination.content)
        grade_hallucination = grade_hallucination_data.get("binary_score", "no").lower() # Default no
        explanation_hallucination = grade_hallucination_data.get("explanation", "N/A")
        print(f"Hallucination Grade: {grade_hallucination.upper()}. Explanation: {explanation_hallucination}")

        if grade_hallucination == "yes":
            # 2. Check if Generation Answers the Question
            print("---Check Output Reliability---")
            try:
                answer_grader_prompt_formatted = answer_grader_prompt.format(
                    question=question, generation=generation
                )
                result_answer = llm_json_mode.invoke(
                    [SystemMessage(content=answer_grader_instructions)]
                    + [HumanMessage(content=answer_grader_prompt_formatted)]
                )
                grade_answer_data = json.loads(result_answer.content)
                grade_answer = grade_answer_data.get("binary_score", "no").lower() # Default no
                explanation_answer = grade_answer_data.get("explanation", "N/A")
                print(f"Answer Relevance Grade: {grade_answer.upper()}. Explanation: {explanation_answer}")

                if grade_answer == "yes":
                    print("---DECISION: Generation IS grounded and answers question. END.---")
                    return "__end__" # Use END signal from langgraph
                # Generation is grounded but doesn't answer question
                elif loop_step < max_retries:
                    print(f"---DECISION: Generation does NOT answer question (Attempt {loop_step+1}/{max_retries}). Adding WEB SEARCH.---")
                    return "web_search_node" # Add web search context and try again
                else:
                    print(f"---DECISION: Max retries ({max_retries}) reached. Generation doesn't answer question. END.---")
                    return "__end__" # Max retries reached
            except json.JSONDecodeError:
                print(f"---GRADE ANSWER: ERROR - Invalid JSON: {result_answer.content}")
                print("---GRADE ANSWER: Assuming 'not useful' due to error.")
                if loop_step < max_retries: return "web_search_node"
                else: return "__end__"
            except Exception as e:
                print(f"---GRADE ANSWER: ERROR - Unexpected error: {e}")
                print("---GRADE ANSWER: Assuming 'not useful' due to error.")
                if loop_step < max_retries: return "web_search_node"
                else: return "__end__"
        # Generation has hallucination
        elif loop_step < max_retries:
            print(f"---DECISION: Generation IS NOT grounded (Attempt {loop_step+1}/{max_retries}). REGENERATING.---")
            # Implicitly regenerate by returning to 'generate'.
            # We *don't* necessarily need web search yet, maybe RAG just failed.
            # The 'generate' node will use the current documents (which might include web results if added previously).
            return "generate" # Re-try generation with existing docs
        else:
            print(f"---DECISION: Max retries ({max_retries}) reached. Generation has hallucination. END.---")
            return "__end__" # Max retries reached

    except json.JSONDecodeError:
        print(f"---CHECK HALLUCINATION: ERROR - Invalid JSON: {result_hallucination.content}")
        print("---CHECK HALLUCINATION: Assuming 'not supported' due to error.")
        if loop_step < max_retries: return "generate" # Retry generation on error
        else: return "__end__"
    except Exception as e:
        print(f"---CHECK HALLUCINATION: ERROR - Unexpected error: {e}")
        print("---CHECK HALLUCINATION: Assuming 'not supported' due to error.")
        if loop_step < max_retries: return "generate" # Retry generation on error
        else: return "__end__"


# --- Build Graph ---
print("--- BUILDING GRAPH ---")
workflow = StateGraph(GraphState)

# Define the nodes
# Use the function name strings as node identifiers
workflow.add_node("web_search_node", web_search_node) # web search node
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generate

# Set entry point
workflow.set_conditional_entry_point(
    route_question, # Function to determine the first node
    {
        # Map return values of route_question to node names
        "web_search_node": "web_search_node",
        "retrieve": "retrieve",
    },
)

# Add edges connecting nodes
workflow.add_edge("retrieve", "grade_documents") # After retrieving, grade the documents

workflow.add_conditional_edges(
    "grade_documents", # Source node
    decide_to_generate, # Function to decide the next step
    {
        # Map return values to node names
        "web_search_node": "web_search_node", # If web search is needed, go to web search
        "generate": "generate", # If documents are relevant, generate answer
    },
)

workflow.add_conditional_edges(
    "generate", # Source node
    grade_generation_v_documents_and_question, # Function to grade the generation
    {
        # Map return values to node names or END
        "__end__": END, # If generation is good or max retries hit, end.
        "generate": "generate", # If hallucinated, try generating again
        "web_search_node": "web_search_node", # If not useful, add web search and regenerate
    }
)

# Edge from web search back to generation - always try to generate after web search
workflow.add_edge("web_search_node", "generate")


# Compile the graph
try:
    graph = workflow.compile()
    print("--- GRAPH COMPILED SUCCESSFULLY ---")

    # # Optional: Render and save the graph visualization
    # try:
    #     graph_png = graph.get_graph().draw_mermaid_png()
    #     with open("agentic-exp/graph.png", "wb") as f:
    #         f.write(graph_png)
    #     print("Graph visualization saved to agentic-exp/graph.png")
    # except Exception as e:
    #     print(f"Could not generate graph visualization: {e}")

except Exception as e:
    print(f"Error compiling graph: {e}")
    exit()


# --- Run the Graph ---
if __name__ == "__main__":
    print("\n" + "="*30 + " RUNNING GRAPH " + "="*30 + "\n")

    # Example 1: Question likely answered by vector store
    print("\n--- Example 1: Agent Memory ---")
    inputs1 = {"question": "What are the types of agent memory? /no_think"}
    current_state_1 = None # Store the latest state before END
    try:
        for event in graph.stream(inputs1, {"recursion_limit": 50}):
            last_key = list(event.keys())[-1]
            print(f"\n--- Output ({last_key}) ---")
            print(event[last_key])
            if END not in event:
                current_state_1 = event[last_key]
            else:
                print("\n--- Graph Reached END --- ")
                break 

        print("\n--- Example 1 Final State --- ")
        if current_state_1:
             # Access data from the last *valid* state before END
             print(f"Question: {inputs1.get('question')}") # Get question from original input
             print(f"Final Generation: {current_state_1.get('generation')}")
        else:
             print("Graph did not complete successfully or END was the first state.")
    except Exception as e:
        print(f"\n--- Error during Example 1 execution: {e} ---")


    print("\n" + "="*30 + "\n")

    # Example 2: Question likely requiring web search
    print("\n--- Example 2: Web Search Question ---")
    inputs2 = {"question": "What big movements happened in the stock market over the last week? /no_think"}
    current_state_2 = None # Store the latest state before END
    try:
        for event in graph.stream(inputs2, {"recursion_limit": 50}):
            last_key = list(event.keys())[-1]
            print(f"\n--- Step Output ({last_key}) ---")
            print(event[last_key])
            # Capture state just before END
            if END not in event:
                current_state_2 = event[last_key]
            else:
                print("\n--- Graph Reached END --- ")
                break 

        print("\n--- Example 2 Final State --- ")
        if current_state_2:
            print(f"Question: {inputs2.get('question')}") # Get question from original input
            print(f"Final Generation: {current_state_2.get('generation')}")
        else:
            print("Graph did not complete successfully or END was the first state.")
    except Exception as e:
        print(f"\n--- Error during Example 2 execution: {e} ---")

    print("\n" + "="*30 + "\n")

    # Example 3: Crypto price question
    print("\n--- Example 3: Crypto Price Question ---")
    inputs3 = {"question": "What movements happened to the Bitcoin and Ethereum price over the last week? /no_think"}
    current_state_3 = None # Store the latest state before END
    try:
        for event in graph.stream(inputs3, {"recursion_limit": 50}):
            last_key = list(event.keys())[-1]
            print(f"\n--- Step Output ({last_key}) ---")
            print(event[last_key])
            # Capture state just before END
            if END not in event:
                current_state_3 = event[last_key]
            else:
                print("\n--- Graph Reached END --- ")
                break 

        print("\n--- Example 3 Final State --- ")
        if current_state_3:
            print(f"Question: {inputs3.get('question')}") # Get question from original input
            print(f"Final Generation: {current_state_3.get('generation')}")
        else:
            print("Graph did not complete successfully or END was the first state.")
    except Exception as e:
        print(f"\n--- Error during Example 3 execution: {e} ---")

    print("\n" + "="*30 + " Graph executed successfully. " + "="*30 + "\n")
