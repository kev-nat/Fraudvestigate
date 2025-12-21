import os
import re
import yaml
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SQLDatabase

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

load_dotenv()

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

st.set_page_config(page_title=config["system"]["app_name"], page_icon="🕵️‍♂️")
st.title(f"{config['system']['app_name']} 🕵️‍♂️")

def get_llm(json_mode=False):
    model_kwargs = {'response_format': {'type': 'json_object'}} if json_mode else {}
    return ChatOpenAI(
        model=config["llm"]["model_name"],
        temperature=config["llm"]["temperature"],
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs=model_kwargs
    )

# TOOL 1: SQL CHAIN
def init_database(user, password, host, port, database):
    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    llm = get_llm()
    
    def get_context(_):
        schema = db.get_table_info()
        try:
            categories = db.run(f'SELECT DISTINCT category FROM "{config["database"]["table_name"]}" LIMIT 20;')
        except:
            categories = "N/A"
        return config["prompts"]["sql_context_enrichment"].format(schema=schema, categories=categories)

    sql_prompt = ChatPromptTemplate.from_template(config["prompts"]["sql_generation"])
    sql_chain = (
        RunnablePassthrough.assign(schema=get_context)
        | sql_prompt
        | llm
        | StrOutputParser()
    )
    
    response_prompt = ChatPromptTemplate.from_template(config["prompts"]["sql_response"])
    
    def run_query_safe(query):
        try:
            cleaned_query = query.replace("```sql", "").replace("```", "").strip()
            return db.run(cleaned_query)
        except Exception as e:
            return f"Error executing SQL: {e}"

    full_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=get_context,
            response=lambda vars: run_query_safe(vars["query"])
        )
        | response_prompt
        | llm
        | StrOutputParser()
    )
    
    return full_chain

# TOOL 2: RAG AGENT
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    retry_count: int

@st.cache_resource
def init_rag_agent():
    embeddings = OllamaEmbeddings(model=config["embeddings"]["model_name"])
    
    vectorstore = Chroma(
        persist_directory=config["files"]["chroma_db_path"],
        embedding_function=embeddings,
        collection_name="fraud_rag_db" 
    )
    retriever = vectorstore.as_retriever()
    
    llm = get_llm()
    grader_llm = get_llm(json_mode=True)
    
    retrieval_grader = PromptTemplate.from_template(config["prompts"]["rag_retrieval_grader"]) | grader_llm | JsonOutputParser()
    hallucination_grader = PromptTemplate.from_template(config["prompts"]["rag_hallucination_grader"]) | grader_llm | JsonOutputParser()
    answer_grader = PromptTemplate.from_template(config["prompts"]["rag_answer_grader"]) | grader_llm | JsonOutputParser()
    question_rewriter = PromptTemplate.from_template(config["prompts"]["rag_question_rewriter"]) | llm | StrOutputParser()
    rag_chain = PromptTemplate.from_template(config["prompts"]["rag_generation"]) | llm | StrOutputParser()

    # NODES
    def retrieve(state):
        return {
            "documents": retriever.invoke(state["question"]), 
            "question": state["question"],
            "retry_count": state.get("retry_count", 0)
        }

    def grade_documents(state):
        filtered_docs = []
        for d in state["documents"]:
            score = retrieval_grader.invoke({"question": state["question"], "document": d.page_content})
            if score['score'].lower() == "yes":
                filtered_docs.append(d)
        return {"documents": filtered_docs, "question": state["question"]}

    def transform_query(state):
        better_question = question_rewriter.invoke({"question": state["question"]})
        # Increment retry count to eventually stop the loop
        return {
            "documents": state["documents"], 
            "question": better_question, 
            "retry_count": state.get("retry_count", 0) + 1 
        }

    def generate(state):
        generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
        generation_clean = re.sub(r'<think>.*?</think>', '', generation, flags=re.DOTALL).strip()
        return {"generation": generation_clean}

    def decide_to_generate(state):
        # CIRCUIT BREAKER: If we looped 3 times, force generation to prevent crash)
        if state.get("retry_count", 0) >= 3:
            return "generate"
            
        if not state["documents"]:
            return "transform_query"
        return "generate"

    def grade_generation(state):
        # If we forced generation due to max retries, skip strict grading to avoid infinite loops
        if state.get("retry_count", 0) >= 3:
            return "useful"

        hallucination_score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
        if hallucination_score['score'] == "no":
            return "not supported" 
        
        answer_score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
        if answer_score['score'] == "no":
            return "not useful" 
            
        return "useful"

    # GRAPH
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {
        "transform_query": "transform_query",
        "generate": "generate"
    })
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges("generate", grade_generation, {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query"
    })

    return workflow.compile()

# MASTER ROUTER
def route_query(user_query, chat_history):
    llm = get_llm(json_mode=True)
    router_prompt = PromptTemplate.from_template(
        config["prompts"]["router_system"] + 
        "\n\nUser Question: {question}\n" +
        "Return JSON with a single key 'tool' with value 'sql_analysis_tool' or 'policy_rag_tool'."
    )
    chain = router_prompt | llm | JsonOutputParser()
    result = chain.invoke({"question": user_query})
    return result.get("tool", "policy_rag_tool")

# UI - STREAMLIT
with st.sidebar:
    st.subheader("Database Connection")
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="5432", key="Port")
    st.text_input("User", value="postgres", key="User")
    st.text_input("Password", type="password", value="111", key="Password")
    st.text_input("Database", value="postgres", key="Database")
    
    if st.button("Connect"):
        try:
            db = init_database(
                st.session_state["User"], st.session_state["Password"],
                st.session_state["Host"], st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected!")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage("Hello! I am your Fraud Analysis Assistant. I can analyze your transaction database or explain fraud policies and charts.")
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"): st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"): st.markdown(message.content)

user_query = st.chat_input("Ask about fraud stats or policies...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"): st.markdown(user_query)
    
    with st.chat_message("AI"):
        try:
            tool_choice = route_query(user_query, st.session_state.chat_history)
            
            if tool_choice == "sql_analysis_tool":
                if "db" not in st.session_state:
                    st.error("Please connect to the database first for SQL analysis.")
                else:
                    # UX: Spinner
                    with st.spinner("Running SQL Query..."):
                        sql_chain = get_sql_chain(st.session_state.db)
                        response_text = sql_chain.invoke({"question": user_query})
                    
                    st.markdown(response_text)
                    st.session_state.chat_history.append(AIMessage(content=response_text))
            
            elif tool_choice == "policy_rag_tool":
                # UX: Dynamic Status Bar
                status_container = st.status("Processing...", expanded=True)
                rag_app = init_rag_agent()
                
                inputs = {"question": user_query, "retry_count": 0}
                final_generation = ""
                
                for output in rag_app.stream(inputs):
                    for key, value in output.items():
                        # Update status label based on current step
                        if key == "retrieve":
                            status_container.update(label="Retrieving Documents...", state="running")
                        elif key == "grade_documents":
                            status_container.update(label="Grading Relevance...", state="running")
                        elif key == "transform_query":
                            status_container.update(label="Refining Search Query...", state="running")
                        elif key == "generate":
                            status_container.update(label="Generating Answer...", state="running")
                            if "generation" in value:
                                final_generation = value["generation"]
                
                status_container.update(label="Finished!", state="complete", expanded=False)
                st.markdown(final_generation)
                st.session_state.chat_history.append(AIMessage(content=final_generation))
                
        except Exception as e:
            st.error(f"An error occurred: {e}")