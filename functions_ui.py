import streamlit as st
import os
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_to_dict, messages_from_dict, Document
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import Chroma
import chromadb
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
import pandas as pd
from langchain.document_loaders import CSVLoader 
from langchain.indexes import VectorstoreIndexCreator 
from langchain.chains import RetrievalQA 
from langchain.llms import OpenAI 
from langchain_community.document_loaders import DataFrameLoader
import constant

os.environ["OPENAI_API_KEY"] = constant.OPENAI_API_KEY
embedding_function = OpenAIEmbeddings()

########################################################################################
# DATA ANALYST PART
########################################################################################
def call_csv_db(embedding_function):
    #collection_name_1 = st.text_input("What is the collection name?: ", key="da_collection")
    collection_name_1 = "csv2"
    csv_db = Chroma(persist_directory=f"./chroma_db/{collection_name_1}", embedding_function=embedding_function)
    
    retriever = csv_db.as_retriever()
    return csv_db, retriever

def DataAnalystRun():
    temp = st.slider("What is the temperature?: ", value=0.0, step=0.1, max_value=1.0, min_value=0.0, key="da_temp")
    st.write("Temperature:", temp)
    llm = OpenAI(temperature = temp)
    embedding_function = OpenAIEmbeddings()

    csv_db, retriever = call_csv_db(embedding_function)

    template = """You are a virtual mechanical engineer and you have academic knowledge about additive manufacturing. You refer excel file for the academic knowledge. Excel file is has information in the each columns as .xlsx format that columns names are as sequiential as "Makale No", "Makale Adı", "DOI", "Proses", "Alt-Proses", "Malzeme", "İçerik", "Hedef", "Girdi", "Çıktı", "ML Model", "Veri", "Veri Boyutu"
    Answer the question based only on the following context:
    {context}

    Question: {question}

    From the excel file, answer which data can be more beneficial for the machine learning model about the Question.
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("What is your question?: ", key="da_question")
    
    # RESPONSE
    st.markdown("Answer")
    result = chain.invoke(query)
    st.write(result)

    # CONTEXT
    docs = csv_db.similarity_search(query)
    st.markdown("Context")
    context = docs[0].page_content
    st.write(context)

    # ESSAY NAME
    st.markdown("Essay Name")
    essay_name = docs[0].metadata.get("Makale Adı")
    st.write(essay_name)

    # ESSAY NO
    st.markdown("Essay No")
    essay_no = docs[0].metadata.get("Makale No")
    st.write(essay_no)

    # Convert result to DataFrame
    result_df = pd.DataFrame({
        'question': [query],
        'answer': [result],
        'context': [context],
        'essay_no': [essay_no],
        'essay_name': [essay_name]
    })

    return result_df
    

#............................................................................
# DATA ANALYST EVAULATE PART
#............................................................................
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
)

def DA_Eval_Run(result_df):
    # Inference
    answer = result_df['answer'].values.tolist()
    question = result_df['question'].values.tolist()
    contexts = result_df['context'].values.tolist()
    ground_truths = result_df['ground_truth'].values.tolist()

    data = {
        "question": question,
        "answer": answer,
        "contexts": [contexts],
        "ground_truth": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    # Evaluation ve diğer işlemler buraya taşınabilir
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_similarity,
        ],
    )

    df = result.to_pandas()
    st.markdown("Evaluation Result")
    st.write(df.head())


########################################################################################
# RESEARCHER PART
########################################################################################

def ResearcherRun(essay_no):
    DOCUMENTS_DIR = f"D:/EgeAMAI/Documents/essay_{essay_no}.pdf"
    loader = PyPDFLoader(DOCUMENTS_DIR)
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    #print('Split docs:', splits)
    
    # Create and store the embeddings in ChromaDB
    embedding_function = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(splits, embedding=embedding_function, 
                                        persist_directory=f"./chroma_db/researcher_{essay_no}")

    st.success("Researcher ingestion complete...")


########################################################################################
# ML ENGINEER PART
########################################################################################
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_to_dict, messages_from_dict, Document
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import Chroma
import chromadb
from operator import itemgetter
import pandas as pd
import constant

# get vector store
def call_vectordb_ml(embedding_function, essay_no):
    #collection_name_1 = input("What is the collection name: ")
    vector_db = Chroma(persist_directory=f"./chroma_db/researcher_{essay_no}", embedding_function=embedding_function)
    
    retriever = vector_db.as_retriever(lambda_val=0.025, k=7, filter=None)
    return retriever

def prompt_templates():
    #CONDENSE_QUESTION_PROMPT
    _template = """
    [INST] 
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a ChromaDB index. This query will be used to retrieve documents with additional context.
    You are a virtual machine learning engineer. You have academic knowledge about additive manufacturing. Answer the question based only on the following context:
    
    Now, with those examples, here is the actual chat history and input question.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    [/INST]
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    #######################################################################
    #ANSWER_PROMPT
    template = """
    [INST]
    Answer the question based only on the following context:
    {context}

    Question: {question}
    [/INST]
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    #######################################################################
    #DEFAULT_DOCUMENT_PROMPT
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    return CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT


CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompt_templates()

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
  ):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def create_rag(llm, retriever,
              CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT):
    # Instantiate ConversationBufferMemory
    memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
    )

    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm,
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
        "question": itemgetter("question"),
        "context": final_inputs["context"]
    }
    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    return final_chain, memory

def call_conversational_rag(question, chain, memory):
    # Prepare the input for the RAG model
    inputs = {"question": question}

    # Invoke the RAG model to get an answer
    result = chain.invoke(inputs)

    # Save the current question and its answer to memory for future context
    memory.save_context(inputs, {"answer": result["answer"]})

    # Return the result
    return result

def main(question, llm, retriever):
    CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompt_templates()

    final_chain, memory = create_rag(llm, retriever,
              CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT)

    result= call_conversational_rag(question, final_chain, memory)

    # Convert result to DataFrame
    result_df = pd.DataFrame({
        'question': [question],
        'answer': [result['answer']],
        'contexts': [result['context']]
    })

    print(result)
    return result, result_df


def MLEngineerRun(essay_no):
    llm = OpenAI(temperature =0.4)
    embedding_function = OpenAIEmbeddings()

    retriever = call_vectordb_ml(embedding_function, essay_no)

    question = """
    Give me all ML works with model works, model error rates, and model accuracy rates. Make a pandas dataframe with these data.
    First column called "Model Name", second column called "Model Error Rate", third column called "Model Accuracy Rate".
    """
    
    result, result_df = main(question, llm, retriever)

    st.markdown("Answer")
    st.write(result['answer'])

    st.markdown("Context")
    st.write(result['context'])

    st.markdown("Answer Dataframe")
    st.write(result_df.head())


########################################################################################
# AM ENGINEER PART
########################################################################################
# get vector store
def call_vectordb(embedding_function):
    #collection_name_1 = st.text_input("What is the collection name: ")
    collection_name_1 = "test2"
    vector_db = Chroma(persist_directory=f"./chroma_db/{collection_name_1}", embedding_function=embedding_function)
    
    retriever = vector_db.as_retriever(lambda_val=0.025, k=7, filter=None)
    return retriever


def prompt_templates():
    #CONDENSE_QUESTION_PROMPT
    _template = """
    [INST]
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a FAISS index. This query will be used to retrieve documents with additional context.

    Let me share a couple examples that will be important.

    If you do not see any chat history, you MUST return the "Follow Up Input" as is:

    ```
    Chat History:

    Follow Up Input: What features can I add as an input of the machine learning model for the additive manufacturing balling situations?
    Standalone Question:
    What features can I add as an input of the machine learning model for the additive manufacturing balling situations?
    ```

    If this is the second question onwards, you should properly rephrase the question like this:

    ```
    Chat History:
    Human: What features can I add as an input of the machine learning model for the additive manufacturing balling situations?
    AI:
    There are several features that can be added as inputs to a machine learning model for predicting balling situations in additive manufacturing. Some of these features include:

    1. Process parameters: These include variables such as feed rate, gas flow rate, and power output, which can affect the formation of beads during the wire arc additive manufacturing process.
    2. Material properties: The physical and chemical properties of the material being used, such as melting point, density, and viscosity, can also impact the formation of beads.
    3. Beam deflection: The deflection of the beam during the wire arc additive manufacturing process can affect the formation of beads.
    4. Arc length: The length of the arc during the wire arc additive manufacturing process can also impact the formation of beads.
    5. Temperature: The temperature of the wire and the workpiece during the wire arc additive manufacturing process can affect the formation of beads.
    6. Wire diameter: The diameter of the wire being used during the wire arc additive manufacturing process can also impact the formation of beads.
    7. Gas pressure: The pressure of the gas used during the wire arc additive manufacturing process can affect the formation of beads.
    8. Feedback control signals: Feedback control signals from sensors placed on the workpiece or the wire can provide information about the formation of beads and can be used as inputs to the machine learning model.

    Now, with those examples, here is the actual chat history and input question.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    [/INST]
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    #######################################################################
    #ANSWER_PROMPT
    template = """
    [INST]
    Answer the question based only on the following context:
    {context}

    Question: {question}
    [/INST]
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    #######################################################################
    #DEFAULT_DOCUMENT_PROMPT
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    return CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT


CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompt_templates()

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
  ):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def create_rag(llm, retriever,
              CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT):
    # Instantiate ConversationBufferMemory
    memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
    )

    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm,
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
        "question": itemgetter("question"),
        "context": final_inputs["context"]
    }
    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    return final_chain, memory

def call_conversational_rag(question, chain, memory):
    # Prepare the input for the RAG model
    inputs = {"question": question}

    # Invoke the RAG model to get an answer
    result = chain.invoke(inputs)

    # Save the current question and its answer to memory for future context
    memory.save_context(inputs, {"answer": result["answer"]})

    # Return the result
    return result

retriever = call_vectordb(embedding_function)

def main(question, llm, retriever):
    CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompt_templates()

    final_chain, memory = create_rag(llm, retriever,
              CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT)

    result= call_conversational_rag(question, final_chain, memory)

    # Convert result to DataFrame
    result_df = pd.DataFrame({
        'question': [question],
        'answer': [result['answer']],
        'contexts': [result['context']]
    })

    print(result)
    return result, result_df

def AMEngineerRun():
    temp = st.slider("What is the temperature?: ", value=0.0, step=0.1, max_value=1.0, min_value=0.0, key="am_temp")
    st.write("Temperature:", temp)
    llm = OpenAI(temperature = temp)
    embedding_function = OpenAIEmbeddings()

    retriever = call_vectordb(embedding_function)

    question = st.text_input("What is your question?: ", key="am_question")

    if st.button("Get Answer"):
        CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompt_templates()

        final_chain, memory = create_rag(llm, retriever, CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT)

        result = call_conversational_rag(question, final_chain, memory)

        # Convert result to DataFrame
        result_df = pd.DataFrame({
            'question': [question],
            'answer': [result['answer']],
            'contexts': [result['context']]
        })

        st.markdown("Answer")
        st.write(result['answer'])
        st.write("-----------------------------")
        st.markdown("Contexts")
        st.write(result['context'])
        st.write("-----------------------------")
        st.markdown("Result Dataframe")
        st.write(result_df.head())
        return result_df

#............................................................................
# AM ENGINEER EVAULATE PART
#............................................................................
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
)

def AM_Eval_Run(result_df):
    # Inference
    answer = result_df['answer'].values.tolist()
    question = result_df['question'].values.tolist()
    contexts = result_df['contexts'].values.tolist()
    ground_truths = result_df['ground_truth'].values.tolist()

    data = {
        "question": question,
        "answer": answer,
        "contexts": [contexts],
        "ground_truth": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    # Evaluation ve diğer işlemler buraya taşınabilir
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_similarity,
        ],
    )

    df = result.to_pandas()
    st.markdown("Evaluation Result")
    st.write(df.head())




########################################################################################
# AM ENGINEER PART
########################################################################################
# get vector store
def call_vectordb(embedding_function):
    collection_name_1 = st.text_input("What is the collection name: ", key="am_collectionxx")
    vector_db = Chroma(persist_directory=f"./chroma_db/{collection_name_1}", embedding_function=embedding_function)
    
    retriever = vector_db.as_retriever(lambda_val=0.025, k=7, filter=None)
    return retriever


def prompt_templates():
    #CONDENSE_QUESTION_PROMPT
    _template = """
    [INST]
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a FAISS index. This query will be used to retrieve documents with additional context.

    Let me share a couple examples that will be important.

    If you do not see any chat history, you MUST return the "Follow Up Input" as is:

    ```
    Chat History:

    Follow Up Input: What features can I add as an input of the machine learning model for the additive manufacturing balling situations?
    Standalone Question:
    What features can I add as an input of the machine learning model for the additive manufacturing balling situations?
    ```

    If this is the second question onwards, you should properly rephrase the question like this:

    ```

    Now, with those examples, here is the actual chat history and input question.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    [/INST]
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    #######################################################################
    #ANSWER_PROMPT
    template = """
    [INST]
    Answer the question based only on the following context:
    {context}

    Question: {question}
    [/INST]
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    #######################################################################
    #DEFAULT_DOCUMENT_PROMPT
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    return CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT


CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompt_templates()

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
  ):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def create_rag(llm, retriever,
              CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT):
    # Instantiate ConversationBufferMemory
    memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
    )

    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm,
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
        "question": itemgetter("question"),
        "context": final_inputs["context"]
    }
    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    return final_chain, memory

def call_conversational_rag(question, chain, memory):
    # Prepare the input for the RAG model
    inputs = {"question": question}

    # Invoke the RAG model to get an answer
    result = chain.invoke(inputs)

    # Save the current question and its answer to memory for future context
    memory.save_context(inputs, {"answer": result["answer"]})

    # Return the result
    return result

retriever = call_vectordb(embedding_function)

def main(question, llm, retriever):
    CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompt_templates()

    final_chain, memory = create_rag(llm, retriever,
              CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT)

    result= call_conversational_rag(question, final_chain, memory)

    # Convert result to DataFrame
    result_df = pd.DataFrame({
        'question': [question],
        'answer': [result['answer']],
        'contexts': [result['context']]
    })

    print(result)
    return result, result_df