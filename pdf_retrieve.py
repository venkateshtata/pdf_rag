import getpass
import os
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import PyPDFDirectoryLoader

os.environ["OPENAI_API_KEY"] = getpass.getpass('Enter OpenAI API Key: ')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass('Enter Langchain API Key: ')


def load_pdfs_safe(directory):
    loader = PyPDFDirectoryLoader(directory)
    try:
        return loader.load()
    except:
        print(f"Error loading PDF:")
        return []
    


docs = load_pdfs_safe("pdf_dataset/")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# print('rag chain output: ', rag_chain.invoke("What is Task Decomposition?"))

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

print('rag_chain_with_source: ', rag_chain_with_source.invoke("what are formal requests ?"))



