import RAG
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import json
import traceback
import os

app = FastAPI()
templates = Jinja2Templates(directory='templates')

# Create static directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
retriever = RAG.vector_store.as_retriever(search_kwargs={"k": 1})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | RAG.llm
    | StrOutputParser()
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    try:
        print(f"\n=== Received Query: {query} ===")
        
        # Get source documents FIRST (before generating answer)
        source_docs = retriever.invoke(query)  # Changed from get_relevant_documents to invoke
        
        # Get the answer
        answer = rag_chain.invoke(query)
        print(f"Answer: {answer}")
        
        source_document = source_docs[0].page_content if source_docs else "No source found"
        doc = source_docs[0].metadata.get('source', 'Unknown') if source_docs else "Unknown"
        
        print(f"Source: {doc}")
        
        # Create response
        response_dict = {
            "answer": answer,
            "source_document": source_document,
            "doc": doc
        }
        
        # Return as JSON directly, not double-encoded
        return JSONResponse(content=response_dict)
        
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        
        return JSONResponse(
            content={
                "answer": f"Sorry, an error occurred: {str(e)}",
                "source_document": "",
                "doc": ""
            },
            status_code=500
        )