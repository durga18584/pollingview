from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
import tempfile
from orchestrator import RAGOrchestrator

# Initialize orchestrator
rag = RAGOrchestrator()

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid RAG Orchestrator",
    description="FAISS + Elasticsearch + Redis + OpenRouter",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Simple HTML form for uploading files and asking questions.
    """
    return """
    <html>
        <head>
            <title>RAG App</title>
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <h1>Hybrid RAG Search</h1>

            <h2>1. Upload Documents</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="files" multiple>
                <button type="submit">Upload</button>
            </form>
            <br>

            <h2>2. Ask a Question</h2>
            <form action="/ask" method="post">
                <input type="text" name="question" placeholder="Enter your question" size="50">
                <button type="submit">Ask</button>
            </form>
        </body>
    </html>
    """

@app.post("/upload")
async def upload(files: list[UploadFile]):
    """
    Ingest one or more documents into FAISS + Elasticsearch.
    """
    total_chunks = 0
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            tmp.write(await file.read())
            file_path = tmp.name
        result = rag.ingest(file_path)
        total_chunks += result["chunks_indexed"]

    return {"status": "success", "total_chunks_indexed": total_chunks}

@app.post("/ask", response_class=HTMLResponse)
async def ask(question: str = Form(...)):
    """
    Ask a question and return the answer in HTML.
    """
    response = rag.answer(question)
    answer = response["answer"] if isinstance(response, dict) else str(response)

    return f"""
    <html>
        <body style="font-family: Arial; margin: 40px;">
            <h1>Question:</h1>
            <p>{question}</p>

            <h1>Answer:</h1>
            <p>{answer}</p>

            <br><a href="/">Go Back</a>
        </body>
    </html>
    """
