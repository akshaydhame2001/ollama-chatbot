from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Mount static files for CSS, JS, etc. (if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize LangChain components
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please provide response to the user queries"),
        ("user", "Question:{question}")
    ]
)
llm = OllamaLLM(model="llama3.2:1b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


# Define the root route with HTML form
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": None})


# Define a POST route for handling form submissions
@app.post("/", response_class=HTMLResponse)
async def handle_query(request: Request, query: str = Form(...)):
    if query:
        response = chain.invoke({"question": query})
    else:
        response = "Please enter a query."
    return templates.TemplateResponse("index.html", {"request": request, "response": response})
