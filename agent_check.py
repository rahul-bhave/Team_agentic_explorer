from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.groq import Groq
from agno.vectordb.lancedb import LanceDb
from sentence_transformers import SentenceTransformer

# 🧠 Custom embedder using SentenceTransformer
class MyEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimensions = 384  # Embedding size of 'all-MiniLM-L6-v2'

    def get_embedding(self, text: str) -> list:
        return self.model.encode(text).tolist()  # Make sure the return is a list of floats
    
    def get_embedding_and_usage(self, text: str):
        # Return both the embedding and None (usage could be a flag or other metadata)
        embedding = self.get_embedding(text)
        return embedding, None  # We don't use the second value in this context

# 📦 LanceDB vector store with custom embedder
vector_db = LanceDb(
    table_name="recipes",
    uri="C:/tmp/lancedb",  # Change this path if needed for storage
    embedder=MyEmbedder()  # Embedder should be passed as an instance
)

# 📚 Load PDF as knowledge base
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)

# Run once to ingest (set recreate=True if you want to reload)
knowledge_base.load(recreate=False)

# 🧠 LLM via Groq
agent = Agent(
    model=Groq(id="llama3-70b-8192"),  # You can change the model ID to your desired Groq model
    knowledge=knowledge_base,
    show_tool_calls=True,
)

# ❓ Ask your question
agent.print_response("How to make Thai curry?", markdown=True)
