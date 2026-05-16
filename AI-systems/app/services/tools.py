from langchain_core.tools import tool
from app.services.rag_service import resume_db

@tool
def check_resume_context(candidate_id: str, query: str) -> str:
    """
    Searches the candidate's parsed resume for specific keywords, project experience, or skills.
    Use this tool when you need to personalize a technical question based on their past work.
    """
    context = resume_db.query_candidate_context(candidate_id, query)
    return context