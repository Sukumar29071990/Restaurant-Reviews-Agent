from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(model= "llama-3.3-70b-versatile", temperature=0.8)

def get_restaurant_review(question: str, retriever : str ):
    template = """
        You are an exeprt in answering the questions about a restaurant. Show your review in list format

        Here are some relevant reviews: {reviews}

        Here is the question to answer: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    return result.content
