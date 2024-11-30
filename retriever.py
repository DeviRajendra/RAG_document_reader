from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.schema import HumanMessage
from tqdm import tqdm

from llama import LLaMa

from PyPDF2 import PdfReader

class Retriever:
    def __init__(self, pdf_path ):
        self.pdf_path = pdf_path
        self.reader = PdfReader(self.pdf_path)
        self.full_text = "".join(page.extract_text() for page in self.reader.pages)
        self.chunks = self.split_text(self.full_text)

        self.documents = [{"content": chunk} for chunk in tqdm(self.chunks)]

        self.texts = [doc["content"] for doc in tqdm(self.documents)]
        self.faq_template = """
            You are an expert research assistant with in-depth knowledge of the provided context. Your task is to carefully read and analyze the context and answer the user's questions with precision. The user may ask about specific topics, summaries, or detailed explanations from the book.

            Follow these instructions:

            1. Base your response **only** on the context provided. Do not make assumptions or include information outside the book.
            2. If the context does not fully address the query, explicitly state that the context does not contain sufficient information.
            3. When answering:
            - Provide concise and accurate responses for straightforward queries.
            - Use examples or explanations from the book for complex questions.
            - For structured information (e.g., chapters, sections), clearly list or describe them as they appear in the book.
            4. Ensure your response is easy to understand and relevant to the query.
            5. In the first conversation of the chat, give precise information about the book and ask to prompt the questions.

            Here is the context from the book:
            <context>
            {context}
            </context>

            Your response should adhere to the instructions above.
            """
        self.retriever = FAISS.from_texts(
            self.texts,
            HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        ).as_retriever(k=5)

    def split_text(self, text, chunk_size=300, overlap=50):
        chunks = []
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def parse_retriever_input(self, params):
        return params["messages"][-1].content

    def retrieve_content(self, prompt):       

        faq_prompt = ChatPromptTemplate.from_messages([
            ("system", self.faq_template),
            MessagesPlaceholder("messages")
        ])

        document_chain = create_stuff_documents_chain(LLaMa(), faq_prompt)

        response = RunnablePassthrough.assign(
            context= self.parse_retriever_input | self.retriever
        ).assign(answer=document_chain).invoke({
        "messages":[
            HumanMessage(content=prompt)
        ]
        })
        return response