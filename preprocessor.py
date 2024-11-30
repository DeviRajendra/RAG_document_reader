from PyPDF2 import PdfReader

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Load and split the paper
pdf_path = "research_paper.pdf"
reader = PdfReader(pdf_path)
full_text = "".join(page.extract_text() for page in reader.pages)
chunks = split_text(full_text)

# Prepare the documents variable
documents = [{"content": chunk} for chunk in chunks]
