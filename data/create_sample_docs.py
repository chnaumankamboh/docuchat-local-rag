import os
from docx import Document

# Paths: the sample_docs folder next to this script (the one ingest.py reads)
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
os.makedirs(folder, exist_ok=True)

# 1️⃣ PDF content
pdf_content = """Company Leave Policy:
- Annual leave: 20 days per year
- Sick leave: 10 days per year
- Maternity/Paternity leave: 90 days
- Leave approval: Submit request to HR at least 5 days in advance

Work From Home Policy:
- Eligible 2 days per week
- Approval required from team lead
"""

with open(os.path.join(folder, "company_policy.txt"), "w", encoding="utf-8") as f:
    f.write(pdf_content)

# 2️⃣ DOCX content
doc = Document()
doc.add_heading("AI Trends 2026", level=1)
doc.add_paragraph("1. Generative AI continues to dominate content creation.")
doc.add_paragraph("2. More companies adopt Retrieval-Augmented Generation (RAG) systems.")
doc.add_paragraph("3. AI in healthcare is growing rapidly for diagnostics and research.")
doc.add_paragraph("4. AI ethics and regulation gain attention worldwide.")
doc.save(os.path.join(folder, "ai_trends.docx"))

# 3️⃣ TXT content
txt_content = """Python Coding Tips:
- Use list comprehensions for concise loops
- Always handle exceptions with try/except
- Use virtual environments for projects
- Document your code properly with comments and docstrings
"""
with open(os.path.join(folder, "tech_tips.txt"), "w", encoding="utf-8") as f:
    f.write(txt_content)

print(f"Sample documents created in {folder}")
