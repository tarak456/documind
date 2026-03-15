import re
from groq import Groq


class RAGEngine:
    """
    RAG Pipeline:
    1. Split document into overlapping chunks
    2. TF-IDF scoring to find most relevant chunks
    3. Feed top chunks to Groq (cloud LLM) for summarization or Q&A
    """

    CHUNK_SIZE    = 600
    CHUNK_OVERLAP = 80
    TOP_K         = 5
    MODEL         = "llama3-8b-8192"   # free & fast on Groq

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.chunks: list[str] = []
        self._idf:   dict[str, float] = {}

    # ── Indexing ──────────────────────────────────────────────────────────────

    def build_index(self, text: str):
        self.chunks = self._split(text)
        self._build_tfidf()

    def _split(self, text: str) -> list[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + self.CHUNK_SIZE]))
            i += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'[a-z]+', text.lower())

    def _build_tfidf(self):
        import math
        N = len(self.chunks)
        df: dict[str, int] = {}
        for chunk in self.chunks:
            for term in set(self._tokenize(chunk)):
                df[term] = df.get(term, 0) + 1
        self._idf = {t: math.log((N + 1) / (v + 1)) + 1 for t, v in df.items()}

    def _score(self, query: str, chunk: str) -> float:
        q_terms = set(self._tokenize(query))
        c_terms  = self._tokenize(chunk)
        tf: dict[str, float] = {}
        for t in c_terms:
            tf[t] = tf.get(t, 0) + 1
        n = len(c_terms) or 1
        return sum((tf.get(t, 0) / n) * self._idf.get(t, 0) for t in q_terms)

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        k = k or self.TOP_K
        scored = sorted(
            [(self._score(query, c), c) for c in self.chunks],
            key=lambda x: x[0], reverse=True,
        )
        return [c for _, c in scored[:k]]

    # ── Shared LLM call ───────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    # ── Summarization ─────────────────────────────────────────────────────────

    def summarize(self, full_text: str, summary_type: str, summary_length: str) -> str:
        length_guide = {
            "Short":  "Write a very concise summary of 3 to 5 sentences only.",
            "Medium": "Write a well-rounded summary of around 150 to 250 words.",
            "Long":   "Write a detailed summary of 300 to 500 words covering all major points.",
        }[summary_length]

        style_guide = {
            "Comprehensive":     "Write a thorough summary covering all main ideas, arguments, and conclusions.",
            "Brief (3-5 lines)": "Write exactly 3 to 5 sentences capturing only the most essential information.",
            "Bullet Points":     "Format the summary as clear bullet points. Each bullet = one key idea.",
            "Executive":         "Write an executive summary: purpose, key findings, and action points.",
            "Technical":         "Write a technical summary focusing on methodology, data, metrics, and conclusions.",
        }[summary_type]

        word_count = len(full_text.split())
        if word_count <= 1500:
            context = full_text
        else:
            query      = "main ideas key points important information summary conclusions"
            top_chunks = self.retrieve(query)
            context    = "\n\n---\n\n".join(top_chunks)
            context   += f"\n\n[Note: Document has ~{word_count:,} words. Showing most relevant sections.]"

        prompt = f"""You are an expert document analyst. Your task is to summarize the document below.

Style: {style_guide}
Length: {length_guide}

Document Content:
{context}

Write the summary directly. Do not say "Here is a summary" or add any preamble."""

        return self._call_llm(prompt)

    # ── Q&A ───────────────────────────────────────────────────────────────────

    def answer_question(self, full_text: str, question: str) -> str:
        word_count = len(full_text.split())
        if word_count <= 1500:
            context = full_text
        else:
            top_chunks = self.retrieve(question, k=6)
            context    = "\n\n---\n\n".join(top_chunks)
            context   += f"\n\n[Document has ~{word_count:,} words total. Showing most relevant sections.]"

        prompt = f"""You are a precise document analyst. Answer the following question using ONLY the information found in the document content below.

Question: {question}

Document Content:
{context}

Instructions:
- Answer directly and specifically based on the document
- If the answer is not found in the document, say "This information is not mentioned in the document."
- Be concise but complete
- Do not add information from outside the document
- Do not start with phrases like "Based on the document" — just answer directly"""

        return self._call_llm(prompt, max_tokens=512)
