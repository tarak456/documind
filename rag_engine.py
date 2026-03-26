import re
import requests


class RAGEngine:
    CHUNK_SIZE    = 600
    CHUNK_OVERLAP = 80
    TOP_K         = 5
    MODEL         = "gemini-2.0-flash"
    API_URL       = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chunks = []
        self._idf   = {}

    def build_index(self, text: str):
        self.chunks = self._split(text)
        self._build_tfidf()

    def _split(self, text: str) -> list:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + self.CHUNK_SIZE]))
            i += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    def _tokenize(self, text: str) -> list:
        return re.findall(r'[a-z]+', text.lower())

    def _build_tfidf(self):
        import math
        N = len(self.chunks)
        df = {}
        for chunk in self.chunks:
            for term in set(self._tokenize(chunk)):
                df[term] = df.get(term, 0) + 1
        self._idf = {t: math.log((N + 1) / (v + 1)) + 1 for t, v in df.items()}

    def _score(self, query: str, chunk: str) -> float:
        q_terms = set(self._tokenize(query))
        c_terms = self._tokenize(chunk)
        tf = {}
        for t in c_terms:
            tf[t] = tf.get(t, 0) + 1
        n = len(c_terms) or 1
        return sum((tf.get(t, 0) / n) * self._idf.get(t, 0) for t in q_terms)

    def retrieve(self, query: str, k=None) -> list:
        k = k or self.TOP_K
        scored = sorted(
            [(self._score(query, c), c) for c in self.chunks],
            key=lambda x: x[0], reverse=True,
        )
        return [c for _, c in scored[:k]]

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        url = self.API_URL.format(model=self.MODEL, key=self.api_key)
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.3,
            },
        }
        try:
            r = requests.post(url, json=body, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            err = r.json().get("error", {}).get("message", r.text[:200])
            return f"API error {r.status_code}: {err}"
        except Exception as e:
            return f"Request failed: {str(e)[:150]}"

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
            top_chunks = self.retrieve("main ideas key points important information summary conclusions")
            context    = "\n\n---\n\n".join(top_chunks)
            context   += f"\n\n[Document has ~{word_count:,} words. Showing most relevant sections.]"

        prompt = f"""You are an expert document analyst. Summarize the document below.

Style: {style_guide}
Length: {length_guide}

Document Content:
{context}

Write the summary directly. Do not add any preamble."""

        return self._call_llm(prompt)

    def answer_question(self, full_text: str, question: str) -> str:
        word_count = len(full_text.split())
        if word_count <= 1500:
            context = full_text
        else:
            top_chunks = self.retrieve(question, k=6)
            context    = "\n\n---\n\n".join(top_chunks)

        prompt = f"""Answer the following question using ONLY the document content below.

Question: {question}

Document Content:
{context}

Instructions:
- Answer directly from the document only
- If not found, say "This information is not mentioned in the document."
- Do not start with "Based on the document" """

        return self._call_llm(prompt, max_tokens=512)
