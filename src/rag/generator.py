"""
src/rag/generator.py

LLM-based explanation generation using retrieved context.
"""

import os

# Fix SSL certificate issue from PyMOL
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class ExplanationGenerator:
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate_explanation(
        self,
        drug_name: str,
        prediction: str,
        confidence: float,
        top_genes: list[str],
        retrieved_context: list[dict]
    ) -> str:
        
        context_text = "\n\n".join([
            f"[Source: {r['metadata'].get('source', 'unknown')}]\n{r['content']}"
            for r in retrieved_context
        ])
        
        gene_list = ", ".join(top_genes[:10])
        
        system_prompt = """You are a clinical genomics expert assistant. Your role is to explain drug response predictions in a clear, evidence-based manner.

Guidelines:
- Explain the biological rationale for the prediction
- Reference specific genes and their known roles in drug response
- Be concise but informative
- Use the provided context to ground your explanation
- If the context does not contain relevant information, say so
- Do not invent information not present in the context"""

        user_prompt = f"""A deep learning model has made the following prediction:

Drug: {drug_name}
Prediction: {prediction}
Confidence: {confidence:.1%}
Top contributing genes: {gene_list}

Using the following knowledge base context, explain why this patient might be a {prediction.lower()} to {drug_name}. Focus on the genes listed above and their known roles in {drug_name} response.

Context:
{context_text}

Provide a clear, concise explanation (3-5 sentences) suitable for a clinical audience."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content