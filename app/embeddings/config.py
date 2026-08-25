"""Embedding model construction for OpenAI-compatible endpoints.

The service talks to any endpoint that implements ``POST {base_url}/embeddings``.
Azure OpenAI exposes this at ``https://<resource>/openai/v1`` where the model
name is the deployment name.
"""

from llama_index.embeddings.openai import OpenAIEmbedding

from app.config import Settings


def build_embed_model(settings: Settings) -> OpenAIEmbedding:
    """Create the embedding client described by *settings*."""
    return OpenAIEmbedding(
        model_name=settings.embedding_model,
        dimensions=settings.embedding_dimension,
        embed_batch_size=settings.embedding_batch_size,
        api_key=settings.embedding_api_key,
        api_base=settings.embedding_base_url,
    )
