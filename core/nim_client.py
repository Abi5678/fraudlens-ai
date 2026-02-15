"""
NVIDIA NIM Client
Unified interface for NVIDIA NIM API services
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import httpx
from openai import AsyncOpenAI
from loguru import logger


# API key resolved from: env var → .env file → Streamlit secrets
_BUILTIN_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Try loading from .env file if not in environment
if not _BUILTIN_API_KEY:
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        _env_path = Path(__file__).parent.parent / ".env"
        if _env_path.exists():
            load_dotenv(_env_path)
            _BUILTIN_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
    except ImportError:
        pass

# Try Streamlit secrets as final fallback (for Streamlit Cloud)
if not _BUILTIN_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
            _BUILTIN_API_KEY = st.secrets["NVIDIA_API_KEY"]
    except Exception:
        pass


@dataclass
class NIMConfig:
    """Configuration for NVIDIA NIM services"""
    api_key: str = ""
    base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model: str = "meta/llama-3.3-70b-instruct"
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    rerank_model: str = "nvidia/nv-rerankqa-mistral-4b-v3"
    parse_model: str = "nvidia/nemotron-parse"
    # Vision/multimodal model for image analysis (deepfake, etc.). Use a model from the integrate API catalog
    # (e.g. meta/llama-3.2-11b-vision-instruct) to avoid "Function not found" 404 for your account.
    vision_model: str = "meta/llama-3.2-11b-vision-instruct"
    timeout: float = 120.0


class NIMClient:
    """
    Unified client for NVIDIA NIM services.
    Supports LLM inference, embeddings, reranking, and document parsing.
    """
    
    def __init__(self, config: Optional[NIMConfig] = None):
        if config is None:
            # Resolve API key: explicit env var → built-in key
            api_key = os.environ.get("NVIDIA_API_KEY", "") or _BUILTIN_API_KEY
            config = NIMConfig(
                api_key=api_key,
                llm_model=os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct"),
                embedding_model=os.environ.get("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"),
                rerank_model=os.environ.get("RERANK_MODEL", "nvidia/nv-rerankqa-mistral-4b-v3"),
                vision_model=os.environ.get("NIM_VISION_MODEL", "meta/llama-3.2-11b-vision-instruct"),
            )
        
        self.config = config
        if not (config.api_key or "").strip():
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "On Hugging Face Spaces: Settings → Variables and secrets → add NVIDIA_API_KEY. "
                "Get a key at https://build.nvidia.com/explore/discover"
            )

        # OpenAI-compatible client for LLM
        self.openai_client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        
        # HTTP client for other endpoints
        self.http_client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.timeout,
        )
        
        logger.info(f"NIM Client initialized with model: {config.llm_model}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Send a chat completion request to NVIDIA NIM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to config.llm_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        model = model or self.config.llm_model
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"NIM chat error: {e}")
            raise
    
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "query",
    ) -> List[List[float]]:
        """
        Generate embeddings for texts using NVIDIA NIM.

        Args:
            texts: List of texts to embed
            model: Embedding model (defaults to config.embedding_model)
            input_type: Type of input - "query" for search queries,
                       "passage" for documents to be searched

        Returns:
            List of embedding vectors
        """
        model = model or self.config.embedding_model

        try:
            response = await self.openai_client.embeddings.create(
                model=model,
                input=texts,
                extra_body={"input_type": input_type},
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"NIM embedding error: {e}")
            raise
    
    async def rerank(
        self,
        query: str,
        passages: List[str],
        model: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages based on relevance to query.
        
        Args:
            query: Query text
            passages: List of passages to rerank
            model: Reranking model
            top_k: Number of top results to return
            
        Returns:
            List of reranked results with scores
        """
        model = model or self.config.rerank_model
        
        try:
            response = await self.http_client.post(
                "/ranking",
                json={
                    "model": model,
                    "query": {"text": query},
                    "passages": [{"text": p} for p in passages],
                }
            )
            response.raise_for_status()

            results = response.json()
            # Sort by score and return top_k
            rankings = sorted(
                results.get("rankings", []),
                key=lambda x: x.get("logit", 0),
                reverse=True
            )[:top_k]

            return [
                {
                    "text": passages[r["index"]],
                    "score": r.get("logit", 0),
                    "index": r["index"]
                }
                for r in rankings
            ]

        except Exception as e:
            logger.warning(f"NIM rerank unavailable ({e}), falling back to passthrough ordering")
            # Graceful fallback: return passages in original order with neutral scores.
            # Use 0.5 (not 1.0) since we cannot confirm relevance without reranking.
            return [
                {"text": p, "score": max(0.1, 0.5 - (i * 0.03)), "index": i}
                for i, p in enumerate(passages[:top_k])
            ]
    
    async def parse_document(
        self,
        file_path: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse a document using Nemotron-Parse via NIM.
        
        Args:
            file_path: Path to document (PDF or image)
            model: Parse model to use
            
        Returns:
            Parsed document with text, tables, and metadata
        """
        model = model or self.config.parse_model
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path, f)}
                response = await self.http_client.post(
                    f"/{model}/parse",
                    files=files,
                )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"NIM parse error: {e}")
            raise
    
    async def close(self):
        """Close HTTP client connections"""
        await self.http_client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Singleton instance
_nim_client: Optional[NIMClient] = None


def get_nim_client() -> NIMClient:
    """Get or create the global NIM client instance"""
    global _nim_client
    if _nim_client is None:
        _nim_client = NIMClient()
    return _nim_client
