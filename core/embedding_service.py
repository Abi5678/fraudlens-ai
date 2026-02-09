"""
Embedding Service & Vector Store
Uses NVIDIA NeMo Retriever for embeddings and Milvus for storage
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from core.nim_client import get_nim_client, NIMClient


@dataclass
class SearchResult:
    """Search result from vector store"""
    text: str
    score: float
    metadata: Dict[str, Any]
    id: str


class EmbeddingService:
    """
    Generate embeddings using NVIDIA NeMo Retriever NIMs.
    """
    
    def __init__(
        self,
        nim_client: Optional[NIMClient] = None,
        model: str = "nvidia/nv-embedqa-e5-v5",
    ):
        self.nim_client = nim_client or get_nim_client()
        self.model = model
        self.dimension = 1024  # nv-embedqa-e5-v5 dimension
        
        logger.info(f"EmbeddingService initialized with model: {model}")
    
    async def embed_text(self, text: str, input_type: str = "query") -> List[float]:
        """Embed a single text"""
        embeddings = await self.nim_client.embed([text], model=self.model, input_type=input_type)
        return embeddings[0]

    async def embed_texts(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        """Embed multiple texts (default input_type=passage for document storage)"""
        return await self.nim_client.embed(texts, model=self.model, input_type=input_type)
    
    async def embed_with_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Embed query and rerank documents for better retrieval.
        Two-stage retrieval: embedding similarity + reranking.
        """
        # First get reranked results
        reranked = await self.nim_client.rerank(
            query=query,
            passages=documents,
            top_k=top_k,
        )
        
        return reranked


class VectorStore:
    """
    Vector store using Milvus for fraud pattern storage.
    """
    
    def __init__(
        self,
        collection_name: str = "fraud_patterns",
        embedding_service: Optional[EmbeddingService] = None,
        host: str = "localhost",
        port: int = 19530,
        use_lite: bool = True,
    ):
        self.collection_name = collection_name
        self.embedding_service = embedding_service or EmbeddingService()
        self.host = host
        self.port = port
        self.use_lite = use_lite
        
        self._collection = None
        self._client = None
        
        logger.info(f"VectorStore initialized: {collection_name}")
    
    async def initialize(self):
        """Initialize Milvus connection and collection"""
        try:
            if self.use_lite:
                # Use Milvus Lite for local development
                from pymilvus import MilvusClient, DataType

                db_path = f"./{self.collection_name}.db"
                self._client = MilvusClient(db_path)

                # Drop and recreate if schema mismatch
                if self._client.has_collection(self.collection_name):
                    self._client.drop_collection(self.collection_name)

                # Create collection with explicit string ID schema
                schema = self._client.create_schema(auto_id=False)
                schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
                schema.add_field("text", DataType.VARCHAR, max_length=65535)
                schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.embedding_service.dimension)
                schema.add_field("metadata", DataType.JSON)

                index_params = self._client.prepare_index_params()
                index_params.add_index(field_name="vector", metric_type="COSINE")

                self._client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params,
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                # Use full Milvus server
                from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
                
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                )
                
                # Define schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_service.dimension),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                ]
                schema = CollectionSchema(fields, description="Fraud patterns")
                
                self._collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                )
                
                # Create index
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self._collection.create_index("embedding", index_params)
                self._collection.load()
                
            logger.info("VectorStore initialized successfully")
            
        except Exception as e:
            logger.error(f"VectorStore initialization error: {e}")
            raise
    
    async def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional IDs (auto-generated if not provided)
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = await self.embedding_service.embed_texts(texts)
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Default metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        try:
            if self.use_lite and self._client:
                # Milvus Lite format
                data = [
                    {
                        "id": id_,
                        "text": text,
                        "vector": embedding,
                        "metadata": metadata,
                    }
                    for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas)
                ]
                self._client.insert(self.collection_name, data)
            else:
                # Full Milvus format
                data = [
                    ids,
                    texts,
                    embeddings,
                    metadatas,
                ]
                self._collection.insert(data)
            
            logger.info(f"Added {len(texts)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        rerank: bool = True,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_expr: Optional filter expression
            rerank: Whether to rerank results
            
        Returns:
            List of SearchResult
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        try:
            if self.use_lite and self._client:
                # Milvus Lite search
                results = self._client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    limit=top_k * 2 if rerank else top_k,  # Get more if reranking
                    output_fields=["text", "metadata"],
                )
                
                search_results = []
                for hits in results:
                    for hit in hits:
                        search_results.append(SearchResult(
                            text=hit.get("entity", {}).get("text", ""),
                            score=hit.get("distance", 0),
                            metadata=hit.get("entity", {}).get("metadata", {}),
                            id=str(hit.get("id", "")),
                        ))
            else:
                # Full Milvus search
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                
                results = self._collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k * 2 if rerank else top_k,
                    expr=filter_expr,
                    output_fields=["text", "metadata"],
                )
                
                search_results = []
                for hits in results:
                    for hit in hits:
                        search_results.append(SearchResult(
                            text=hit.entity.get("text", ""),
                            score=hit.score,
                            metadata=hit.entity.get("metadata", {}),
                            id=str(hit.id),
                        ))
            
            # Rerank if enabled
            if rerank and search_results:
                texts = [r.text for r in search_results]
                reranked = await self.embedding_service.embed_with_rerank(
                    query=query,
                    documents=texts,
                    top_k=top_k,
                )
                
                # Rebuild results with rerank scores
                reranked_results = []
                for r in reranked:
                    idx = r["index"]
                    reranked_results.append(SearchResult(
                        text=r["text"],
                        score=r["score"],
                        metadata=search_results[idx].metadata,
                        id=search_results[idx].id,
                    ))
                
                return reranked_results[:top_k]
            
            return search_results[:top_k]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    async def delete(self, ids: List[str]):
        """Delete documents by ID"""
        try:
            if self.use_lite and self._client:
                self._client.delete(
                    collection_name=self.collection_name,
                    filter=f'id in {ids}',
                )
            else:
                expr = f'id in {ids}'
                self._collection.delete(expr)
            
            logger.info(f"Deleted {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise
    
    async def close(self):
        """Close connections"""
        if self._client:
            self._client = None
        if self._collection:
            from pymilvus import connections
            connections.disconnect("default")


# Fraud pattern database - pre-loaded patterns
FRAUD_PATTERNS = [
    {
        "id": "staged_accident_1",
        "text": "Staged collision with minimal vehicle damage but extensive claimed injuries. Multiple claimants from same vehicle reporting soft tissue injuries.",
        "metadata": {"category": "staged_accident", "severity": "high"},
    },
    {
        "id": "phantom_vehicle_1", 
        "text": "Claim involves vehicle that cannot be verified or located. VIN does not match registration records.",
        "metadata": {"category": "phantom_vehicle", "severity": "critical"},
    },
    {
        "id": "medical_mill_1",
        "text": "Excessive medical treatment at known fraud clinic. Treatment extends far beyond expected recovery time for injury type.",
        "metadata": {"category": "medical_fraud", "severity": "high"},
    },
    {
        "id": "prior_damage_1",
        "text": "Photos show pre-existing damage inconsistent with reported accident. Rust or weathering visible on damaged areas.",
        "metadata": {"category": "prior_damage", "severity": "medium"},
    },
    {
        "id": "identity_fraud_1",
        "text": "Claimant information does not match policy records. Social security number linked to multiple identities.",
        "metadata": {"category": "identity_fraud", "severity": "critical"},
    },
    {
        "id": "premium_fraud_1",
        "text": "Policy obtained with false information about vehicle usage, location, or driver history.",
        "metadata": {"category": "premium_fraud", "severity": "medium"},
    },
    {
        "id": "inflated_claim_1",
        "text": "Claimed repair costs significantly exceed market value for similar damage. Parts priced above MSRP.",
        "metadata": {"category": "inflated_claim", "severity": "medium"},
    },
    {
        "id": "fraud_ring_1",
        "text": "Multiple claims with shared addresses, phone numbers, or providers. Coordinated timing of incidents.",
        "metadata": {"category": "fraud_ring", "severity": "critical"},
    },
]


async def initialize_fraud_patterns(vector_store: VectorStore):
    """Load initial fraud patterns into vector store"""
    texts = [p["text"] for p in FRAUD_PATTERNS]
    metadatas = [p["metadata"] for p in FRAUD_PATTERNS]
    ids = [p["id"] for p in FRAUD_PATTERNS]
    
    await vector_store.add_documents(texts, metadatas, ids)
    logger.info(f"Loaded {len(FRAUD_PATTERNS)} fraud patterns")
