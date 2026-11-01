"""Personal-memory Chroma index; SQL ownership/status/revision remains authoritative."""
import hashlib
import copy
from app.services.vector_store import ChromaKnowledgeStore


def memory_index_fingerprint(settings):
    return hashlib.sha256(f'{settings.long_term_memory_collection}:{settings.openai_base_url}:{settings.openai_embedding_model}'.encode()).hexdigest()[:12]


class PersonalMemoryVectors:
    def __init__(self, settings):
        self.settings = settings
        self._store = None
        self._collection = None

    def _open(self):
        if not self.settings.long_term_memory_vector_enabled:
            raise RuntimeError('MemoryVectorsDisabled')
        if self._collection is None:
            # Reuse the existing embedding provider and persistent client, never its knowledge collection.
            embedding_settings = copy.copy(self.settings)
            embedding_settings.knowledge_vector_enabled = True
            self._store = ChromaKnowledgeStore(embedding_settings)
            if not self._store.can_embed:
                raise RuntimeError('MemoryVectorsUnavailable')
            fingerprint = memory_index_fingerprint(self.settings)
            self._collection = self._store.client.get_or_create_collection(
                name=f'{self.settings.long_term_memory_collection}_{fingerprint}', embedding_function=None,
                metadata={'hnsw:space': 'cosine', 'embedding_model': self.settings.openai_embedding_model})
        return self._collection

    def upsert(self, rows):
        collection = self._open()
        if not rows:
            return
        deleted = [str(r.id) for r in rows if r.status == 'DELETED']
        if deleted:
            collection.delete(ids=deleted)
        rows = [r for r in rows if r.status != 'DELETED']
        if not rows:
            return
        vectors = self._store.embed_texts([r.content for r in rows])
        collection.upsert(ids=[str(r.id) for r in rows], embeddings=vectors,
            documents=[r.content for r in rows], metadatas=[{
                'user_id': r.user_id, 'revision': r.revision, 'status': r.status,
                'memory_type': r.memory_type} for r in rows])

    def indexed_records(self, user_id):
        result = self._open().get(where={'user_id': user_id}, include=['metadatas'])
        return {str(key): (meta.get('revision'), meta.get('status')) for key, meta in zip(
            result.get('ids', []), result.get('metadatas', []))}

    def search(self, user_id, text, limit, include_completed=False):
        collection = self._open()
        count = collection.count()
        if not count:
            return []
        vector = self._store.embed_texts([text])[0]
        filters = [{'user_id': user_id}]
        filters.append({'status': {'$in': ['ACTIVE', 'COMPLETED']} if include_completed else 'ACTIVE'})
        result = collection.query(query_embeddings=[vector], n_results=min(count, max(1, limit)),
            where={'$and': filters}, include=['metadatas', 'distances'])
        return [(int(key), meta.get('revision'), float(distance)) for key, meta, distance in zip(
            result['ids'][0], result['metadatas'][0], result['distances'][0])]
