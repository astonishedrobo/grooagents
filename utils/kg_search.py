import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import tiktoken

from graphrag.config.load_config import load_config
from graphrag.config.enums import ModelType
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.drift_search.drift_context import (
    DRIFTSearchContextBuilder,
)
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag import api
from graphrag.query.structured_search.drift_search.state import QueryState

# ----------------------
# Module-level constants
# ----------------------
CWD = Path.cwd()
PROJECT_DIRECTORY = CWD / "cache" / "kbgraph"
INPUT_DIR = PROJECT_DIRECTORY / "output"
LANCEDB_URI = str(INPUT_DIR / "lancedb")
COMMUNITY_LEVEL = 2

# -------------------------
# Global Search Setup
# -------------------------
CONF = load_config(PROJECT_DIRECTORY)
ENTITIES_DF = pd.read_parquet(PROJECT_DIRECTORY / "output" / "entities.parquet")
COMMUNITIES_DF = pd.read_parquet(PROJECT_DIRECTORY / "output" / "communities.parquet")
COMMUNITY_REPORTS_DF = pd.read_parquet(PROJECT_DIRECTORY / "output" / "community_reports.parquet")

# -------------------------
# Drift Search Initialization
# -------------------------
# Load environment variables
dotenv_path = CWD / ".env"
load_dotenv(dotenv_path)
API_KEY = os.environ["GRAPHRAG_API_KEY"]
LLM_MODEL = os.environ["GRAPHRAG_LLM_MODEL"]
EMBED_MODEL = os.environ["GRAPHRAG_EMBEDDING_MODEL"]

# Load static data once
entity_df       = pd.read_parquet(INPUT_DIR / "entities.parquet")
community_df    = pd.read_parquet(INPUT_DIR / "communities.parquet")
relationship_df = pd.read_parquet(INPUT_DIR / "relationships.parquet")
text_unit_df    = pd.read_parquet(INPUT_DIR / "text_units.parquet")
report_df       = pd.read_parquet(INPUT_DIR / "community_reports.parquet")

# Build indexer adapters
entities      = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)
relationships = read_indexer_relationships(relationship_df)
text_units    = read_indexer_text_units(text_unit_df)
reports       = read_indexer_reports(
    report_df,
    community_df,
    COMMUNITY_LEVEL,
    content_embedding_col="full_content_embeddings",
)

# Setup vector stores
# Entity-description embeddings (for entity context)
description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
description_embedding_store.connect(db_uri=LANCEDB_URI)

# Full-content report embeddings (for context building)
full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
full_content_embedding_store.connect(db_uri=LANCEDB_URI)
# Load report embeddings into report objects
read_indexer_report_embeddings(reports, full_content_embedding_store)

# Language model & embedding model setup
chat_config = LanguageModelConfig(
    api_key=API_KEY,
    type=ModelType.OpenAIChat,
    model=LLM_MODEL,
    max_retries=20,
)
chat_model = ModelManager().get_or_create_chat_model(
    name="local_search", model_type=ModelType.OpenAIChat, config=chat_config
)
token_encoder = tiktoken.encoding_for_model(LLM_MODEL)

embed_config = LanguageModelConfig(
    api_key=API_KEY,
    type=ModelType.OpenAIEmbedding,
    model=EMBED_MODEL,
    max_retries=20,
)
text_embedder = ModelManager().get_or_create_embedding_model(
    name="local_search_embedding", model_type=ModelType.OpenAIEmbedding, config=embed_config
)

# DRIFT search configuration
drift_params = DRIFTSearchConfig(
    temperature=0,
    max_tokens=12_000,
    primer_folds=1,
    drift_k_followups=3,
    n_depth=3,
    n=1,
)

# Build the DRIFT context builder & searcher once
context_builder = DRIFTSearchContextBuilder(
    model=chat_model,
    text_embedder=text_embedder,
    entities=entities,
    relationships=relationships,
    reports=reports,
    entity_text_embeddings=description_embedding_store,
    text_units=text_units,
    token_encoder=token_encoder,
    config=drift_params,
)

drift_searcher = DRIFTSearch(
    model=chat_model,
    context_builder=context_builder,
    token_encoder=token_encoder,
)

# -------------------------
# Async API Functions
# -------------------------
async def global_search(query: str):
    """
    Perform a global search over entities, communities, and reports.
    Reuses module-level DataFrames and config for speed.
    """
    response, context = await api.global_search(
        config=CONF,
        entities=ENTITIES_DF,
        communities=COMMUNITIES_DF,
        community_reports=COMMUNITY_REPORTS_DF,
        community_level=COMMUNITY_LEVEL,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=query,
    )
    return response, context


async def drift_search(query: str):
    """
    Perform a DRIFT search using pre-initialized vector stores,
    models, and context builder. This runs in constant time
    after module load.
    """
    drift_searcher.query_state = QueryState()
    result = await drift_searcher.search(query)
    return result.response


if __name__ == "__main__":
    q = "what is the vulnerability of assam?"
    resp = asyncio.run(drift_search(q))
    print(resp)
