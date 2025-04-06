from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, trim_messages
from langchain_core.tools import tool, ToolException, InjectedToolArg
from langgraph.prebuilt import InjectedState
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from .orchestrate import orchestrator
from grooagents.utils.tools import create_cache_dir
from grooagents.utils.embeddings_model.trainer import EmbeddingsTrainer
from grooagents.utils.embeddings_model.inference import EmbeddingsInference
import os
import random
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pickle as pkl
import pandas as pd

################## Helper Tools ##################
def __llm(query: str, model, system_prompt: str = None):
    """
    Call the LLM with the given query and model and optional system prompt.
    """
    if system_prompt:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
    else:
        messages = [
            HumanMessage(content=query)
        ]
    response = model.invoke(messages)
    return response

def find_optimal_clusters() -> int:
    """
    Find the optimal number of clusters for the given data.
    """
    return random.randint(10, 22)

################## LLM Tools ##################

@tool
def load_file():
    """
    Load a file from the local filesystem given a path.
    """
    print("Loading file...")
    pass

@tool
def select_features(query, state: Annotated[dict, InjectedState]):
    """
    Select features to use for downstream analysis.
    
    Params:
    - query: The query to use for selecting features (e.g, Given the list of following variables select the ).
    - state: The state of the pipeline.
    """
    print("Selecting features...")
    all_features = state.get("all_features", [])

    # call LLM to select the features
    model = state.get("model")
    selected_features = __llm(query + str(all_features), model)


@tool
def generate_embeddings(state: Annotated[dict, InjectedState]) -> dict:
    """
    Generate embeddings for each row in the DataFrame.
    
    Params:
    - state: The state of the pipeline.

    Outputs:
    - embeddings are generated and saved to local storage.
    """
    # Load the DataFrame and get the Selected features
    dataP = os.path.join(os.getcwd(), "input", "data.csv")
    df = pd.read_csv(dataP)

    # Train or Load existing Embedding model
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, "cache", "embedding_model.pt")):
        # Train the model
        trainer = EmbeddingsTrainer(data_file=dataP, save_model_path=os.path.join(cwd, "cache", "embedding_model.pt"))
        trainer.train()
    
    # Load the model
    inference_model = EmbeddingsInference(data_file=dataP, model_path=os.path.join(cwd, "cache", "embedding_model.pt"))

    # Generate embeddings for each row in the DataFrame
    embeddings = inference_model.inference()

    # Save the embeddings to local storage
    create_cache_dir()
    with open('embeddings.pkl', 'wb') as f:
        pkl.dump(embeddings, f)

    return {'messages': 'Embeddings generated and saved to local storage.'}

    
@tool
def run_clustering(state: Annotated[dict, InjectedState]) -> dict:
    """
    Run clustering on the embeddings.
    
    Params:
    - state: The state of the pipeline.

    Outputs:
    - clusters are generated and saved to local storage.
    """
    # Load the embeddings from local storage
    cwd = os.getcwd()
    embeddings_path = os.path.join(cwd, "cache", "embeddings.pkl")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
    else:
        embeddings = pkl.load(embeddings_path)

    # Run clustering algorithm
    cluster_num = find_optimal_clusters()
    clustering_model = AgglomerativeClustering(n_clusters=cluster_num, linkage='ward')
    clusters = clustering_model.fit_predict(embeddings)

    # Save the clusters to local storage
    create_cache_dir()
    np.save(os.path.join(cwd, "cache", "clusters.npy"), clusters)
    
    return {'messages': 'Clusters generated and saved to local storage.'}