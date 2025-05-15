from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, trim_messages
from langchain_core.tools import tool, ToolException, InjectedToolArg
from langgraph.prebuilt import InjectedState
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from .orchestrate import orchestrator
from grooagents.utils.tools import create_cache_dir, load_config
from grooagents.utils.embeddings_model.trainer import EmbeddingsTrainer
from grooagents.utils.embeddings_model.inference import EmbeddingsInference
from grooagents.utils.embeddings_model.preprocess import load_and_preprocess_data
import os
import random
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pickle as pkl
import pandas as pd
import torch
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import threading
import json
from rapidfuzz import process, fuzz

################## Helper Tools ##################
def __llm(query: str, model, system_prompt: str = None):
    """
    Call the LLM with the given query and model and optional system prompt.
    """
    model = ChatOpenAI(model=model, temperature=0.7)
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
    return random.randint(10, 20)

def preprocess_data(scaler_type="standard"):
    """
    Preprocess the data for downstream analysis.
    
    Params:
    - state: The state of the pipeline.

    Outputs:
    - Preprocessed data is saved to local storage.
    """
    print("Preprocessing data...")
    config = load_config()
    continuous_cols = config.get("continuous_cols", [])
    categorical_cols = config.get("categorical_cols", [])

    df = load_and_preprocess_data(
        data_path=os.path.join(os.getcwd(), "input", "data.csv"),
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        scaler_type=scaler_type
    )

    # Save the preprocessed data to local storage
    create_cache_dir()
    df.to_csv(os.path.join(os.getcwd(), "cache", "preprocessed_data.csv"), index=False)



################## LLM Tools ##################

@tool
def table_lookup(value: str, primary_query: str, limit: int = 1, score_cutoff: int = 60, file_path: str = None) -> dict:
    """
    Load a CSV file (from config or provided file_path), perform a fuzzy search across all cell values,
    and return all rows containing the best-matching value as a Markdown-formatted mini-table.
    If file_path is provided, it overrides the default CSV path from config.

    Parameters
    ----------
    value : str
        The search string (can include typos) to match against every cell. Only mention the column value to match.
    primary_query : str
        The primary query that the agent is trying to answer.
    limit : int
        Number of top fuzzy matches to consider for cell values.
    score_cutoff : int
        Minimum similarity score (0-100) to accept a match.
    file_path : str, optional
        Path to the CSV file. If provided, overrides the default path from config.

    Returns
    -------
    dict
        A dictionary with 'messages' key containing either the markdown table or error message.
    """
    import os
    import pandas as pd
    from rapidfuzz import process, fuzz

    # Resolve CSV path
    if file_path:
        csv_path = os.path.join(os.getcwd(), "cache", file_path)
        print(csv_path)
    else:
        cwd = os.getcwd()
        config = load_config()
        csv_path = config.get("data_path", os.path.join(cwd, "input", "data.csv"))

    if not os.path.exists(csv_path):
        return {"messages": f"CSV file not found at {csv_path}"}

    # Load data
    df = pd.read_csv(csv_path)

    # Build list of unique cell values
    flat_vals = pd.Series(df.values.ravel()).astype(str).unique().tolist()

    # Fuzzy-match query against cell values
    best = process.extract(
        value,
        flat_vals,
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=score_cutoff
    )
    if not best:
        return {"messages": f"No close cell-value match for query: {value!r}"}
    
    best_value = best[0][0]

    # Filter rows containing the matched value in any column
    mask = df.astype(str).apply(lambda row: row.eq(best_value).any(), axis=1)
    matched_rows = df[mask]

    # Convert matched rows to markdown
    markdown_table = matched_rows.to_markdown(index=False)

    # Analyze the results using LLM to answer the primary query
    system_prompt = (
        "You are a data analyst. Given a query and a markdown table of results, "
        "provide a brief, clear, and natural language answer to the query based on the data shown."
    )
    
    analysis_query = f"Query: {primary_query}\n\nData:\n{markdown_table}\n\nPlease analyze this data to answer the query."
    
    # Use the existing __llm function to get the analysis
    response = __llm(analysis_query, 'gpt-4.1-mini-2025-04-14', system_prompt)
    analysis = response.content if hasattr(response, 'content') else str(response)
    
    return {"messages": f"Analysis: {analysis}"}
    # return {"messages": matched_rows.to_markdown(index=False)}

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
    preprocess_data()
    dataP = os.path.join(os.getcwd(), "cache", "preprocessed_data.csv")
    df = pd.read_csv(dataP)
    config = load_config()

    # Train or Load existing Embedding model
    cwd = os.getcwd()
    create_cache_dir()
    
    if not os.path.exists(os.path.join(cwd, "cache", "embedding_model.pt")):
        # Train the model
        trainer = EmbeddingsTrainer(data_file=dataP, save_model_path=os.path.join(cwd, "cache", "embedding_model.pt"))
        trainer.train()
    
    # Load the model
    variables = config.get("variables", [])
    inference_model = EmbeddingsInference(data_file=dataP, model_path=os.path.join(cwd, "cache", "embedding_model.pt"), variables=variables)

    # Generate embeddings for each row in the DataFrame
    embeddings = inference_model.inference()

    # Save the embeddings to local storage
    with open(os.path.join(cwd, "cache", 'embeddings.pkl'), 'wb') as f:
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
        with open(embeddings_path, 'rb') as f:
            embeddings = pkl.load(f)

    # Run clustering algorithm
    cluster_num = find_optimal_clusters()
    clustering_model = AgglomerativeClustering(n_clusters=cluster_num, linkage='ward')
    clusters = clustering_model.fit_predict(embeddings)

    # Save the clusters to local storage
    create_cache_dir()
    np.save(os.path.join(cwd, "cache", "clusters.npy"), clusters)
    
    return {'messages': f'{cluster_num} Clusters generated and saved to local storage.'}

#################### Feature Importance Analysis Tools ####################

@tool
def generate_feature_importance(state: Annotated[dict, InjectedState]) -> dict:

    """
    Generate feature importance for the selected features for.
    
    Params:
    - state: The state of the pipeline.

    Outputs:
    - feature importance is generated and saved to local storage.
    """
    # LLM (Orchestrator) to decide the tool calls
    model = ChatOpenAI(model="gpt-4o")
    tools_list = [shap_generator, groupwise_shap_generator, shap_to_nlp]
    model = model.bind_tools(tools_list)

    system_prompt = """
    You are a feature importance analysis assistant.
    You will be provided with the state of the pipeline.
    Your task is to decide the next steps for feature importance analysis.
    Based on the state of the pipeline, you will call the appropriate tools.
    """
    
    if not fi_reasoner(state):
        shap_generator()
    else:
        pass

    return {'messages': 'Feature importance analysis completed.'}

@tool
def shap_to_nlp(state: Annotated[dict, InjectedState], rowwise_desc: bool = False) -> dict:
    """
    Convert SHAP tables to natural language.
    
    Params:
    - state: The state of the pipeline.
    - rowwise_desc: Whether to generate rowwise description or not. [True, False] (defaults = False)

    Outputs:
    - SHAP tables are converted to natural language and saved to local storage.
    """

    model = ChatOpenAI(model="gpt-4o")
    cwd = os.getcwd()
    # Rowwise description (so for loop for each row)
    if rowwise_desc:
        pass

    # Clusterwise description (so for loop for each cluster)
    clusterwise_shap_to_nlp(os.path.join(cwd, "cache", "overall_shap_summary.csv"), model=model)

    return {'messages': 'SHAP tables converted to natural language and saved to local storage.'}

def rowwise_shap_to_nlp(path: str, model) -> dict:
    """
    Convert SHAP tables to natural language.
    
    Params:
    - state: The state of the pipeline.
    - rowwise_desc: Whether to generate rowwise description or not. (True/False)

    Outputs:
    - SHAP tables are converted to natural language and saved to local storage.
    """
    pass
    

def clusterwise_shap_to_nlp(path: str, model) -> dict:
    """
    Convert SHAP tables to natural language.
    
    Params:
    - state: The state of the pipeline.
    - rowwise_desc: Whether to generate rowwise description or not. (True/False)

    Outputs:
    - SHAP tables are converted to natural language and saved to local storage.
    """
    create_cache_dir()

    df = pd.read_csv(path)
    descriptions = {}
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        features_list = cluster_df[['feature', 'mean_abs_shap', 'direction']].to_dict(orient='records')
        system_prompt = (
            "You are a data storyteller. You're given a list of key features that define a cluster. "
            "Provide a human-readable summary describing the cluster's characteristics based on these features. "
            "Do not mention SHAP values, technical metrics, or directions."
        )
        query = f"For cluster {cluster}, the top features are {features_list}. Describe the cluster in clear, user-friendly language."
        response = __llm(query, model, system_prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        descriptions[f'Cluster_{cluster}'] = summary

    # Save summaries to JSON
    output_path = os.path.join(os.getcwd(), "cache", "clusterwise_shap_nlp.json")
    with open(output_path, "w") as f:
        json.dump(descriptions, f, indent=2)
    return {'messages': 'Clusterwise SHAP tables converted to natural language and saved to local storage.'}


def fi_reasoner(state: Annotated[dict, InjectedState]) -> bool:
    """
    Reasoner for feature importance analysis.
    - Analyzes if the user already asked for the groupwise analysis in the previous steps. 
    If not and different features (that can belong to different groups) were used in the previous steps,
    then ask the user if they want to do groupwise analysis.
    - If the user agrees, then chooses groups and verifies with the user.
    
    Params:
    - state: The state of the pipeline.

    Outputs:
    - updates (populates) the logs file with the selected groups [if any].

    Returns:
    - True if the user agreed to do groupwise analysis, False otherwise.
    """
    return False

def shap_generator():
    """
    Generate SHAP Tables for the selected features.

    Outputs:
    - SHAP tables are generated and saved to local storage.
    """
    # Load the preprocessed data
    if os.path.exists(os.path.join(os.getcwd(), "cache", "preprocessed_data.csv")):
        df = pd.read_csv(os.path.join(os.getcwd(), "cache", "preprocessed_data.csv"))
    else:
        raise FileNotFoundError("Preprocessed data file not found. Please run the preprocessing step first.")
    
    # Load the cluster labels
    if os.path.exists(os.path.join(os.getcwd(), "cache", "clusters.npy")):
        clusters = np.load(os.path.join(os.getcwd(), "cache", "clusters.npy"))
    else:
        raise FileNotFoundError("Clusters file not found. Please run the clustering step first.")
    
    df["cluster"] = clusters
    
    # Train XGBoost model
    X, y = df[[col for col in df.columns if col != "cluster"]], df["cluster"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        objective='multi:softprob',  # Use softprob for probability outputs needed by TreeExplainer
        eval_metric='mlogloss',
        # tree_method='hist', # Optional: often faster for large datasets
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_estimators=1000,
        max_depth=5,
        random_state=42,
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    # Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Clusterwise SHAP Contributions
    cluster_specific_summaries, overall_summary = create_full_cluster_summary(
        shap_values,
        [col for col in df.columns if col != "cluster"],
        X
    )
    overall_summary.to_csv(os.path.join(os.getcwd(), "cache", "overall_shap_summary.csv"), index=False)

def groupwise_shap_generator():
    """
    Generate SHAP Tables for each groups.

    Outputs:
    - SHAP tables are generated and saved to local storage.
    """
    if os.path.exists(os.path.join(os.getcwd(), "cache", "preprocessed_data.csv")):
        df = pd.read_csv(os.path.join(os.getcwd(), "cache", "preprocessed_data.csv"))
    else:
        raise FileNotFoundError("Preprocessed data file not found. Please run the preprocessing step first.")
    pass


def create_cluster_feature_table(shap_values, cluster_idx, feature_names, X_data):
    """
    Create a structured table of feature contributions for a specific cluster.
    Assumes shap_values shape is (samples, features, clusters).
    """
    if not (0 <= cluster_idx < shap_values.shape[2]):
         raise ValueError(f"cluster_idx {cluster_idx} is out of bounds for shap_values with {shap_values.shape[2]} classes.")

    # Extract SHAP values for the specific cluster
    # shap_values is list for multi-class, index directly if numpy array
    if isinstance(shap_values, list):
         cluster_shap = shap_values[cluster_idx] # Shape (samples, features)
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
         cluster_shap = shap_values[:, :, cluster_idx] # Shape (samples, features)
    else:
         raise TypeError("Unsupported shap_values format. Expected list or 3D numpy array.")

    # Check if X_data needs filtering (if shap_values were calculated on X_test)
    if X_data.shape[0] != cluster_shap.shape[0]:
        print(f"Warning: X_data rows ({X_data.shape[0]}) != SHAP rows ({cluster_shap.shape[0]}). Ensure correct data is passed.")
        # Potentially filter X_data here if necessary, e.g., X_data = X_data.loc[X_test.index]

    # Calculate metrics
    mean_abs_shap = np.mean(np.abs(cluster_shap), axis=0)
    mean_shap = np.mean(cluster_shap, axis=0)

    # Ensure feature_names matches the number of features in SHAP values
    if len(feature_names) != cluster_shap.shape[1]:
        raise ValueError(f"Number of feature_names ({len(feature_names)}) does not match number of features in SHAP values ({cluster_shap.shape[1]}).")

    contribution_df = pd.DataFrame({
        'feature': list(feature_names),
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': mean_shap,
        'direction': ['Positive' if x > 0 else 'Negative' for x in mean_shap],
        'feature_value_mean': X_data[feature_names].mean().values # Mean feature value across the provided X_data
    })

    contribution_df = contribution_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    contribution_df['rank'] = range(1, len(contribution_df) + 1)

    total_impact = contribution_df['mean_abs_shap'].sum()
    contribution_df['pct_contribution'] = (contribution_df['mean_abs_shap'] / total_impact) * 100 if total_impact > 0 else 0

    return contribution_df[['rank', 'feature', 'mean_shap', 'mean_abs_shap',
                          'pct_contribution', 'direction', 'feature_value_mean']]

def create_full_cluster_summary(shap_values, feature_names, X_data):
    """
    Create comprehensive feature contribution summary across all clusters.
    """
    cluster_summaries = {}
    overall_summaries_list = []

    # Get number of clusters from SHAP values shape
    if isinstance(shap_values, list):
        n_clusters = len(shap_values)
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        n_clusters = shap_values.shape[2]
    else:
         raise TypeError("Unsupported shap_values format. Expected list or 3D numpy array.")


    print(f"Generating summaries for {n_clusters} clusters...")

    for cluster_idx in range(n_clusters):
        try:
            cluster_df = create_cluster_feature_table(
                shap_values,
                cluster_idx,
                feature_names,
                X_data # Pass the relevant data (e.g., X or X_test)
            )

            # Add cluster information
            cluster_df['cluster'] = cluster_idx # Or use actual cluster labels if preferred

            cluster_summaries[f'Cluster_{cluster_idx}'] = cluster_df
            overall_summaries_list.append(cluster_df)

        except Exception as e:
            print(f"Error processing cluster {cluster_idx}: {str(e)}")
            continue

    # Create overall summary combining all clusters
    if overall_summaries_list:
        overall_summary_df = pd.concat(overall_summaries_list, axis=0, ignore_index=True)
    else:
        overall_summary_df = pd.DataFrame()

    return cluster_summaries, overall_summary_df

# Document Querying
def run_async(coro):
    """
    Synchronously run an async coroutine, whether or not there's
    already a running event loop in this thread.
    """
    import asyncio
    try:
        # If no loop is running, this will raise RuntimeError
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # ⁠— no loop: safe to use asyncio.run
        return asyncio.run(coro)
    else:
        # ⁠— a loop is running: spin up a NEW loop in a THREAD
        result = {}
        def _runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result["value"] = new_loop.run_until_complete(coro)

        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        return result["value"]


@tool
def kb_query(state: Annotated[dict, InjectedState], query: str) -> dict:
    """
    Query the knowledge base using DRIFT search with the given query.

    Params:
    - state: LangGraph state dictionary.
    - query: The natural language query.

    Returns:
    - A dictionary containing the search result.
    """
    import asyncio
    from grooagents.utils.kg_search import drift_search
    
    # Create a new event loop in this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async function in this loop
        response = loop.run_until_complete(drift_search(query))
    finally:
        # Clean up
        loop.close()
    
    cwd = os.getcwd()
    # Save the response to local storage
    with open(os.path.join(cwd, "cache", "drift_search_response.txt"), 'w') as f:
        f.write(response)
    
    return {"messages": response}

@tool
def visualize(state: Annotated[dict, InjectedState], query: str) -> dict:
    """
    Visualize the data using the given query.

    Params:
    - state: LangGraph state dictionary.
    - query: The natural language query.

    Returns:
    - A dictionary containing the visualization result.
    """
    # Implement visualization logic here
    pass
    return {"messages": "Visualization result."}

# Add new tool to list CSV files in cache directory
@tool
def list_cache_files() -> dict:
    """
    List all CSV files in the cache directory.
    Returns a dict with 'messages' key containing the list of files.
    """
    import os
    cache_dir = os.path.join(os.getcwd(), "cache")
    if not os.path.exists(cache_dir):
        return {"messages": "Cache directory not found."}
    files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
    return {"messages": f"Available CSV files in cache: {', '.join(files)}"}

@tool
def peek_csv(file_path: str, num_rows: int = 1) -> dict:
    """Show the structure and first few rows of a CSV file"""
    try:
        import pandas as pd
        
        # Read the CSV file
        file_path = os.path.join(os.getcwd(), "cache", file_path) 
        df = pd.read_csv(file_path)
        
        # Get column information
        columns_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'non_null_counts': df.count().to_dict(),
            'sample_rows': df.head(num_rows).to_dict('records')
        }
        
        # Format the output message
        message = f"CSV File Structure for {file_path}:\n\n"
        message += "Columns:\n"
        for col in columns_info['columns']:
            message += f"- {col} ({columns_info['dtypes'][col]})\n"
        
        message += f"\nNon-null counts:\n"
        for col, count in columns_info['non_null_counts'].items():
            message += f"- {col}: {count}\n"
        
        message += f"\nFirst {num_rows} rows:\n"
        for i, row in enumerate(columns_info['sample_rows']):
            message += f"\nRow {i+1}:\n"
            for col, val in row.items():
                message += f"- {col}: {val}\n"
        
        return {"messages": message}
        
    except Exception as e:
        return {"messages": f"Error reading CSV file: {str(e)}"}

@tool 
def kg_indexing(state: Annotated[dict, InjectedState], file_type='text', input_dir="input") -> dict:
    """
    Index the knowledge graph.
    
    Params:
    - file_type: ['text', 'str', 'json']. (default: 'text')
    - input_dir: The directory to index. (default: 'input')
    """
    from grooagents.utils.kg_index import index_kg, update_input_config
    import shutil

    update_input_config({'file_type': file_type})
    update_input_config({'base_dir': input_dir})

    run_async(index_kg())

    cache_dirs = ['community_reporting', 'extract_graph', 'summarize_descriptions', 'text_embedding']
    for dir in cache_dirs:
        # move the files to the new directory cache/kbgraph/cache
        for file in os.listdir(os.path.join(os.getcwd(), "cache", dir)):
            shutil.move(os.path.join(os.getcwd(), "cache", dir, file), os.path.join(os.getcwd(), "cache", "kbgraph", "cache", dir, file))
        # delete the directory
        shutil.rmtree(os.path.join(os.getcwd(), "cache", dir))
        
    return {"messages": "Knowledge graph indexed at cache/kbgraph."}
    