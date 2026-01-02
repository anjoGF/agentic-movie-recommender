from dataclasses import dataclass
import torch


@dataclass
class POCConfig:
    # -----------------------------
    # Data
    # -----------------------------
    data_dir: str = "data/ml-latest-small"
    movielens_url: str = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    # Embeddings (persisted)

    embedding_model_id: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_path: str = "embeddings/movie_embeddings.npy"
    embedding_ids_path: str = "embeddings/movie_ids.json"

    # Retrieval sizes

    cf_k: int = 200
    semantic_k: int = 80
    final_k: int = 20

    llm_model_id: str = "microsoft/phi-2"
    llm_max_new_tokens: int = 160
    llm_temperature: float = 0.2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Verbosity

    verbose: bool = True
    verbose_prompts: bool = False
    verbose_openai: bool = True

    # Strategy mode

    # v1 = baseline hybrid
    # v2 = agentic + critic + advantage-weighted
    strategy_version: str = "v2"

    # Planner constraints

    enforce_hybrid_for_search: bool = True

    # CF should never dominate semantic intent for search queries
    min_cf_weight: float = 0.20
    max_cf_weight: float = 0.60

    # Advantage-Weighted Ranking
    # How strongly we favor "advantaged" items over baseline popularity
    advantage_alpha: float = 0.6

    # Penalizes popularity bias (higher = more niche)
    novelty_lambda: float = 0.20

    critic_topn: int = 10

    # Popularity control
    popularity_mean_threshold: float = 0.65

    # Minimum unique genres in top-N
    genre_diversity_min_unique: int = 4

    min_primary_genre_ratio: float = 0.40

    # Amount to reduce CF when genre drift is detected
    genre_drift_cf_penalty: float = -0.30

    # Corresponding semantic boost
    genre_drift_semantic_boost: float = 0.30

    prefer_weight_adjustments: bool = True

    allow_hard_removal_without_weight_shift: bool = False
