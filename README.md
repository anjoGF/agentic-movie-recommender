
# Agentic Recommendation System (POC)

Most recommendation engines eventually default to showing the same few blockbusters because they rely too heavily on historical popularity data. This project is an experiment in using **LangGraph** to manage the balance between behavioral data and semantic intent—detecting when a list is getting "boring" or "drifting" and fixing the ranking logic on the fly.

## The Logic

Instead of a standard linear pipeline, I used a state machine where different agents handle specific parts of the recommendation lifecycle.

### 1. Intent & Planning (`intent_agent.py` / `planner_agent.py`)

The system starts by figuring out the "why" behind a request.

* If you ask for something "moody" or "not cheesy," the **Intent Agent** flags those as qualitative constraints rather than just keywords.
* The **Planner** then decides the retrieval mix. For very specific queries, it forces the system to trust **Semantic Search** more than the **Collaborative Filtering** (CF) scores.

### 2. Hybrid Retrieval & Advantage Ranking

The system runs two retrieval paths in parallel:

* **CF**: Captures "people who liked this also liked that" signals.
* **Semantic**: Uses embeddings to find titles that match the "vibe" or tone of the description.

To keep big blockbusters from winning every time, I wrote an **Advantage-Weighted Ranker (`ranker_v2.py`)**. It calculates an "Advantage Score" by subtracting a movie's baseline popularity from its relevance score. This effectively penalizes the stuff everyone has already seen to surface niche titles that actually fit the specific query.

### 3. The Critic & Rerank Loop

The **Critic Agent** acts as a monitor for "Genre Drift." For example, if you asked for a "Scary Sci-Fi" and the CF tool suggests *Star Wars* because it’s popular, the Critic identifies that *Star Wars* doesn't actually fit the "Scary" constraint.

Instead of just deleting the movie, the Critic triggers a **Rerank Loop**. It updates the weights (e.g., "Reduce CF influence by 30%, Boost Semantic by 30%") and runs the ranker again. This is a self-correcting loop that handles failure without requiring a user to refresh or rephrase.

### 4. Grounded Explanations

The **Explainer Agent** avoids generic "You might like this" fluff. It looks at the original intent and justifies the choice based on specific metadata. If the system couldn't find a perfect match, the explanation is written to reflect that uncertainty rather than providing a fake justification.

## Configuration & Tuning

You can adjust the system’s behavior in `config.py`:

* `novelty_lambda`: Sets the strength of the penalty for high-popularity items.
* `advantage_alpha`: Controls how much weight is given to intent-alignment over baseline utility.
* `min_primary_genre_ratio`: The threshold the Critic uses to determine if a list has drifted too far.

## Running the POC

The full end-to-end flow is in `01_agentic_recommender_poc.ipynb`. The notebook handles:

1. Downloading the MovieLens dataset.
2. Generating the local vector index.
3. Executing the agent graph.
4. Printing the "Trace Log" so you can see exactly when and why the Critic decided to pivot the strategy.

The goal of this project isn't to replace the underlying recommendation models, but to build an orchestration layer that can detect its own failures and adapt.

---

*This Proof of Concept demonstrates the future of "Active Recommendation" where AI agents monitor and optimize the user experience in real-time.*
