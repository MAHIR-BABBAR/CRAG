# Context Packing

After retrieval, CRAG packs child (and optional parent) chunks under a character budget.
Each block is tagged with a citation index like [1], [2] so an agent can ground answers.

Retrieval quality labels: empty, low, or high — a lightweight signal for agent branching.
