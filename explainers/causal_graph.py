"""
causal_graph.py

Loads the causal DAG from config and computes causal coalition weights.
P(S | do(i), G) is approximated by counting d-connected paths between
features in S and feature i in graph G, given the do(i) intervention.

The weight reflects how causally plausible it is that coalition S would
naturally form around feature i in the real data generating process.
"""

import yaml
import numpy as np
from itertools import combinations


class CausalGraph:
    def __init__(self, dag_config_path: str, feature_names: list):
        """
        Parameters:
            dag_config_path: path to causal_dag.yaml
            feature_names: list of feature names in model order
        """
        with open(dag_config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.feature_to_idx = {f: i for i, f in enumerate(feature_names)}

        # Build adjacency: edges are directed cause -> effect
        self.edges = []
        self.adjacency = {f: set() for f in feature_names}
        self.reverse_adjacency = {f: set() for f in feature_names}

        for edge in config.get('edges', []):
            cause, effect = edge[0], edge[1]
            if cause in self.feature_to_idx and effect in self.feature_to_idx:
                self.edges.append((cause, effect))
                self.adjacency[cause].add(effect)
                self.reverse_adjacency[effect].add(cause)

        self.protected_attributes = config.get('protected_attributes', [])
        self.proxy_candidates = config.get('proxy_candidates', [])

        # Precompute causal distances between all feature pairs
        # using BFS on undirected version of the graph (d-connectivity proxy)
        self._causal_distances = self._compute_all_distances()

    def _compute_all_distances(self) -> dict:
        """
        BFS distance between all pairs of features on undirected graph.
        Used to estimate d-connectivity: closer features = more causally related.
        """
        distances = {}
        for feature in self.feature_names:
            distances[feature] = self._bfs_distances(feature)
        return distances

    def _bfs_distances(self, source: str) -> dict:
        """BFS shortest path distances from source on undirected graph."""
        visited = {source: 0}
        queue = [source]
        while queue:
            current = queue.pop(0)
            neighbors = self.adjacency[current] | self.reverse_adjacency[current]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        # Features not reachable get distance = infinity
        for f in self.feature_names:
            if f not in visited:
                visited[f] = float('inf')
        return visited

    def causal_plausibility(self, feature_i: str, feature_j: str) -> float:
        """
        Returns a score in [0, 1] representing how causally related
        feature_j is to feature_i.

        Score = 1.0 if direct edge exists
        Score = 0.5 if distance 2 (one intermediate)
        Score = 1/(distance) for further
        Score = 0.0 if unreachable (d-separated)
        """
        if feature_i not in self._causal_distances:
            return 0.0

        dist = self._causal_distances[feature_i].get(feature_j, float('inf'))

        if feature_i == feature_j:
            return 0.0  # feature i never in its own coalition
        elif dist == 1:
            return 1.0  # direct causal relationship
        elif dist == 2:
            return 0.5  # one intermediary
        elif dist == float('inf'):
            return 0.0  # d-separated — no causal path
        else:
            return 1.0 / dist  # diminishing plausibility with distance

    def coalition_weight(self, coalition: frozenset, feature_i: str) -> float:
        """
        Computes w^causal(S, i, G) = P(S | do(i), G)

        Approximated as the product of causal plausibility scores
        for each member j of coalition S with respect to feature i.

        Product form ensures that a coalition with one d-separated feature
        gets near-zero weight overall — the double gate effect.

        Parameters:
            coalition: frozenset of feature names in coalition S
            feature_i: the feature whose SHAP value we are computing

        Returns:
            float: unnormalised causal weight for this coalition
        """
        if len(coalition) == 0:
            # Empty coalition — baseline plausibility = 1.0
            return 1.0

        # Product of individual plausibility scores
        weight = 1.0
        for feature_j in coalition:
            plausibility = self.causal_plausibility(feature_i, feature_j)
            weight *= plausibility

            # Early exit if weight already zero
            if weight == 0.0:
                return 0.0

        return weight

    def get_causal_neighbourhood(self, feature_i: str, max_distance: int = 2) -> list:
        """
        Returns features within max_distance causal steps of feature_i.
        Used to restrict coalition search space.

        Parameters:
            feature_i: target feature
            max_distance: maximum causal distance to include

        Returns:
            list of feature names in causal neighbourhood
        """
        if feature_i not in self._causal_distances:
            return [f for f in self.feature_names if f != feature_i]

        neighbourhood = []
        for f, dist in self._causal_distances[feature_i].items():
            if f != feature_i and dist <= max_distance:
                neighbourhood.append(f)

        return neighbourhood

    def get_all_coalition_weights(self, feature_i: str, all_features: list) -> dict:
        """
        Precomputes normalised causal weights for ALL possible coalitions
        of features in all_features excluding feature_i.

        Returns dict: frozenset(coalition) -> normalised_weight

        This is called once per feature per SHAP computation to avoid
        recomputing weights for every sample.
        """
        other_features = [f for f in all_features if f != feature_i]
        n = len(other_features)

        raw_weights = {}

        # Enumerate all 2^n subsets
        for size in range(n + 1):
            for combo in combinations(other_features, size):
                coalition = frozenset(combo)
                raw_weights[coalition] = self.coalition_weight(coalition, feature_i)

        # Normalise so all weights sum to 1
        total = sum(raw_weights.values())
        if total == 0:
            # Fallback to uniform if all weights zero (disconnected graph)
            uniform = 1.0 / len(raw_weights)
            return {k: uniform for k in raw_weights}

        return {k: v / total for k, v in raw_weights.items()}
