"""Deterministic node, fold and undirected relation splits."""
import hashlib

import numpy as np
import torch


def crossfit_fold_ids(number_of_nodes, seed):
    """Independent of downstream label partitions; use the opposite fold to fit."""
    if number_of_nodes < 2:
        raise ValueError('Two-fold prediction needs at least two nodes.')
    order = torch.randperm(number_of_nodes, generator=torch.Generator().manual_seed(seed))
    folds = torch.zeros(number_of_nodes, dtype=torch.long)
    folds[order[number_of_nodes // 2:]] = 1
    return folds


def stream_seed(root_seed, dataset, stream, repeat=0):
    key = f'{root_seed}/{dataset}/{stream}/{repeat}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')


def node_partition(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    a, b = int(.6 * n), int(.2 * n)
    return {'train':order[:a], 'validation':order[a:a+b], 'test':order[a+b:]}


def node_analysis_sets(n, enabled, seed):
    count = int(.1 * n) if enabled else 0
    analysis = np.sort(np.random.default_rng(seed).permutation(n)[:count])
    eligible = np.setdiff1d(np.arange(n, dtype=np.int64), analysis)
    return {'analysis':analysis, 'gnn_recovery_targets':eligible,
            'scale_calibration':eligible.copy(), 'weight_update_targets':eligible.copy(),
            'case_ids':analysis[:0] if not enabled else
                np.random.default_rng(seed+1).permutation(analysis)[:6]}


def crossfit_partition(n, seed):
    if n < 4:
        raise ValueError('Each fitted fold needs distinct fit and internal-validation nodes.')
    folds = crossfit_fold_ids(n, seed).numpy()
    result = {'fold_id':folds}
    for predicted_fold in [0, 1]:
        predict = np.flatnonzero(folds == predicted_fold)
        allowed = np.flatnonzero(folds != predicted_fold)
        shuffled = np.random.default_rng(seed + predicted_fold + 1).permutation(allowed)
        validation_count = max(1, int(.2 * len(allowed)))
        result[f'predict_{predicted_fold}'] = predict
        result[f'local_validation_{predicted_fold}'] = shuffled[:validation_count]
        result[f'fit_{predicted_fold}'] = shuffled[validation_count:]
    return result


def canonical_relations(edge_index):
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('edge_index must have shape [2, E].')
    pairs = np.sort(edges.T, axis=1)
    return np.unique(pairs[pairs[:,0] != pairs[:,1]], axis=0)


def relation_partition(pairs, seed):
    order = np.random.default_rng(seed).permutation(len(pairs))
    validation_count, test_count = int(.1 * len(pairs)), int(.2 * len(pairs))
    return {'train':np.sort(order[validation_count+test_count:]),
            'validation':np.sort(order[:validation_count]),
            'test':np.sort(order[validation_count:validation_count+test_count])}


def propagation_edges(pairs):
    return np.concatenate([pairs.T, pairs[:,::-1].T], axis=1)


def adjacency_lists(pairs, n):
    adjacent = [set() for _ in range(n)]
    for a, b in pairs:
        adjacent[int(a)].add(int(b))
        adjacent[int(b)].add(int(a))
    return adjacent


def negative_candidates(anchors, adjacent, n, count, seed):
    """Uniform distinct targets conditional on a supplied forbidden adjacency.

    Training supplies train-only adjacency; evaluation supplies all known
    positives. This function does not look up or load a different graph.
    """
    rng = np.random.default_rng(seed)
    result = np.empty((len(anchors), count), dtype=np.int32)
    for row, anchor in enumerate(anchors):
        anchor = int(anchor)
        forbidden = adjacent[anchor] | {anchor}
        if n - len(forbidden) < count:
            raise ValueError('Insufficient distinct non-neighbors for the fixed candidate count.')
        selected, seen = [], set()
        while len(selected) < count:
            draws = rng.integers(0, n, size=max(32, 2 * (count-len(selected))))
            for value in draws:
                value = int(value)
                if value not in forbidden and value not in seen:
                    selected.append(value)
                    seen.add(value)
                    if len(selected) == count:
                        break
        result[row] = selected
    return result
