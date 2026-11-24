"""Downstream readouts and their evaluation metrics."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrozenMLPReadout(nn.Module):
    def __init__(self, dimension, classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(), nn.Dropout(.2), nn.Linear(256, classes))

    def forward(self, graph, features):
        return self.layers(features.detach())


class UndirectedLinkReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1))

    def forward(self, first_features, second_features):
        symmetric_pair = first_features.detach()*second_features.detach()
        return self.network(symmetric_pair).squeeze(-1)


@torch.no_grad()
def standardization_from_training_nodes(features,train_ids):
    values=features[train_ids].double()
    mean=values.mean(0).float()
    deviation=values.std(0,unbiased=False).float()
    deviation=torch.where(deviation>1e-6,deviation,torch.ones_like(deviation))
    return mean,deviation


def pair_keys(pairs, nodes):
    return np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64) * nodes + np.maximum(pairs[:, 0], pairs[:, 1])


def training_negatives(train_pairs, nodes, seed):
    rng = np.random.default_rng(seed)
    source = np.where(rng.integers(0, 2, len(train_pairs)).astype(bool), train_pairs[:, 0], train_pairs[:, 1])
    target = rng.integers(0, nodes, len(source)); known = pair_keys(train_pairs, nodes)
    while True:
        pairs = np.column_stack([source, target])
        invalid = (source == target) | np.isin(pair_keys(pairs, nodes), known)
        if not invalid.any(): return pairs.astype(np.int64)
        target[invalid] = rng.integers(0, nodes, int(invalid.sum()))


@torch.no_grad()
def score_queries(head, features, positives, negatives, relation_batch_size):
    device = features.device; positive_scores = []; negative_scores = []
    head.eval()
    for start in range(0, len(positives), relation_batch_size):
        p = torch.from_numpy(positives[start:start + relation_batch_size].astype(np.int64)).to(device)
        n = torch.from_numpy(negatives[start:start + relation_batch_size].astype(np.int64)).to(device)
        positive_scores.append(head(features[p[..., 0]], features[p[..., 1]]).cpu())
        anchors = p[..., 0, None].expand_as(n)
        negative_scores.append(head(features[anchors.reshape(-1)], features[n.reshape(-1)]).reshape_as(n).cpu())
    return torch.cat(positive_scores), torch.cat(negative_scores)


def classification_metrics(labels, predictions, num_classes):
    if labels.ndim != 1 or predictions.shape != labels.shape or labels.numel() == 0:
        raise ValueError('Expected matching nonempty label and prediction vectors.')
    if labels.dtype != torch.long or predictions.dtype != torch.long:
        raise ValueError('Class IDs must use torch.long.')
    if ((labels < 0) | (labels >= num_classes) | (predictions < 0) | (predictions >= num_classes)).any():
        raise ValueError('Class IDs outside the declared public vocabulary.')
    confusion = torch.bincount(labels*num_classes+predictions, minlength=num_classes**2)
    confusion = confusion.reshape(num_classes,num_classes).double()
    true_positive = confusion.diag()
    denominator = confusion.sum(0)+confusion.sum(1)
    per_class = torch.where(denominator > 0, 2*true_positive/denominator, 0)
    return {'accuracy':float(true_positive.sum()/labels.numel()),
            'macro_f1':float(per_class.mean()), 'nodes':labels.numel()}


def undirected_link_metrics(positive_scores, negative_scores, k=10):
    """Scores are [relations,2] and [relations,2,K]; each endpoint is a query.

    A tie uses rank 1 + number-greater + 0.5 * number-equal, as in OGB MRR.
    Average per-query reciprocal ranks, then average relations; do not invert
    an averaged rank. Each relation contributes exactly two endpoint queries.
    """
    if positive_scores.ndim != 2 or positive_scores.shape[1] != 2:
        raise ValueError('Two endpoint queries are required per undirected relation.')
    if negative_scores.shape[:-1] != positive_scores.shape or negative_scores.shape[-1] == 0:
        raise ValueError('Negative scores must share both relation/query dimensions.')
    if positive_scores.numel() == 0 or not torch.isfinite(positive_scores).all() or not torch.isfinite(negative_scores).all():
        raise ValueError('Empty or nonfinite evaluation scores.')
    greater = (negative_scores > positive_scores[...,None]).sum(-1).double()
    equal = (negative_scores == positive_scores[...,None]).sum(-1).double()
    rank = 1 + greater + .5 * equal
    return {'mrr':float(rank.reciprocal().mean(1).mean()),
            f'hits_at_{k}':float((rank <= k).double().mean(1).mean()),
            'relations':positive_scores.shape[0], 'queries':positive_scores.numel()}


def fit_node_readout(features, labels, partition, num_classes, config, seed):
    device = features.device
    train, validation, test = (torch.from_numpy(partition[name]).to(device)
                               for name in ('train', 'validation', 'test'))
    labels = labels.to(device)
    mean, deviation = standardization_from_training_nodes(features, train)
    x = (features - mean) / deviation
    torch.manual_seed(seed)
    head = FrozenMLPReadout(x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    best = -1.
    stale_epochs = 0
    for epoch in range(1, config['epochs'] + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(None, x)[train], labels[train])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite node readout loss.')
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            metrics = classification_metrics(labels[validation], head(None, x)[validation].argmax(1),
                                             num_classes)
        if metrics['accuracy'] > best:
            best = metrics['accuracy']
            best_metrics = metrics
            best_state = deepcopy(head.state_dict())
            selected_epoch, stale_epochs = epoch, 0
        else:
            stale_epochs += 1
            if stale_epochs >= config['patience']:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_metrics = classification_metrics(labels[test], head(None, x)[test].argmax(1), num_classes)
    return {'selected_epoch': selected_epoch, 'validation': best_metrics, 'test': test_metrics}


def fit_link_readout(features, pairs, validation_queries, test_queries, config):
    device = features.device
    train_nodes = torch.from_numpy(np.unique(pairs)).long().to(device)
    mean, deviation = standardization_from_training_nodes(features, train_nodes)
    x = (features - mean) / deviation
    settings = config['readout']
    torch.manual_seed(config['head_seed'])
    head = UndirectedLinkReadout(x.shape[1], settings['hidden_dim']).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    best = -1.
    relation_batch = config['evaluation_relation_batch_size']
    for epoch in range(settings['epochs']):
        head.train()
        negatives = training_negatives(pairs, len(features), config['negative_seed'] + epoch)
        order = np.random.default_rng(config['order_seed'] + epoch).permutation(len(pairs))
        for start in range(0, len(pairs), settings['batch_size']):
            ids = order[start:start + settings['batch_size']]
            batch = np.concatenate([pairs[ids], negatives[ids]], axis=0)
            u, v = torch.from_numpy(batch.T.astype(np.int64)).to(device)
            labels = torch.cat([torch.ones(len(ids), device=device), torch.zeros(len(ids), device=device)])
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(x[u], x[v]), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite link readout loss.')
            loss.backward()
            optimizer.step()

        positive_scores, negative_scores = score_queries(head, x, *validation_queries, relation_batch)
        metrics = undirected_link_metrics(positive_scores, negative_scores)
        if metrics['mrr'] > best:
            best = metrics['mrr']
            best_metrics = metrics
            best_state = deepcopy(head.state_dict())
            selected_epoch = epoch + 1

    head.load_state_dict(best_state)
    positive_scores, negative_scores = score_queries(head, x, *test_queries(), relation_batch)
    return {'selected_epoch': selected_epoch, 'validation': best_metrics,
            'test': undirected_link_metrics(positive_scores, negative_scores)}
