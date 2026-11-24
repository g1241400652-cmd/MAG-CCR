"""Run MAG-CCR or its matched full-feature control."""
import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import dgl
import numpy as np
import torch

from .model import downstream_features, fit_encoder, fit_modality_scale, fit_residuals
from .protocol import (adjacency_lists, canonical_relations, crossfit_partition,
                       negative_candidates, node_analysis_sets, node_partition,
                       propagation_edges, relation_partition, stream_seed)
from .readout import fit_link_readout, fit_node_readout


def load_openmag(data_dir, dataset):
    directory = data_dir / dataset
    raw_graph = dgl.load_graphs(str(directory / f'{dataset}Graph.pt'))[0][0]
    edges = torch.stack(raw_graph.edges(order='eid')).numpy()
    modalities = []
    for name in ('text', 'image'):
        filename = f'{dataset}_{name}_clip_vit_large_patch14_768d.npy'
        features = np.load(directory / f'{name}_features' / filename, allow_pickle=False)
        if features.shape != (raw_graph.num_nodes(), 768) or features.dtype != np.float32:
            raise ValueError(f'Unexpected feature shape or dtype in {filename}')
        modalities.append(torch.from_numpy(features))
    return raw_graph, edges, modalities


def demo_data():
    n = 64
    generator = torch.Generator().manual_seed(17)
    text = torch.randn(n, 8, generator=generator)
    image = .8 * text + .4 * torch.randn(n, 8, generator=generator)
    x = np.arange(n)
    edges = np.stack([np.tile(x, 2), np.concatenate([(x + 1) % n, (x + 2) % n])])
    graph = dgl.graph((edges[0], edges[1]), num_nodes=n)
    graph.ndata['label'] = text[:, :3].argmax(1)
    return graph, edges, (text, image)


def make_queries(positive_pairs, adjacency, n, count, seed):
    positives = np.stack([positive_pairs, positive_pairs[:, ::-1]], axis=1)
    anchors = positives[..., 0].reshape(-1)
    negatives = negative_candidates(anchors, adjacency, n, count, seed)
    return positives, negatives.reshape(len(positives), 2, count)


def run(args, protocol):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    dataset = 'Demo' if args.demo else args.dataset
    config = deepcopy(protocol['presets'][args.task])
    seeds = protocol['seeds'][args.task]['Grocery' if args.demo else dataset][args.seed]
    config.update(seeds)
    local_config = dict(protocol['local_predictor'])
    candidates = protocol['lp_negative_candidates']
    if args.demo:
        local_config.update(maximum_epochs=2, hidden_dim=16)
        config.update(pretraining_epochs=2, architecture=dict(hidden_dim=16, graph_dim=8, dropout=.1))
        config['readout']['epochs'] = 2
        candidates = 16
        raw_graph, edges, modalities = demo_data()
    else:
        raw_graph, edges, modalities = load_openmag(args.data_dir, dataset)
    n = raw_graph.num_nodes()
    pairs = canonical_relations(edges)
    features = tuple(values.to(device) for values in modalities)
    root = protocol['root_seed']
    eligible_sets = node_analysis_sets(n, dataset in protocol['analysis_datasets'],
                                       stream_seed(root, dataset, 'Q3_analysis'))
    eligible = torch.from_numpy(eligible_sets['gnn_recovery_targets'])
    calibration = torch.from_numpy(eligible_sets['scale_calibration'])
    if args.task == 'lp':
        partition = relation_partition(pairs, stream_seed(root, dataset, 'LP_split'))
        train_pairs = pairs[partition['train']].astype(np.int32)
        message_pairs = train_pairs.astype(np.int64)
    else:
        partition = node_partition(n, stream_seed(root, dataset, 'NC_split'))
        message_pairs = pairs

    source, destination = propagation_edges(message_pairs)
    graph = dgl.add_self_loop(dgl.graph((source, destination), num_nodes=n)).to(device)
    if args.target == 'residual':
        folds = crossfit_partition(n, stream_seed(root, dataset, 'OOF', args.seed))
        targets = fit_residuals(features, folds, dataset, root, args.seed, local_config)
    else:
        targets = modalities
    if args.task == 'nc':
        scales = torch.stack([fit_modality_scale(target[calibration], local_config['scale_floor'])
                              for target in targets]).to(device)
    else:
        scales = torch.ones(2, device=device)
    joined = torch.cat(features, dim=1)
    embedding = fit_encoder(graph, joined, [x.to(device) for x in targets], scales, eligible,
                            calibration, config)
    readout_input = downstream_features(*features, embedding)

    if args.task == 'nc':
        labels = raw_graph.ndata['label'].flatten().long()
        classes = 3 if args.demo else protocol['public_num_classes'][dataset]
        metrics = fit_node_readout(readout_input, labels, partition, classes, config['readout'],
                                   config['head_seed'])
    else:
        adjacency = adjacency_lists(pairs, n)

        def queries(split):
            return make_queries(pairs[partition[split]].astype(np.int32), adjacency, n, candidates,
                                stream_seed(root, dataset, 'LP_candidates_' + split))

        metrics = fit_link_readout(readout_input, train_pairs, queries('validation'),
                                   lambda: queries('test'), config)
    return {'dataset': dataset, 'task': args.task, 'target': args.target, 'seed': args.seed,
            'target_scales': scales.tolist(), **metrics}


def main():
    protocol = json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('nc', 'lp'), default='nc')
    parser.add_argument('--target', choices=('residual', 'full-feature'), default='residual')
    parser.add_argument('--dataset', default='Grocery')
    parser.add_argument('--data-dir', type=Path, default=Path('openmag'))
    parser.add_argument('--seed', type=int, choices=range(5), default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--demo', action='store_true', help='Run two epochs per stage on a synthetic graph.')
    args = parser.parse_args()
    if not args.demo and args.dataset not in protocol['seeds'][args.task]:
        parser.error(f"Datasets for {args.task}: {', '.join(protocol['seeds'][args.task])}")
    print(json.dumps(run(args, protocol), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
