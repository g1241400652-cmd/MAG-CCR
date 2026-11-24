"""Node-local predictors and jointly masked graph reconstruction."""

from copy import deepcopy

import torch
from dgl.nn import GraphConv
from torch import nn
from torch.nn import functional as F

from .protocol import stream_seed


class LocalPredictor(nn.Module):
    def __init__(self, input_mean, target_mean, hidden_dim):
        super().__init__()
        self.register_buffer('input_mean',input_mean.detach().clone())
        self.register_buffer('target_mean',target_mean.detach().clone())
        self.network=nn.Sequential(nn.Linear(input_mean.numel(),hidden_dim),nn.ReLU(),
                                   nn.Linear(hidden_dim,target_mean.numel()))
        # Epoch zero is exactly the fit-only mean reference.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self,x):
        return self.target_mean+self.network(x-self.input_mean)


def make_predictor(x,y,fit_ids,hidden_dim):
    ids=fit_ids.to(x.device)
    return LocalPredictor(x[ids].double().mean(0).float(),
                          y[ids].double().mean(0).float(),hidden_dim).to(x.device)


@torch.no_grad()
def predict_nodes(model,x,ids,batch_size):
    model.eval()
    return torch.cat([model(x[part.to(x.device)]).cpu()
                      for part in ids.split(batch_size)],dim=0)


@torch.no_grad()
def evaluation_mse(model,x,y,ids,batch_size):
    model.eval()
    squared=0.0
    for part in ids.split(batch_size):
        index=part.to(x.device)
        squared+=(model(x[index])-y[index]).double().square().sum().item()
    return squared/(len(ids)*y.shape[1])


def validate_membership(fit_ids,validation_ids,prediction_ids,n):
    groups=[set(map(int,x)) for x in [fit_ids,validation_ids,prediction_ids]]
    if any(not x for x in groups):
        raise ValueError('Fit, internal validation and OOF prediction sets must be nonempty.')
    if any(len(s)!=len(x) for s,x in zip(groups,[fit_ids,validation_ids,prediction_ids])):
        raise ValueError('Duplicate node IDs in membership list.')
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
        raise ValueError('A node cannot be both fitted/selected and predicted by this predictor.')
    if set.union(*groups)!=set(range(n)):
        raise ValueError('The membership lists must partition all allowed unlabeled nodes.')


@torch.no_grad()
def fit_modality_scale(residuals, minimum=1e-6):
    """One scalar over calibration nodes AND coordinates, fitted once per modality."""
    if residuals.ndim != 2 or residuals.numel() == 0:
        raise ValueError('Expected a nonempty node-by-coordinate calibration matrix.')
    return residuals.double().square().mean().sqrt().clamp_min(minimum).to(residuals.dtype)


def fixed_residual(target_features, out_of_fold_prediction):
    return (target_features - out_of_fold_prediction).detach()


def node_huber_errors(prediction_scaled, residual_raw, fixed_scale, delta=1.0):
    """Return one error per node, averaging coordinates within one modality."""
    target_scaled = residual_raw.detach() / fixed_scale.detach()
    return F.huber_loss(prediction_scaled, target_scaled, reduction='none', delta=delta).mean(dim=-1)


def weighted_recovery_loss(errors, node_weights, task_coefficients=(0.5, 0.5)):
    """Fixed task coefficients and detached per-target weights play distinct roles."""
    if errors.shape != node_weights.shape or errors.ndim != 2 or errors.shape[1] != 2:
        raise ValueError('Expected matching node-by-two-modality errors and weights.')
    coefficients = errors.new_tensor(task_coefficients)
    return (errors * node_weights.detach() * coefficients).sum(dim=-1).mean()


def downstream_features(text_features, image_features, graph_features):
    """Only these three frozen inputs; no local predictions or recovery-head outputs."""
    return torch.cat([text_features.detach(), image_features.detach(), graph_features.detach()], dim=-1)


def original_edge_coefficients(graph):
    source,destination=graph.edges(order='eid')
    out_degree=graph.out_degrees().float().clamp_min(1)
    in_degree=graph.in_degrees().float().clamp_min(1)
    return (out_degree[source].rsqrt()*in_degree[destination].rsqrt()).detach()


class GraphEncoder(nn.Module):
    def __init__(self,input_dim=1536,hidden_dim=256,output_dim=256,dropout=.1):
        super().__init__()
        self.mask_token=nn.Parameter(torch.zeros(input_dim))
        self.first=GraphConv(input_dim,hidden_dim,norm='none')
        self.normalization=nn.LayerNorm(hidden_dim)
        self.dropout=nn.Dropout(dropout)
        self.second=GraphConv(hidden_dim,output_dim,norm='none')

    def forward(self,graph,features,edge_coefficients,masked_nodes):
        if self.training and masked_nodes is None:
            raise ValueError('Unmasked feature extraction requires a frozen evaluation encoder.')
        if masked_nodes is not None:
            selected=torch.zeros(len(features),dtype=torch.bool,device=features.device)
            selected.index_fill_(0,masked_nodes,True)
            x=torch.where(selected[:,None],self.mask_token[None,:],features)
        else:x=features
        h=self.first(graph,x,edge_weight=edge_coefficients)
        h=self.dropout(torch.relu(self.normalization(h)))
        return self.second(graph,h,edge_weight=edge_coefficients)


class RecoveryHead(nn.Module):
    def __init__(self,graph_dim,hidden_dim,target_mean):
        super().__init__()
        self.register_buffer('target_mean',target_mean.detach().clone())
        self.layers=nn.Sequential(nn.Linear(graph_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,len(target_mean)))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self,h):return self.target_mean+self.layers(h)


class MaskedRecoveryModel(nn.Module):
    def __init__(self,target_means,hidden_dim=256,graph_dim=256,dropout=.1):
        super().__init__()
        self.encoder=GraphEncoder(2*target_means.shape[1],hidden_dim,graph_dim,dropout)
        self.text_head=RecoveryHead(graph_dim,hidden_dim,target_means[0])
        self.image_head=RecoveryHead(graph_dim,hidden_dim,target_means[1])

    def masked_predictions(self,graph,features,coefficients,masked_nodes):
        if masked_nodes is None or not len(masked_nodes):
            raise ValueError('Recovery requires nonempty explicitly masked target IDs.')
        h=self.encoder(graph,features,coefficients,masked_nodes)[masked_nodes]
        return self.text_head(h),self.image_head(h)


def recovery_errors(predictions,raw_targets,scales,target_ids,delta=1.):
    return torch.stack([node_huber_errors(p,t[target_ids],s,delta)
                        for p,t,s in zip(predictions,raw_targets,scales)],dim=1)


class StableGraphConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, graph, features, edge_weight):
        source, destination = graph.edges(order='eid')
        transform_first = self.weight.shape[0] > self.weight.shape[1]
        x = features @ self.weight if transform_first else features
        messages = x[source.long()] * edge_weight[:, None]
        out = x.new_zeros((graph.num_nodes(), x.shape[1]))
        out.index_add_(0, destination.long(), messages)
        if not transform_first: out = out @ self.weight
        return out + self.bias


def use_stable_convolutions(model):
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('Stable graph aggregation requires Torch deterministic algorithms')
    model.encoder.first = StableGraphConv(model.encoder.first)
    model.encoder.second = StableGraphConv(model.encoder.second)
    return model


def fit_predictor(x, y, fit_ids, validation_ids, seed, config):
    torch.manual_seed(seed)
    model = make_predictor(x, y, fit_ids, config['hidden_dim'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    sampler = torch.Generator().manual_seed(seed + 1)
    eval_batch = config['evaluation_batch_size']
    best_mse = evaluation_mse(model, x, y, validation_ids, eval_batch)
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0

    for _ in range(config['maximum_epochs']):
        model.train()
        order = fit_ids[torch.randperm(len(fit_ids), generator=sampler)]
        for ids in order.split(config['batch_size']):
            ids = ids.to(x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x[ids]), y[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite local prediction loss.')
            loss.backward()
            optimizer.step()

        mse = evaluation_mse(model, x, y, validation_ids, eval_batch)
        if mse < best_mse:
            best_mse, stale_epochs = mse, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= config['patience']:
                break

    model.load_state_dict(best_state)
    return model.eval()


def fit_residuals(features, partitions, dataset, root_seed, repeat, config):
    residuals = []
    names = ('text', 'image')
    for index, target_name in enumerate(names):
        source = features[1 - index]
        source_name = names[1 - index]
        target = features[index]
        prediction = torch.empty_like(target, device='cpu')
        for fold in (0, 1):
            fit, validation, predict = [
                torch.from_numpy(partitions[f'{name}_{fold}'])
                for name in ('fit', 'local_validation', 'predict')
            ]
            validate_membership(fit, validation, predict, len(target))
            seed = stream_seed(root_seed, dataset, f'local_{source_name}_to_{target_name}_fold{fold}', repeat)
            model = fit_predictor(source, target, fit, validation, seed, config)
            prediction[predict] = predict_nodes(model, source, predict, config['evaluation_batch_size'])
        residuals.append(fixed_residual(target.cpu(), prediction))
    return residuals


def fit_encoder(graph, features, targets, scales, eligible, calibration, config):
    torch.manual_seed(config['encoder_seed'])
    calibration = calibration.to(features.device)
    means = torch.stack([(target[calibration] / scale).mean(0)
                         for target, scale in zip(targets, scales)])
    model = MaskedRecoveryModel(means, **config['architecture']).to(features.device)
    use_stable_convolutions(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    coefficients = original_edge_coefficients(graph)
    mask_generator = torch.Generator().manual_seed(config['mask_seed'])
    mask_count = int(config['mask_ratio'] * len(eligible))
    if mask_count < 1:
        raise ValueError('The eligible set is too small to sample a mask.')
    for _ in range(config['pretraining_epochs']):
        selection = torch.randperm(len(eligible), generator=mask_generator)[:mask_count]
        masked = eligible[selection].to(features.device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model.masked_predictions(graph, features, coefficients, masked)
        errors = recovery_errors(predictions, targets, scales, masked, config['huber_delta'])
        loss = weighted_recovery_loss(errors, torch.ones_like(errors), config['task_coefficients'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite reconstruction loss.')
        loss.backward()
        optimizer.step()
    model.encoder.requires_grad_(False).eval()
    with torch.no_grad():
        return model.encoder(graph, features, coefficients, None)
