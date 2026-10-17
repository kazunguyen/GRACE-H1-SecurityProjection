"""Small residual metric learner over cached embeddings; train/validation only.

Local label relevance is a retrieval proxy, not demonstration utility for an LLM.
The teacher skip path and bounded residual limit geometric changes. No torch is
imported by the NumPy preparation path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import os
import time
import numpy as np

from .data import load_training_data, embedding_recipe
from .io import resolve_root, fingerprint, file_hash, read_json, write_json
from .metrics import binary_metrics
from .projection import normalization
from .retrieval import common_candidates

SCHEMA = 'grace_projection_v2'


@dataclass(frozen=True)
class ProjectionV2Config:
    objective: str = 'residual_local'
    dimension: int = 256
    hidden_dimension: int = 256
    residual_radius: float = 0.25
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    temperature: float = 0.1
    margin: float = 0.2
    local_weight: float = 0.1
    semi_hard_weight: float = 0.05
    geometry_weight: float = 1.0
    positive_neighbors: int = 8
    negative_neighbors: int = 32
    negatives_per_anchor: int = 8
    hard_warmup_epochs: int = 3
    hard_ramp_epochs: int = 5
    patience: int = 8
    seed: int = 42
    selection_min_gain: float = 0.002
    selection_max_class_drop: float = 0.02
    selection_min_teacher_overlap: float = 0.5
    device: str = 'auto'

    def validate(self):
        if self.objective not in ('residual_ce', 'residual_local'):
            raise ValueError('Unsupported V2 objective')
        sizes = (self.dimension, self.hidden_dimension, self.epochs, self.patience,
                 self.positive_neighbors, self.negative_neighbors, self.negatives_per_anchor,
                 self.hard_ramp_epochs)
        if min(sizes) < 1 or self.batch_size < 4 or self.batch_size % 2:
            raise ValueError('Positive sizes and an even batch_size >= 4 required')
        if not 0 <= self.dropout < 1 or not 0 < self.residual_radius < 1:
            raise ValueError('Invalid dropout or residual bound')
        if self.learning_rate <= 0 or self.temperature <= 0 or self.hard_warmup_epochs < 0:
            raise ValueError('Invalid optimization scales')
        if min(self.weight_decay, self.margin, self.local_weight, self.semi_hard_weight,
               self.geometry_weight, self.selection_min_gain, self.selection_max_class_drop) < 0:
            raise ValueError('Loss weights and tolerances must be nonnegative')
        if not 0 <= self.selection_min_teacher_overlap <= 1:
            raise ValueError('Overlap must be in [0,1]')


def normalize_np(x):
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)).astype(np.float32)


def fit_teacher(x, dimension):
    """Same train-only whitening rule as the paired baseline."""
    if dimension > x.shape[1]:
        raise ValueError('Teacher dimension exceeds encoder dimension')
    mean = np.asarray(x, dtype=np.float64).mean(0)
    centered = np.asarray(x, dtype=np.float64) - mean
    cov = centered.T @ centered / max(len(x) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:dimension]
    kernel = vectors[:, order] / np.sqrt(np.maximum(values[order], 1e-8))[None, :]
    return mean, kernel, normalize_np(centered @ kernel)


def training_indices(rows):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['code_sha256'], []).append(i)
    return np.asarray([ids[0] for ids in groups.values()
                       if len({rows[i]['label'] for i in ids}) == 1], dtype=np.int64)


def balanced_batches(labels, batch_size, seed):
    """Visit majority once; repeat shuffled minority cycles; equal anchor counts."""
    if batch_size < 4 or batch_size % 2:
        raise ValueError('Use an even anchor batch size >= 4')
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(labels == c) for c in (0, 1)]
    if min(map(len, classes)) < 2:
        raise ValueError('Each training class needs at least two unique codes')
    length = max(map(len, classes))
    streams = [np.concatenate([rng.permutation(ids) for _ in range(math.ceil(length / len(ids)))])[:length]
               for ids in classes]
    half = batch_size // 2
    for start in range(0, length, half):
        yield rng.permutation(np.concatenate([s[start:start + half] for s in streams]))


def mine_local_bank(teacher, labels, positive_k=8, negative_k=32, chunk=256):
    """Indices are local to the deduplicated train bank; no validation labels."""
    labels = np.asarray(labels)
    groups = [np.flatnonzero(labels == c) for c in (0, 1)]
    if min(map(len, groups)) < 2:
        raise ValueError('Need >=2 unique samples per class for positive mining')
    pk = min(positive_k, min(map(len, groups)) - 1)
    nk = min(negative_k, min(map(len, groups)))
    positives = np.empty((len(labels), pk), dtype=np.int64)
    negatives = np.empty((len(labels), nk), dtype=np.int64)
    for c in (0, 1):
        anchors, other = groups[c], groups[1 - c]
        for start in range(0, len(anchors), chunk):
            ids = anchors[start:start + chunk]
            scores = teacher[ids] @ teacher[anchors].T
            scores[np.arange(len(ids)), np.searchsorted(anchors, ids)] = -np.inf
            positions = np.argsort(-scores, axis=1, kind='stable')[:, :pk]
            positives[ids] = anchors[positions]
            scores = teacher[ids] @ teacher[other].T
            positions = np.argsort(-scores, axis=1, kind='stable')[:, :nk]
            negatives[ids] = other[positions]
        print(f'mined class {c}: {len(anchors)} anchors, positive k={pk}, negative k={nk}', flush=True)
    return positives, negatives


def build_model(input_dimension, config):
    import torch
    from torch import nn

    class ResidualHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dimension, config.hidden_dimension)
            self.fc2 = nn.Linear(config.hidden_dimension, config.dimension)
            self.dropout = nn.Dropout(config.dropout)
            self.classifier = nn.Linear(config.dimension, 2)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, standardized, teacher):
            residual = self.fc2(self.dropout(nn.functional.gelu(self.fc1(standardized), approximate='tanh')))
            delta = residual / (1 + torch.linalg.vector_norm(residual, dim=-1, keepdim=True))
            z = nn.functional.normalize(teacher + config.residual_radius * delta, dim=-1, eps=1e-8)
            return z, self.classifier(z)

    return ResidualHead()


def metric_losses(anchor, positive, negative, teacher, temperature, margin, ids=None):
    import torch
    import torch.nn.functional as F
    pos_sim = (anchor * positive).sum(-1)
    neg_sim = (anchor[:, None, :] * negative).sum(-1)
    logits = torch.cat((pos_sim[:, None], neg_sim), dim=1) / temperature
    local = F.cross_entropy(logits, torch.zeros(len(anchor), device=anchor.device, dtype=torch.long))
    active = (neg_sim < pos_sim[:, None]) & (neg_sim > pos_sim[:, None] - margin)
    semi = (torch.relu(margin + neg_sim - pos_sim[:, None]) * active).sum() / active.sum().clamp_min(1)
    mask = ~torch.eye(len(anchor), device=anchor.device, dtype=torch.bool)
    if ids is not None:
        mask &= ids[:, None].ne(ids[None, :])
    difference = (anchor @ anchor.T - teacher @ teacher.T).square()
    geometry = difference[mask].mean() if mask.any() else anchor.sum() * 0
    return local, semi, geometry, float(active.float().mean().detach())


def ranked_top5(train_vectors, query_vectors, candidate_positions, candidate_ids):
    result = []
    for query, positions, ids in zip(query_vectors, candidate_positions, candidate_ids):
        distances = ((train_vectors[positions] - query) ** 2).sum(1)
        # Match shortlist() distance then canonical ID tie-breaking.
        order = sorted(range(len(positions)), key=lambda j: (float(distances[j]), ids[j]))[:5]
        result.append(np.asarray(positions)[order])
    return result


def retrieval_metrics(ranked, baseline_ranked, train_labels, query_labels):
    precision, hit, overlap = [], [], []
    for a, b, label in zip(ranked, baseline_ranked, query_labels):
        relevant = train_labels[a] == label
        precision.append(float(relevant.mean()))
        hit.append(float(relevant.any()))
        overlap.append(len(set(a) & set(b)) / len(b))
    per_class = {str(c): float(np.asarray(precision)[query_labels == c].mean()) for c in (0, 1)}
    if not all(np.isfinite(list(per_class.values()))):
        raise ValueError('Both independent validation classes are required')
    return {'macro_label_precision_at_5': float(np.mean(list(per_class.values()))),
            'per_class_label_precision_at_5': per_class,
            'macro_label_hit_at_5': float(np.mean([np.asarray(hit)[query_labels == c].mean() for c in (0, 1)])),
            'teacher_top5_overlap': float(np.mean(overlap)), 'validation_queries': len(query_labels),
            'scope': 'label relevance proxy in common top50; not LLM demonstration utility'}


def acceptable(candidate, baseline, cfg):
    return (candidate['macro_label_precision_at_5'] >= baseline['macro_label_precision_at_5'] + cfg.selection_min_gain
            and candidate['teacher_top5_overlap'] >= cfg.selection_min_teacher_overlap
            and all(candidate['per_class_label_precision_at_5'][c] >= value - cfg.selection_max_class_drop
                    for c, value in baseline['per_class_label_precision_at_5'].items()))


def save_arrays(path, arrays):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def project_arrays(matrix, weights, batch_size=2048):
    output = []
    for start in range(0, len(matrix), batch_size):
        e = np.asarray(matrix[start:start + batch_size], dtype=np.float32)
        teacher = normalize_np((e.astype(np.float64) - weights['teacher_mean']) @ weights['teacher_kernel'])
        u = (e - weights['mean']) / weights['scale']
        h = u @ weights['fc1_weight'].T + weights['fc1_bias']
        h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
        residual = h @ weights['fc2_weight'].T + weights['fc2_bias']
        delta = residual / (1 + np.linalg.norm(residual, axis=1, keepdims=True))
        output.append(normalize_np(teacher + float(weights['residual_radius']) * delta))
    return np.concatenate(output) if output else np.empty((0, weights['teacher_kernel'].shape[1]), dtype=np.float32)


def project_embeddings_v2(matrix, manifest, root, batch_size=2048):
    root = resolve_root(root)
    path = root / manifest['path']
    path.resolve().relative_to(root)
    if manifest.get('schema') != SCHEMA or file_hash(path) != manifest['projection_sha256']:
        raise ValueError('V2 projection schema or checksum differs')
    with np.load(path, allow_pickle=False) as stored:
        weights = {key: stored[key] for key in stored.files}
    if matrix.shape[1] != len(weights['mean']) or not all(np.isfinite(a).all() for a in weights.values()):
        raise ValueError('Invalid V2 input dimension/weights')
    return project_arrays(matrix, weights, batch_size)


def train_projection_v2(root, dataset, config=None, *, stop_after_epoch=None):
    """stop_after_epoch is an interruption boundary for resume checks, not a subset benchmark."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import torch.nn.functional as F
    root = resolve_root(root)
    cfg = config or ProjectionV2Config()
    cfg.validate()
    train, valid, x, v, tm, vm, vi = load_training_data(root, dataset)
    ti = training_indices(train)
    vi = np.asarray(vi, dtype=np.int64)
    labels = np.asarray([r['label'] for r in train], dtype=np.int64)
    y = labels[ti]
    vy = np.asarray([r['label'] for r in valid], dtype=np.int64)[vi]
    if len(np.unique(vy)) != 2:
        raise ValueError('Need both independent validation classes')
    source_names = ('projection_v2.py', 'projection.py', 'data.py', 'retrieval.py', 'reference.py', 'metrics.py')
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in source_names}
    spec = {'config': asdict(cfg), 'dataset': dataset, 'train': tm, 'validation': vm,
            'train_indices': ti.tolist(), 'validation_indices': vi.tolist(), 'sources': sources}
    signature = fingerprint(spec)
    folder = root / 'artifacts' / 'projection_v2' / dataset / cfg.objective / signature[:16]
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / 'manifest.json'
    if final.exists():
        manifest = read_json(final)
        project_embeddings_v2(x[:1], manifest, root)
        return manifest
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else cfg.device
    torch.manual_seed(cfg.seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    teacher_mean, teacher_kernel, teacher_train = fit_teacher(x, cfg.dimension)
    teacher_valid = normalize_np((np.asarray(v, dtype=np.float64) - teacher_mean) @ teacher_kernel)
    mean, scale = normalization(x, ti)
    model = build_model(x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    clean_valid = [{k: value for k, value in row.items() if k != 'label'} for row in valid]
    common = common_candidates(root, dataset, 'validation', clean_valid, train)
    lookup = {row['record_id']:i for i, row in enumerate(train)}
    ids = [common[i]['initial_candidate_ids'] for i in vi]
    positions = [np.asarray([lookup[rid] for rid in values]) for values in ids]
    baseline_rank = ranked_top5(teacher_train, teacher_valid[vi], positions, ids)
    baseline = retrieval_metrics(baseline_rank, baseline_rank, labels, vy)
    bank_key = fingerprint({'train':tm['files'], 'indices':ti.tolist(), 'dimension':cfg.dimension,
                            'positive_k':cfg.positive_neighbors, 'negative_k':cfg.negative_neighbors,
                            'source':sources['projection_v2.py']})
    bank_path = root / 'cache' / 'projection_v2' / (bank_key + '.npz')
    bank_meta = bank_path.with_suffix('.json')
    if bank_path.exists() and bank_meta.exists():
        if read_json(bank_meta)['sha256'] != file_hash(bank_path):
            raise ValueError('Mining bank checksum differs')
        with np.load(bank_path, allow_pickle=False) as b:
            positives, negatives = b['positives'], b['negatives']
    else:
        positives, negatives = mine_local_bank(teacher_train[ti], y, cfg.positive_neighbors, cfg.negative_neighbors)
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        save_arrays(bank_path, {'positives':positives, 'negatives':negatives})
        write_json(bank_meta, {'sha256':file_hash(bank_path), 'train_identity':tm['rows_identity'],
                               'labels_scope':'train unique/unambiguous only', 'key':bank_key})
    def arrays():
        return dict(mean=mean, scale=scale, teacher_mean=teacher_mean, teacher_kernel=teacher_kernel,
                    residual_radius=np.asarray(cfg.residual_radius, dtype=np.float32),
                    **{name.replace('.', '_'): value.detach().cpu().numpy() for name,value in model.state_dict().items()})
    def tensors(matrix, teacher, indices):
        raw = (np.asarray(matrix[indices], dtype=np.float32) - mean) / scale
        return torch.from_numpy(raw).to(device), torch.from_numpy(teacher[indices]).to(device)
    def evaluate():
        model.eval()
        with torch.no_grad():
            tv, qv, predictions = [], [], []
            for matrix, teachers, output in ((x, teacher_train, tv), (v, teacher_valid, qv)):
                for start in range(0,len(matrix),2048):
                    indices=np.arange(start,min(start+2048,len(matrix)))
                    z, logits = model(*tensors(matrix, teachers, indices))
                    output.append(z.cpu().numpy())
                    if matrix is v: predictions.extend(logits.argmax(-1).cpu().tolist())
        rank = ranked_top5(np.concatenate(tv), np.concatenate(qv)[vi], positions, ids)
        result = retrieval_metrics(rank, baseline_rank, labels, vy)
        result['classifier_mcc_proxy'] = binary_metrics(vy, np.asarray(predictions)[vi])['mcc']
        return result
    history = []
    start, stale = 0, 0
    best = {'epoch':0, 'retrieval':baseline, 'selection_status':'baseline_fallback'}
    checkpoint = folder / 'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved['fingerprint'] != signature or saved['device'] != device:
            raise ValueError('Resume identity/device differs')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda'): torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        start, stale, best, history = saved['epoch'] + 1, saved['stale'], saved['best'], saved['history']
    else:
        save_arrays(folder / 'projection.npz', arrays())
        write_json(folder / 'best.json', best)
    started = time.perf_counter()
    for epoch in range(start, cfg.epochs):
        if stale >= cfg.patience: break
        model.train()
        rng = np.random.default_rng(cfg.seed + epoch)
        totals=np.zeros(5,dtype=np.float64); count=0
        for anchor_ids in balanced_batches(y, cfg.batch_size, cfg.seed + epoch):
            pos_ids = positives[anchor_ids, rng.integers(positives.shape[1], size=len(anchor_ids))]
            nk = min(cfg.negatives_per_anchor, negatives.shape[1])
            choices = np.stack([rng.choice(negatives.shape[1], size=nk, replace=False) for _ in anchor_ids])
            neg_ids = negatives[anchor_ids[:,None], choices]
            combined = ti[np.concatenate([anchor_ids,pos_ids,neg_ids.ravel()])]
            z, logits = model(*tensors(x,teacher_train,combined))
            b=len(anchor_ids); anchor=z[:b]; positive=z[b:2*b]; negative=z[2*b:].reshape(b,nk,-1)
            ce=F.cross_entropy(logits[:b],torch.as_tensor(y[anchor_ids],device=device))
            local,semi,geometry,active=metric_losses(anchor,positive,negative,
                torch.as_tensor(teacher_train[ti[anchor_ids]],device=device),cfg.temperature,cfg.margin,
                torch.as_tensor(anchor_ids,device=device))
            ramp=min(1.0,max(0.0,(epoch+1-cfg.hard_warmup_epochs)/cfg.hard_ramp_epochs))
            loss=ce+cfg.geometry_weight*geometry
            if cfg.objective=='residual_local': loss=loss+cfg.local_weight*local+cfg.semi_hard_weight*ramp*semi
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite V2 loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            totals+=np.asarray([float(loss.detach()),float(local.detach()),float(semi.detach()),float(geometry.detach()),active])*b
            count+=b
        scheduler.step()
        metrics=evaluate()
        improved=acceptable(metrics,baseline,cfg) and metrics['macro_label_precision_at_5']>best['retrieval']['macro_label_precision_at_5']
        if improved:
            best={'epoch':epoch+1,'retrieval':metrics,'selection_status':'validation_proxy_gain'}; stale=0
            save_arrays(folder/'projection.npz',arrays());write_json(folder/'best.json',best)
        else: stale+=1
        history.append({'epoch':epoch+1,'loss':float(totals[0]/count),'local_loss':float(totals[1]/count),
                        'semi_hard_loss':float(totals[2]/count),'geometry_loss':float(totals[3]/count),
                        'semi_hard_active_fraction':float(totals[4]/count),'balanced_anchor_draws':count,
                        'retrieval':metrics,'accepted':bool(improved)})
        saved={'fingerprint':signature,'device':device,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),'epoch':epoch,'stale':stale,'best':best,'history':history,
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else []}
        torch.save(saved,folder/'last.tmp.pt');os.replace(folder/'last.tmp.pt',checkpoint)
        write_json(folder/'history.json',history)
        print(f"{dataset}/{cfg.objective} epoch {epoch+1}: loss={history[-1]['loss']:.4f}, macro P@5={metrics['macro_label_precision_at_5']:.4f}, overlap={metrics['teacher_top5_overlap']:.3f}, accepted={improved}",flush=True)
        if stop_after_epoch is not None and epoch+1>=stop_after_epoch:
            return {'status':'interrupted_at_epoch_boundary','path':str(checkpoint),'epoch':epoch+1}
    manifest={'schema':SCHEMA,'dataset':dataset,'objective':cfg.objective,'config':asdict(cfg),
              'training_fingerprint':signature,'encoder_recipe':embedding_recipe(tm),'source':sources,
              'train_identity':tm['rows_identity'],'validation_identity':vm['rows_identity'],
              'train_rows':len(train),'training_unique_unambiguous_rows':len(ti),
              'validation_selection_rows':len(vi),'validation_excluded_train_duplicates':len(valid)-len(vi),
              'validation_baseline':baseline,'best':best,'selection_status':best['selection_status'],
              'selection':'validation macro label P@5 with per-class/geometry gates; proxy, not LLM utility',
              'path':(folder/'projection.npz').relative_to(root).as_posix(),'projection_sha256':file_hash(folder/'projection.npz'),
              'parameters':sum(p.numel() for p in model.parameters()),'device':device,'torch':torch.__version__,
              'numpy':np.__version__,'gpu_name':torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
              'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
              'train_seconds_this_session':time.perf_counter()-started,'completed_epochs':len(history),
              'test_labels_or_responses_used':False}
    write_json(final,manifest)
    return manifest
