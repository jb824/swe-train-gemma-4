"""Chunked SFT and SoFT targets (arXiv:2609.32493v2, equations 7 and 24)."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .numerics import require_finite


def project(hidden, weight, softcap):
  logits = F.linear(hidden, weight).float()
  return torch.tanh(logits / softcap) * softcap if softcap is not None else logits


def probability_floor(gold_probs, rstar):
  if not 0 <= rstar <= 1 or gold_probs.numel() == 0:
    raise ValueError('SoFT requires probabilities and rstar in [0, 1]')
  p = gold_probs.double()
  if not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any():
    raise ValueError('Invalid reference probabilities')
  if rstar == 0:
    return 0.0
  if rstar == 1:
    return 1.0
  gap = (1 - p).sum()
  lo, hi = 0.0, 1.0
  for _ in range(48):
    mid = (lo + hi) / 2
    if (mid - p).clamp_min(0).sum() < rstar * gap:
      lo = mid
    else:
      hi = mid
  return (lo + hi) / 2


def reference_support(logits, gold, top_k):
  require_finite([('logits', logits)], 'SoFT reference logits', distributed=False)
  probs = logits.float().softmax(-1)
  values, ids = probs.topk(min(top_k, probs.shape[-1]), dim=-1)
  gold_probs = probs.gather(1, gold[:, None]).squeeze(1)
  present = ids.eq(gold[:, None]).any(-1)
  return {'ids': torch.cat((ids, gold[:, None]), -1).cpu(),
          'base_probs': torch.cat((values, torch.where(present, 0, gold_probs)[:, None]), -1).cpu(),
          'gold_probs': gold_probs.cpu()}


def construct_targets(support, gold, floor):
  if not 0 <= floor <= 1:
    raise ValueError('SoFT probability floor must be finite and in [0, 1]')
  require_finite(support.items(), 'SoFT reference probabilities', distributed=False)
  base = support['base_probs'].float()
  p = support['gold_probs'].float()
  if (base < 0).any() or (base > 1).any() or (p < 0).any() or (p > 1).any():
    raise ValueError('Invalid reference probabilities')
  qgold = torch.maximum(p, torch.full_like(p, floor))
  scale = (1 - qgold) / (1 - p).clamp_min(1e-12)
  q = base * scale[:, None]
  matches = support['ids'].eq(gold[:, None])
  # Gold appears once among positive-mass entries; the extra gold slot is
  # zeroed when it already occurred in top-K.
  matches &= base > 0
  # Extremely small gold probabilities may underflow to zero outside top-K.
  missing = ~matches.any(-1)
  matches[missing, -1] = True
  q = torch.where(matches, qgold[:, None], q)
  tail = (1 - base.sum(-1)).clamp_min(0) * scale
  # Correct only floating-point mass drift, not the selected support.
  total = q.sum(-1) + tail
  q, tail = q / total[:, None], tail / total
  require_finite([('probabilities', q), ('tail', tail)], 'SoFT targets', distributed=False)
  return q, tail


def sparse_soft_loss(logits, ids, target_probs, tail_probs):
  logits = logits.float()
  logz = logits.logsumexp(-1)
  selected = logits.gather(1, ids) - logz[:, None]
  loss = -(target_probs * selected).sum(-1)
  # Compute complement directly: 1 - sum(p_selected) loses precision when
  # tail mass is small. Duplicate gold IDs are harmless for this mask.
  tail_logits = logits.scatter(1, ids, float('-inf'))
  has_tail = torch.isfinite(tail_logits).any(-1)
  tail_logits = torch.where(has_tail[:, None], tail_logits, torch.zeros_like(tail_logits))
  logtail = tail_logits.logsumexp(-1) - logz
  return (loss - torch.where(has_tail, tail_probs * logtail, 0)).sum()


def chunked_objective(hidden, weight, labels, *, chunk_tokens=128, softcap=None,
                      denominator=None, soft_ids=None, soft_probs=None, soft_tail=None):
  keep = labels[:, 1:].reshape(-1).ne(-100)
  states = hidden[:, :-1].reshape(-1, hidden.shape[-1])[keep]
  gold = labels[:, 1:].reshape(-1)[keep]
  if not len(gold):
    raise ValueError('Batch has no supervised tokens')
  if soft_ids is not None and len(soft_ids) != len(gold):
    raise ValueError('Reference targets do not align with supervised positions')
  total = states.sum() * 0
  for start in range(0, len(gold), chunk_tokens):
    end = start + chunk_tokens
    if soft_ids is None:
      def loss(x, w, y):
        return F.cross_entropy(project(x, w, softcap), y, reduction='sum')
      total = total + checkpoint(loss, states[start:end], weight, gold[start:end], use_reentrant=False)
    else:
      def loss(x, w, ids, probs, tail):
        return sparse_soft_loss(project(x, w, softcap), ids, probs, tail)
      total = total + checkpoint(loss, states[start:end], weight, soft_ids[start:end],
                                 soft_probs[start:end], soft_tail[start:end], use_reentrant=False)
  return total / (denominator if denominator is not None else len(gold))
