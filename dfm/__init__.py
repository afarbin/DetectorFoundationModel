"""Detector Foundation Model - shared backbone, many heads.

The unification of the two model lines in this repository
(review: reviews/btagging_masking_review.md, Part 4):

- ``foundation_model/`` - Mohammad Ali's calorimeter cell GNN (fixed
  detector-geometry edges, masked-cell pretraining, edge classification).
- ``btagging/`` - Umar's track Set-Transformer with a jet-query decoder.

Both become *modalities* of one :class:`~dfm.encoder.SharedEventEncoder`:
detector elements are typed tokens (type embedding + coordinates + features),
mixed locally per modality (fixed calo-neighbor edges / track k-NN EdgeConv)
and globally by a mask-aware ISAB stack, optionally across modalities.
Downstream heads attach to the token embeddings:

- :class:`~dfm.heads.EdgeClassifierHead` - cells -> topocluster edges
  (5-class compatible, binary P(same-cluster) recommended).
- :class:`~dfm.heads.JetQueryDecoder` - jet queries cross-attend to any
  token set; binary b-tag, 3-class b/c/light, or heteroscedastic (mu, sigma)
  regression for jet calibration (review Part 5).
- :class:`~dfm.masking.MaskedTokenPretraining` - modality-agnostic masked
  reconstruction with a mask indicator, usable for cells and tracks alike.

All attention in this package is padding-mask aware (fixes review issue M1),
k-NN guards small events (M4), masking carries group ids explicitly (P1),
the loss is batch-safe (P2), and masked tokens are flagged to the model (P3).
"""

from dfm.tokens import TokenBatch, combine_token_batches
from dfm.encoder import SharedEventEncoder, ModalityConfig
from dfm.heads import EdgeClassifierHead, JetQueryDecoder, ReconstructionHead
from dfm.masking import MaskedTokenPretraining

__all__ = [
    "TokenBatch", "combine_token_batches",
    "SharedEventEncoder", "ModalityConfig",
    "EdgeClassifierHead", "JetQueryDecoder", "ReconstructionHead",
    "MaskedTokenPretraining",
]
