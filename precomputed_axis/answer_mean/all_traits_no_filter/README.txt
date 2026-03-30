Folder purpose
==============

Activation source = answer_mean.
This is the main paper-faithful source because the paper says trait vectors use all response tokens.

Vector construction: mean(all positive activations) - mean(all negative activations).
No score-based filtering is applied. This is useful if paper filtering was only dataset curation.

Saved file format
=================

Each .pt file contains a dict with:
- vector: Tensor[n_layers, hidden_dim]
- trait: trait name
- activation_position
- filter_name
- passed_filter
- score statistics
- matched pair statistics
