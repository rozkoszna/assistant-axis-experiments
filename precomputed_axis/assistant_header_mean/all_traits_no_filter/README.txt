Folder purpose
==============

Activation source = assistant_header_mean.
This is a prompt-side ablation averaging over the assistant header span.

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
