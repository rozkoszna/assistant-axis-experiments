Folder purpose
==============

Activation source = assistant_header_mean.
This is a prompt-side ablation averaging over the assistant header span.

Trait kept only if overall mean positive judge score minus overall mean negative judge score >= 50.
This is a simple trait-level separation filter.

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
