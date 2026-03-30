Folder purpose
==============

Activation source = assistant_header_mean.
This is a prompt-side ablation averaging over the assistant header span.

For each prompt pair p in {0..4}, count how many of its 40 matched questions satisfy:
positive_score - negative_score >= 50.
Trait kept only if at least 3 of the 5 prompt pairs each have >= 10 such matched question wins.
This is a stricter paper-inspired interpretation.

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
