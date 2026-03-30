Folder purpose
==============

Activation source = assistant_header_mean.
This is a prompt-side ablation averaging over the assistant header span.

Trait kept only if at least 10 matched (prompt_index, question_index) pairs satisfy:
positive_score - negative_score >= 50.
This is a paper-inspired interpretation of the ambiguous appendix wording.

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
