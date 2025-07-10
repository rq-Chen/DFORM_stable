# Diffeomorphic Vector Field Alignment Network (DFORM)

This repository implements DFORM, a method to align and compare dynamical systems as well as to identify low-dimensional motifs in high-dimensional systems. This method is a further development of the following preprint:

Chen, R., Vedovati, G., Braver, T., & Ching, S. (2024). DFORM: Diffeomorphic vector field alignment for assessing dynamics across learned models (No. arXiv:2402.09735). arXiv. [http://arxiv.org/abs/2402.09735](http://arxiv.org/abs/2402.09735)

A new preprint describing the current version of DFORM is coming soon.


## Outline

The main script is `DFORM.py` which defines the model architecture `DFORM` and the `Training` and `Testing` functions. The hyperparmeters were detailed in the comments of `Training` function. To train the model with multiple initializations and pick the best one, you can use the `get_topo_sim` function instead of `Training`.

`example_systems.py` defines some example dynamical systems. `visualizations.py` defines some functions to visualize vector fields: `PlotField`, `PlotVec`, `PlotTraj` and `PlotDFORM`.