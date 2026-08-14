"""
step3_train.py — PCA compression and neural posterior training.

Usage
-----
    python scripts/step3_train.py --config config/run_uds_dblplaw_tauhalf.py
"""

from common import load_config, build_fit

cfg, args = load_config()

fit = build_fit(cfg)
fit.train(n_pca_components=cfg.n_pca_components,
          density_estimator=cfg.neural_posterior_model)
