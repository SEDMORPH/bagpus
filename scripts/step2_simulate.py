"""
step2_simulate.py — Generate training and test simulations.

Skips batches that already exist. To generate additional independent training
batches (e.g. on different machines), run again with --batch 2, --batch 3, ...

Usage
-----
    python scripts/step2_simulate.py --config config/run_uds_dblplaw_tauhalf.py [--batch N]
"""

from common import load_config, set_stellar_grid, build_fit

cfg, args = load_config()
set_stellar_grid(cfg)

fit = build_fit(cfg)
fit.simulate(nsims=cfg.nsims_train, batch=args.batch)
fit.simulate_test(nsims=cfg.nsims_test)
