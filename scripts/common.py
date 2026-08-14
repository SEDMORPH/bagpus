""" Shared helpers for the pipeline scripts: config loading and reconstruction
of the Observations / PopulationModel / Fit objects from a config module. """

import argparse
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bagpus


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config .py file')
    parser.add_argument('--batch', type=int, default=1,
                        help='Training batch index (step2 only)')
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location('config', args.config)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg, args


def set_stellar_grid(cfg):
    """ Switch bagpipes to the configured stellar library (must happen before
    any simulation). """
    if getattr(cfg, 'stellar_grid', None):
        from bagpus import grids
        grids.change_grid(neb_grid_name=cfg.stellar_grid,
                          stellar_grid_name=cfg.stellar_grid,
                          grid_dir_name=cfg.dir_stellar_grids)


def obs_path(cfg):
    return os.path.join(cfg.output_dir, cfg.runID + '_observations.pt')


def load_observations(cfg):
    path = obs_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found — run step1_prepare.py first.')
    return torch.load(path, weights_only=False)


def build_fit(cfg):
    """ Reconstruct the Fit object (deterministic given config + saved
    observations). """
    obs = load_observations(cfg)
    model = bagpus.PopulationModel(cfg.pop_instructions, obs, cosmology=cfg.cosmo)
    fit = bagpus.Fit(model, run=cfg.runID, working_dir=cfg.output_dir,
                     ngal_sims=cfg.ngal_sims, ncores=cfg.ncores)
    return fit
