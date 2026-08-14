"""
step1_prepare.py — Load the catalogue and build the Observations object.

The catalogue-loading block below is DATASET-SPECIFIC: to use bagpus with a
different survey, this is the only code you need to rewrite. Everything the
rest of the pipeline needs is captured in the Observations object.

Writes
------
{output_dir}/{runID}_observations.pt

Usage
-----
    python scripts/step1_prepare.py --config config/run_uds_dblplaw_tauhalf.py
"""

import os

import numpy as np
import torch

from common import load_config

import bagpus

cfg, args = load_config()

# ---------------------------------------------------------------------------
# DATASET-SPECIFIC: load and select the UDS DR11 sample.
#
# Replace this block for a different survey. You need, per galaxy:
#   sc (N,2), sc_err (N,2), redshifts (N,), and optionally a depth proxy
#   magnitude (N,) used to fit the noise floor on the brighter half.
# ---------------------------------------------------------------------------
from astropy.table import Table, join

udsdr11_sc = Table.read(cfg.dir_data + cfg.sc_file)
udsdr11_photo = Table.read(cfg.dir_data + cfg.photo_file)
udsdr11 = join(udsdr11_sc, udsdr11_photo, keys='ID')

ind_good = np.where(
    (udsdr11['CHISQ'] < 3)
    & np.isfinite(udsdr11['CHISQ'])
    & (udsdr11['SC'][:, 0] > -50)
    & (udsdr11['KMAG'] <= cfg.klim)
    & (udsdr11['MASS_SC'][:, 1] > cfg.masslim)
    & (udsdr11['ZUSED'] > cfg.zmin)
    & (udsdr11['ZUSED'] < cfg.zmax)
)
udsdata = udsdr11[ind_good]

print(f'Galaxies after selection cuts: {len(udsdata)}')
print(f'Median redshift: {np.median(udsdata["ZUSED"]):.3f}')

ind_q_SC = np.where((udsdata['CLASS'] == 1) | (udsdata['CLASS'] == 5))[0]
print(f'Quiescent fraction (SC classes): {len(ind_q_SC) / len(udsdata):.3f}')

sc = np.array(udsdata['SC'])
sc_err = np.array(udsdata['SCERR'])
redshifts = np.array(udsdata['ZUSED'])
mag = np.array(udsdata['KMAG'])
# ---------------------------------------------------------------------------
# end of dataset-specific block
# ---------------------------------------------------------------------------

obs = bagpus.Observations(
    sc=sc, sc_err=sc_err, redshifts=redshifts, mag=mag,
    eigenbasis_file=cfg.dir_data + cfg.eigenbasis_file,
    filter_list_file=cfg.filter_list_file,
    filter_dir=cfg.dir_filters,
    n_eigenvectors=cfg.n_eigenvectors,
    pdf_range=cfg.pdf_range, pdf_bins=cfg.pdf_bins,
    filter_mask=cfg.filter_mask,
    zmin=cfg.zmin, zmax=cfg.zmax,
)

os.makedirs(cfg.output_dir, exist_ok=True)
path = os.path.join(cfg.output_dir, cfg.runID + '_observations.pt')
torch.save(obs, path)
print(f'Saved: {path}')
