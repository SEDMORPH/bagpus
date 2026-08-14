# Installation

bagpus requires Python ≥ 3.10.

```bash
git clone https://github.com/vwild/bagpus.git
cd bagpus
pip install -e .
```

This installs the Python dependencies (numpy, scipy, astropy, torch, sbi,
scikit-learn, bagpipes, fast_histogram, joblib). We recommend a dedicated
conda environment:

```bash
conda create -n bagpus python=3.11
conda activate bagpus
pip install -e .
```

## bagpipes stellar grids

bagpus drives [bagpipes](https://bagpipes.readthedocs.io) as its forward
model, and inherits its stellar population grids. The default bagpipes
installation ships BC03 grids; the bagpus release paper used the CB19
(Charlot & Bruzual 2019) models.

To switch libraries, place the grid FITS files in a directory and call, before
any simulation:

```python
import bagpus
bagpus.grids.change_grid(neb_grid_name='cb19', stellar_grid_name='cb19',
                         grid_dir_name='/path/to/Bagpipes_grids/')
```

`bagpus.grids.list_grids('/path/to/Bagpipes_grids/')` shows which grids are
available in a directory. Contact the authors for the CB19 grid files if you
need them.

## Checking the installation

```python
import bagpus
print(bagpus.__version__)
print(bagpus.models.SFH_MODELS)   # {'dblplaw': ...}
```

Then run the [quickstart notebook](examples/1_quickstart) — it exercises the
whole pipeline on mock data in a few minutes.
