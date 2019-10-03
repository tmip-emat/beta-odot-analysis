# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ODOT Beta-Test - Meta-Model Development Notebook

# %% [markdown]
# <div style="color:red; border:1px solid; padding:5px; max-width:800px; font-size:80%;">
# The purpose of this test and the following analysis was to evaluate ODOT’s new Activity Based Model (ABM); 
# specifically the ability of the ABM to provide information about emerging technologies.  To help to achieve 
# that purpose a realistic, but fictitious, set of regional ABM inputs was developed.  At the end of this 
# beta test, several flaws in the performance measure creation and methodology were noted as potential 
# improvements for future analysis, but were not corrected in this dataset and resulting analysis.  The 
# information in this data and analysis serves as an example for how to use TMIP-EMAT using realistic data.  
# This dataset and analysis should not be used to draw any specific conclusions about transportation policy’s 
# impact on system performance and outcomes.
# </div>
#

# %% [markdown]
# In this notebook, we walk through the development of a MetaModel for the 
# ODOT SOABM using TMIP-EMAT.  The documentation here presumes that the
# model scoping, initial experimental designs, and initial core model runs
# have all been completed previously.

# %%
import os
import numpy
import pandas

# %% [markdown]
# If you do not have the correct version of EMAT (at least 0.2.5) installed to run this workbook, 
# visit the [TMIP-EMAT website](https://tmip-emat.github.io/source/emat.conda.html#managing-environments) 
# for installation instructions.

# %%
import emat
emat.versions()

# %% [markdown]
# Enable some logging. This is optional but convenient, especially for
# keeping track of run-time.

# %%
from emat.util.loggers import log_to_stderr, TimingLog
log = log_to_stderr(level=20)
log.info('Logging Starts')

# %% [markdown]
# ## Load Existing Scope and Data

# %% [markdown]
# Connect to DB file with populated LHS experiment design, and read scope 
# and experimental data.

# %%
db = emat.SQLiteDB("soabm_v2.db", initialize=False)

# %%
db.read_scope_names()

# %%
scope = db.read_scope('SOABM')
scope.info()

# %%
db.read_design_names('SOABM')

# %% [markdown]
# Load the data from the core model experiments into a `pandas.DataFrame`.

# %%
core_experiments = db.read_experiment_all(scope_name='SOABM', design_name='odot_lhs', ensure_dtypes=True)
core_experiments.info()

# %% [markdown]
# Load a modified scope from the YAML file.  This modified version adds some new 
# output features that were not available for the initial model runs.

# %%
scope1 = emat.Scope('SOABM_scope.yaml') 
scope1.info()

# %% [markdown]
# ## Visualization and Analysis of Core Model Experiments
#
# ### Scatterplot Matrix: Performance Measures w.r.t. Input Parameters 

# %%
from emat.analysis import display_experiments
display_experiments(scope1, core_experiments)

# %% [markdown]
# ### Feature Scoring

# %%
from emat.analysis import feature_scores
feature_scores(scope, core_experiments, return_type='styled')

# %% [markdown]
# ## Derive Meta Models

# %%
from emat.model.meta_model import create_metamodel

# %%
db_a = emat.database.SQLiteDB('soabm_live_analysis_v2.db', initialize=True)
# Note running this will overwrite any existing version of `soabm_live_analysis_v2.db` 

# %%
scope1.store_scope(db_a)

# %%
db_a.write_experiment_all(
    scope1.name, 
    'odot_lhs', 
    emat.SOURCE_IS_CORE_MODEL, 
    core_experiments,
)

# %%
with TimingLog():
    mm = create_metamodel(scope1, core_experiments, db=db_a)
mm

# %%
mm.function.regression.lr.r2

# %%
mm.function.regression.lr.coefficients_summary()

# %%
with TimingLog():
    cv_scores = mm.function.cross_val_scores()

# %%
cv_scores

# %%
big = scope1.design_experiments(n_samples=5000, sampler='mc', db=db_a)

# %%
with TimingLog():
    big_runs = mm.run_experiments(big, db=db_a)

# %% [markdown] {"toc-hr-collapsed": true}
# ### Contrast Core and Meta Model Results

# %%
from emat.analysis import contrast_experiments
contrast_experiments(scope1, big_runs, core_experiments, mass=100)

# %%
