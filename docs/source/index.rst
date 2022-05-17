:github_url: https://github.com/DeepDriveMD/DeepDriveMD-pipeline

DeepDriveMD Documentation
=========================

DeepDriveMD: Deep-Learning Driven Adaptive Molecular Simulations.


:Release: |release|
:Date: |today|

Summary
=======

**deepdrivemd** is a Python package for coupling molecular dynamics ensemble simulations to
sampling agents guided by machine learning.

DeepDriveMD can support two modes of execution, one is synchronous and runs MD simulations,
aggregation, training, and inference stages in a pipeline where each stage blocks the others
and the stages communicate via the filesystem (**DeepDriveMD-F**).

The second, and more optimal, mode of execution is asynchronous where each of the previously
mention stages run continously as independent components communicating via adios2_ to stream
data between concurrently running workflow components, enabling efficient feedback between
simulations and learning algorithms (**DeepDriveMD-S**).

Both modes of execution are implemented using `RADICAL-Ensemble Toolkit`_ to enable support
for large scale runs on high-performance computing platforms.

Additional information can be found on our website_. 

.. _adios2: https://adios2.readthedocs.io/en/latest/

.. _RADICAL-Ensemble Toolkit: https://radicalentk.readthedocs.io/en/stable/

.. _website: https://deepdrivemd.github.io/

Getting involved
================

Please report **bugs** or **enhancement requests** through the `Issue
Tracker`_.

.. _Issue Tracker: https://github.com/DeepDriveMD/DeepDriveMD-pipeline/issues

Installing DeepDriveMD
======================

To install the latest release, clone the code from the `main branch`_ and use pip to install the package.

.. _main branch: https://github.com/DeepDriveMD/DeepDriveMD-pipeline

pip
---

Installation with `pip` and a *minimal set of dependencies*:

.. code-block:: bash 

   git clone https://github.com/DeepDriveMD/DeepDriveMD-pipeline
   cd deepdrivemd
   pip install -e .

.. toctree::
   :maxdepth: 3

   pages/api

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
