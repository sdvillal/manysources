Introduction
------------

We are revisiting through theory, discussion and (not so) anecdotical evidence several issues
that arise when appliying statistical learning [#f1]_ over chemical datasets with scaffold
overrepresentation, a problem commonly called *analog bias* in the cheminformatics literature.

.. image:: https://raw.githubusercontent.com/sdvillal/manysources/master/doc/posters/Poster_GordonSeminar_Montanari_20150617.png

Because of the way chemical collections are usually constructed (by exploring substitutions
around scaffolds of interest), the way molecules are often represented when fed to statistical
models (1D and 2D descriptors dominate the academic literature) and the way these models work
(learning repeated discriminative or correlative patterns), two problems pervade models built
using the average chemical dataset:


* **Overoptimistic evaluation**
*inflated performance estimates irrelevant for generalisation*

* **Scaffold overfitting**
*models that give too much importance to overrepresented features and miss what often is more interesting,
activity cliffs*


Model <-> Example <-> Predicition interactions (with a chemical twist)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tools in this repository are useful on their own too. Some highlights:

* We use **unfolded, unhashed fingerprints**.

  * Pros: good for model interpretability and avoiding hash clashes effects. They usually provide `elevated`_ `performance`_ [#f2]_
  * Cons: no more regularisation by hashing, one needs a model that scales well with very high dimensionality

* We provide a general framework to **understand individual molecule predictions** in the context of a concrete dataset.

  * Linking to feature importance and assessing too the influence of other molecules and selecting influential molecules.
  * Providing quantitative and qualitative insight in the warts of evaluation and the whys of predcitions
  * Can be extended to provide different hypothesis for individual molecules prediction on model deployment time.


Usage
-----

We include several of the datasets we use in our study on this repository (see *data*).

* Use *manysources/datasets.py* for feature extraction.

* Use *manysources/experiments.py* to generate new results. These build and evaluate models
for many different data partitions (note that we run these for a few hours in something like 30 parallel jobs).

* Use *manysources/hub-py* to easily link everything, from molecules to features to model to prediction and back.

* There are some example analysis in *manysources/analyses*.


Installation
------------

We recommend using the `anaconda`_ python scientific distribution
to install *manysources* and its dependencies. Dependencies are in *setup.py*.
So assuming we are using a conda environment, these commands install the required
software:

.. code-block:: sh

  conda install numpy scipy pandas h5py matplotlib seaborn joblib scikit-learn cytoolz networkx numba
  conda install -c rdkit rdkit
  pip install whatami tsne argh

To install manysources itself there are a few options:

* `download a zip file`_ or `clone the manysources repository`_ and then tweak $PYTHONPATH or use *pip install -e*

...or...

* *pip install git+git://github.com/sdvillal/manysources.git*

Proper releases are coming (at least there will be one when a publication happens).


Work in progress, but come back soon
------------------------------------

This is the code we (`Floriane`_, `Santi`_) are using in our experiments.
A documented, stable and more featureful release is happening in 2015 Q4.
In the meantime, feel free to peek around the code, it is not too bad!


.. [#f1] But our research is also relevant to other QSAR methods, from statistical pharmacophore mining to
         docking evaluation.
.. [#f2] One can also use the analog bias to do well in competitions ;-)

.. _anaconda: https://store.continuum.io/cshop/anaconda/
.. _download a zip file: https://github.com/sdvillal/manysources/archive/master.zip
.. _clone the manysources repository: https://github.com/sdvillal/manysources
.. _elevated: http://www.tdtproject.org/blog/strong-showing-in-tdts-2014-challenge
.. _performance: https://github.com/sdvillal/ccl-malaria
.. _Floriane: floriane.montanari@gmail.com
.. _Santi: sdvillal@gmail.com
