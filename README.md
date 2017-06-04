# OtterTune

OtterTune is a new tool that’s being developed by students and researchers in the [Carnegie Mellon Database Group](http://db.cs.cmu.edu/projects/autotune/) that can automatically find good settings for a DBMS’s configuration knobs. The goal is to make it easier for anyone to deploy a DBMS without any expertise in database administration. To tune new DBMS deployments, OtterTune reuses training data gathered from previous tuning sessions. Because OtterTune doesn’t need to generate an initial dataset for training its ML models, tuning time is drastically reduced.

We are in the process of creating a [website](http://ottertune.cs.cmu.edu/) where we will soon make OtterTune available as an online-tuning service. This repository contains OtterTune's machine learning libraries that will be used by the [front-end code](https://github.com/oltpbenchmark/website.git).

For more information, see our [paper](http://db.cs.cmu.edu/papers/2017/p1009-van-aken.pdf).

### Contributors

* [Dana Van Aken](http://www.cs.cmu.edu/~dvanaken/)
* [Andy Pavlo](http://www.cs.cmu.edu/~pavlo/)
* [Geoff Gordon](http://www.cs.cmu.edu/~ggordon/)
* Bohan Zhang
