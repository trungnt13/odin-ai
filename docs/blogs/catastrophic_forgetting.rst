Catastrophic forgetting
=======================

Should you retrain a network using unrecognized samples?

Identify the issue
------------------
Due to the dense representation used by CNNs, the network will tend to forget previously learned information.

Solutions
---------
One must either
1) Retrain on old samples, i.i.d.
2) Adopt a different, more tabular representation (such as sparse codes) to drastically reduce interference
3) In this particular instance, since this isn't quite online learning (it integrates new information in batches, not one sample at a time), it may be possible to use a recent technique developed by DeepMind for reducing forgetting between two reinforcement learning tasks: https://arxiv.org/pdf/1612.00796v2.pdf, or https://arxiv.org/pdf/1606.04671.pdf. This may work on this supervised learning task as well.
