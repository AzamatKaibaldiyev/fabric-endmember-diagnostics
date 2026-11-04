# Train/validation splits

`split_fabric_vf10_seed{0..4}.npz` hold the training and validation indices
used in the paper (fabric-level split, 10% of fabrics for validation). Training
loads the file for its seed when it is present here and writes it otherwise.

The indices are released because they cannot be regenerated from the code
alone, and each run prints the split's fingerprint so a comparison can confirm
it scores the same fabrics.
