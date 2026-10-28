# Field reference

A projection JSON contains:

- `schema_version`: serialization format used by the fixture generator.
- `record_id`: unique shard inventory label.
- `layer_index` and `shard_index`: non-semantic partition metadata.
- `matrix_shape`: 2D dimensions of each Q/K/V projection.
- `projection`: three arrays of signed float coefficients.
- `checksum_hint`: short fingerprint of the matrix payload, not a security attestation.
- `tags`: descriptive search labels used for corpus grouping.

A sample JSONL record contains:

- `sample_id`: stable identifier.
- `text`: text consumed by the token feature generator.
- `collection`: grouping marker for experiments.
- `source_tag`: local notebook classification; never a network location.

A result JSON contains a run configuration and sample-level diagnostics. No private data or third-party model artifacts are included in this repository.
