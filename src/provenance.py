#!/usr/bin/env python3


def get_consistent_metadata(data, key):
    values = set(datum[key] for datum in data)
    if len(values) > 1:
        raise ValueError(
            f"Different {key} values {values} cannot be combined in one fit."
        )

    return values.pop() if values else None


def describe_inputs(data, description, specific_keys, consistent_keys, **extra):
    return {
        "_description": description,
        "data_sources": [{key: datum[key] for key in specific_keys} for datum in data],
        **{key: get_consistent_metadata(data, key) for key in consistent_keys},
        **extra,
    }
