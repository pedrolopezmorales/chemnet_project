import difflib


def build_normalized_lookup(valid_names):
    lookup = {}
    for name in valid_names:
        name_str = str(name).strip()
        if name_str and name_str.lower() not in lookup:
            lookup[name_str.lower()] = name_str
    return lookup


def resolve_case_insensitive_name(query, valid_names, normalized_lookup=None):
    if query is None:
        return query

    query_str = str(query).strip()
    if not query_str:
        return query_str

    lookup = normalized_lookup or build_normalized_lookup(valid_names)
    return lookup.get(query_str.lower(), query_str)


def get_close_matches_custom(query, valid_names, n=3, cutoff=0.6, normalized_lookup=None):
    if query is None:
        return []

    query_str = str(query).strip()
    if not query_str:
        return []

    normalized_map = normalized_lookup or build_normalized_lookup(valid_names)
    matched_keys = difflib.get_close_matches(
        query_str.lower(),
        list(normalized_map.keys()),
        n=n,
        cutoff=cutoff,
    )
    return [normalized_map[key] for key in matched_keys]
