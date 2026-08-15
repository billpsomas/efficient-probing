"""The one list of attentive poolings, and what each one asks the encoder for.

An encoder does not need to know which pooling is attached to it -- only which
tokens to hand back. So --cls_features is remapped to a token selection before it
reaches forward_features():

    pos              -> gap    global average pool over patch tokens
    <pooling>        -> pos    patch tokens, pooling decides how to reduce them
    <pooling>_all    -> both   patch tokens and [cls], for the `_all` variants
    anything else    -> unchanged (cls, gap, raw, both, ...)

This lived twice, verbatim, in models_vit and models_simmim. It sits in util/ so
that both can import it without either depending on the other -- these are the two
files that produce the published numbers, and an edge between them would be worse
than the duplication it removes.
"""

# Attentive poolings over patch tokens only (the default)
ATTENTIVE_POOLINGS = [
    "abmilp", "simpool", "esimpool",
    "clip", "siglip", "aim", "ep", "cbam", "coca",
    "cait", "dinovit", "jepa", "dolg", "cae",
]

# The same poolings, over patch tokens AND the [cls] token
ATTENTIVE_POOLINGS_ALL = [name + "_all" for name in ATTENTIVE_POOLINGS]


def map_cls_features(return_features):
    """Translate a --cls_features value into the token selection to extract."""
    if return_features == "pos":
        return "gap"
    if return_features in ATTENTIVE_POOLINGS:
        return "pos"
    if return_features in ATTENTIVE_POOLINGS_ALL:
        return "both"
    return return_features
