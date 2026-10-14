"""
Shared text normalization for synonym/entity matching (Stages 4-5).

MatKG entity strings and PubChem synonyms disagree on case, hyphens,
inserted spaces, token fusion ("Trifluoromethylsulfonyl" vs
"trifluoromethyl)sulfonyl") and British spellings ("sulphonyl"). Both
sides are therefore reduced to a canonical squashed token string with
token-boundary offsets. A synonym matches an entity iff the synonym's
squash appears in the entity's squash starting at a token start and
ending at a token end (the word-boundary rule); an exact match is the
full-span case.
"""

import re
import unicodedata

# ASCII hyphen/underscore plus the U+2010..U+2015 dash block
_SEPARATORS = re.compile(r"[\s‐-―_-]+")
_TOKEN = re.compile(r"[a-z0-9]+")


def canon(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return s.replace("sulph", "sulf")


def norm_key(s):
    return _SEPARATORS.sub("", canon(s))


def tokens_of(s):
    return tuple(_TOKEN.findall(canon(s)))


def cased_squash(s):
    """Case-preserving squash for short-string matching: 'SN' must not
    meet the element 'Sn', nor 'FEC' the iron carbide 'FeC'."""
    s = unicodedata.normalize("NFKC", s)
    return "".join(re.findall(r"[A-Za-z0-9]+", s))


def squash_bounds(tokens):
    """-> (squashed string, token start offsets, token end offsets)."""
    starts, ends = [], []
    pos = 0
    for t in tokens:
        starts.append(pos)
        pos += len(t)
        ends.append(pos)
    return "".join(tokens), starts, ends
