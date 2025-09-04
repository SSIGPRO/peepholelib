import nltk
from nltk.corpus import wordnet as wn

def synset_to_wnid(s):
    # WNID = pos + 8 cifre dell'offset del synset (per i nomi: 'n')
    return f"{s.pos()}{s.offset():08d}"

def get_synsets_from_concept(concept, lang='eng'):

    return wn.synsets(concept, lang=lang, pos=wn.NOUN)

def choose_seed_synsets_cli(concept, lang='eng', prefilter_prefix=None, default_first=True):
    """
    Show candidate synsets and let the user pick by index.
    - prefilter_prefix e.g. 'dog.n.' to keep only specific senses, or None for all noun senses
    - default_first: hitting ENTER selects index 0
    Returns: list[Synset]
    """
    cands = get_synsets_from_concept(concept, lang=lang)
    if prefilter_prefix:
        cands = [s for s in cands if s.name().startswith(prefilter_prefix)]
    if not cands:
        raise RuntimeError(f"No noun synsets found for '{concept}' (lang={lang}).")

    print(f"Candidate synsets for '{concept}' (lang={lang}):")
    for i, s in enumerate(cands):
        print(pretty_synset_row(i, s))

    prompt = "Choose one or more by index (comma-separated)."
    if default_first: prompt += " PRESS ENTER for [0]."
    raw = input(prompt + " > ").strip()

    idxs = []
    if not raw:
        if default_first:
            idxs = [0]
        else:
            raise RuntimeError("No selection made.")
    else:
        for tok in raw.split(","):
            tok = tok.strip()
            if tok:
                j = int(tok)
                if j < 0 or j >= len(cands):
                    raise ValueError(f"Index {j} out of range [0..{len(cands)-1}]")
                idxs.append(j)

    # Deduplicate while preserving order
    seen = set()
    sel = []
    for j in idxs:
        if j not in seen:
            sel.append(cands[j])
            seen.add(j)
    return sel

def pretty_synset_row(i, s):
    lemmas = ", ".join(s.lemma_names())
    return f"[{i}] {s.name():<18}  {s.definition()}  (lemmas: {lemmas})"

def hyponym_closure_synsets(seed_synsets):
    seen = set(seed_synsets)
    stack = list(seed_synsets)
    while stack:
        s = stack.pop()
        for h in s.hyponyms():
            if h not in seen:
                seen.add(h)
                stack.append(h)
    return seen

def wnid_to_synset(wnid: str):
    """
    Converte un WNID tipo 'n02084071' in un oggetto Synset di WordNet.
    """
    pos = wnid[0]   # 'n', 'v', 'a', 'r'
    offset = int(wnid[1:])
    return wn.synset_from_pos_and_offset(pos, offset)