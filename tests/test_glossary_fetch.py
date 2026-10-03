from subai.glossary_fetch import diff_characters, norm

CURATED = [
    {"name": "Melek Yücel", "aliases": ["Melo"], "actor": "Elçin Afacan", "episodes": 52},
    {"name": "Kiraz", "aliases": ["Kiraz"], "actor": "Maya Başol", "episodes": 13},
]


def test_norm_is_accent_and_turkish_i_insensitive():
    assert norm("Maya Başol") == norm("MAYA BASOL")
    assert norm("Çağrı Çıtanak") == norm("Cagri Citanak")
    assert norm("İlkyaz") == norm("Ilkyaz")


def test_diff_matches_ascii_spelling_and_flags_new_actor():
    tmdb = [{"actor": "Maya Basol", "character": "Kiraz", "episodes": 13},
            {"actor": "Derya Aldemir", "character": "", "episodes": 1}]
    d = diff_characters(CURATED, tmdb, [])
    assert [m["actor"] for m in d["missing_in_glossary"]] == ["Derya Aldemir"]
    assert d["name_differs"] == [] and d["episode_count_differs"] == []


def test_diff_ignores_middle_names():
    cur = [{"name": "Alptekin Bolat", "aliases": [], "actor": "Ahmet Mark Somers", "episodes": 20}]
    d = diff_characters(cur, [{"actor": "Ahmet Somers", "character": "Alptekin Bolat", "episodes": 20}], [])
    assert d["missing_in_glossary"] == []


def test_diff_flags_name_and_count_mismatch_but_accepts_alias():
    tmdb = [{"actor": "Elçin Afacan", "character": "Melek Yücel \"Melo\"", "episodes": 50},
            {"actor": "Maya Başol", "character": "Cherry", "episodes": 13}]
    d = diff_characters(CURATED, tmdb, [])
    assert d["episode_count_differs"] == [{"actor": "Elçin Afacan", "curated": 52, "tmdb": 50}]
    assert [n["curated"] for n in d["name_differs"]] == ["Kiraz"]
