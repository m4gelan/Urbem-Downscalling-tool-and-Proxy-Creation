"""Central colormap palette for proxy-layer overlays across sectors.

Before this module every ``*_context.py`` hard-coded its own colormap
(``viridis`` for Industry / Fugitive, ``YlOrRd`` for Public Power population,
``plasma`` / ``inferno`` / ``cividis`` / ``Greens`` for Waste, etc.) which made
it impossible to tell "which colour means population" across sector maps.

:data:`PROXY_PALETTE` assigns a single matplotlib colormap name to every proxy
kind used by the sector builders:

* ``population`` / ``p_pop`` -> ``YlOrRd``   (pop heat, same family across sectors)
* ``osm``                    -> ``PuBu``     (roads / pipelines / rail / POIs)
* ``clc``                    -> ``YlGn``     (CORINE mask, sector score)
* ``p_g``                    -> ``plasma``   (final per-group blend, same hue as weights)
* ``residual``               -> ``cividis``
* ``imperv``                 -> ``bone``
* ``treatment``              -> ``OrRd``
* ``agglomeration``          -> ``Purples``
* ``industrial_mask``        -> ``Oranges``
* ``hotmap_res`` / ``heat_res``      -> ``OrRd``
* ``hotmap_nonres`` / ``heat_nonres`` -> ``YlGnBu``
* ``ship``                   -> ``Blues``
* ``solid_waste``            -> ``YlOrBr``
* ``wastewater``             -> ``BuPu``

:func:`palette_for_title` matches substrings (case-insensitive) in a layer title
to the corresponding palette key, so callers do not have to remember the exact
mapping. The fallback is ``viridis`` which preserves prior behaviour.
"""
from __future__ import annotations

PROXY_PALETTE: dict[str, str] = {
    "population": "YlOrRd",
    "p_pop": "YlOrRd",
    "osm": "PuBu",
    "clc": "YlGn",
    "corine": "YlGn",
    "p_g": "plasma",
    "pg": "plasma",
    "residual": "cividis",
    "imperv": "bone",
    "treatment": "OrRd",
    "agglomeration": "Purples",
    "industrial_mask": "Oranges",
    "hotmap_res": "OrRd",
    "heat_res": "OrRd",
    "hotmap_nonres": "YlGnBu",
    "heat_nonres": "YlGnBu",
    "solid_waste": "YlOrBr",
    "wastewater": "BuPu",
    "ship": "Blues",
    "pipeline": "PuBu",
    "railway": "PuBu",
    "weights": "plasma",
}


# Title substrings are tested in order so more specific matches win. The first
# match decides the colormap.
_TITLE_RULES: tuple[tuple[str, str], ...] = (
    # Residential / non-residential diff is handled explicitly elsewhere (RdBu).
    ("heat_nonres", PROXY_PALETTE["heat_nonres"]),
    ("heat_res", PROXY_PALETTE["heat_res"]),
    ("hotmap nonres", PROXY_PALETTE["hotmap_nonres"]),
    ("hotmap nonresidential", PROXY_PALETTE["hotmap_nonres"]),
    ("hotmap residential", PROXY_PALETTE["hotmap_res"]),
    ("hotmap_res", PROXY_PALETTE["hotmap_res"]),

    # Group-level / final blend proxies
    (" p_g ", PROXY_PALETTE["p_g"]),
    ("(p_g", PROXY_PALETTE["p_g"]),
    ("p_g ", PROXY_PALETTE["p_g"]),
    ("p_pop", PROXY_PALETTE["p_pop"]),
    (" · POP", PROXY_PALETTE["p_pop"]),
    ("_layer", PROXY_PALETTE["p_g"]),

    # Fugitive auxiliary / mixture tokens
    ("VIIRS", "inferno"),
    ("GEM COAL", "inferno"),
    ("GEM_OG", "plasma"),
    ("GEM_G2", "cividis"),
    ("treatment plants", PROXY_PALETTE["treatment"]),
    ("treatment", PROXY_PALETTE["treatment"]),
    ("agglomeration", PROXY_PALETTE["agglomeration"]),
    ("industrial clc", PROXY_PALETTE["industrial_mask"]),
    ("industrial_clc_mask", PROXY_PALETTE["industrial_mask"]),
    ("industrial mask", PROXY_PALETTE["industrial_mask"]),
    ("solid waste", PROXY_PALETTE["solid_waste"]),
    ("solid_waste", PROXY_PALETTE["solid_waste"]),
    ("wastewater", PROXY_PALETTE["wastewater"]),
    ("ww stack", PROXY_PALETTE["wastewater"]),
    ("ww ", PROXY_PALETTE["wastewater"]),
    ("residual", PROXY_PALETTE["residual"]),
    ("imperv", PROXY_PALETTE["imperv"]),

    # Shipping
    ("shipping", PROXY_PALETTE["ship"]),
    ("ais", PROXY_PALETTE["ship"]),
    ("ports", PROXY_PALETTE["ship"]),

    # Generic proxy channels
    ("osm", PROXY_PALETTE["osm"]),
    ("pipeline", PROXY_PALETTE["pipeline"]),
    ("railway", PROXY_PALETTE["railway"]),
    ("road", PROXY_PALETTE["osm"]),

    ("clc sector score", PROXY_PALETTE["clc"]),
    ("clc ", PROXY_PALETTE["clc"]),
    ("corine", PROXY_PALETTE["clc"]),

    # Population last (many other titles mention 'pop' accidentally)
    ("population", PROXY_PALETTE["population"]),
    (" pop ", PROXY_PALETTE["population"]),
    ("p_pop", PROXY_PALETTE["p_pop"]),

    ("weight", PROXY_PALETTE["weights"]),
)


def palette_for_title(title: str, *, default: str = "viridis") -> str:
    """Return the matplotlib colormap name assigned to ``title``.

    Matching is case-insensitive substring on :data:`_TITLE_RULES` and falls back
    to ``default`` (``"viridis"``) when nothing matches.
    """
    if title is None:
        return default
    t = str(title).lower()
    for needle, cmap in _TITLE_RULES:
        if needle in t:
            return cmap
    return default


def palette_for_key(key: str, *, default: str = "viridis") -> str:
    """Return the palette assigned to a machine-readable proxy key.

    ``key`` is the config key (``"population"``, ``"uwwtd_agglomerations"``, ...)
    used by the sector context builders. Falls back to ``default``.
    """
    if not key:
        return default
    k = str(key).lower()
    if k in PROXY_PALETTE:
        return PROXY_PALETTE[k]
    if k.startswith("uwwtd_"):
        sub = k.replace("uwwtd_", "", 1)
        if "agglomeration" in sub:
            return PROXY_PALETTE["agglomeration"]
        if "treatment" in sub:
            return PROXY_PALETTE["treatment"]
    if k.startswith("osm_") or k.startswith("osm "):
        return PROXY_PALETTE["osm"]
    if k.startswith("clc_") or k.startswith("corine"):
        return PROXY_PALETTE["clc"]
    if k.startswith("p_g"):
        return PROXY_PALETTE["p_g"]
    if k.startswith("residual"):
        return PROXY_PALETTE["residual"]
    if "imperv" in k:
        return PROXY_PALETTE["imperv"]
    if "population" in k or k == "pop":
        return PROXY_PALETTE["population"]
    return default
