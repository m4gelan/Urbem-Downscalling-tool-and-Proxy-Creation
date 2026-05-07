"""GNFR I / off-road proxy constants shared across subsector modules."""

IDX_GNFR_I = 12

BAD_RAILWAY = frozenset({"tram", "subway"})
RAILWAY_LIFECYCLE_DISALLOW = frozenset({"abandoned", "disused", "razed", "proposed"})

INDUSTRIAL_OFFROAD_FAMILIES = frozenset(
    {
        "landuse_industrial",
        "landuse_commercial",
        "landuse_construction",
        "landuse_quarry",
        "man_made_works",
        "industrial_depot",
    }
)
