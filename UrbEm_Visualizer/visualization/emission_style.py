from __future__ import annotations

EPS = 0.01

BACKGROUND_THRESHOLD = {
    "NOx": 1.0,
    "NMVOC": 0.5,
    "CO": 1.0,
    "SO2": 0.1,
    "NH3": 0.2,
    "PM10": 0.3,
    "PM25": 0.2,
}

COLORMAP = {
    "NOx": "YlOrRd",
    "NMVOC": "YlGn",
    "CO": "Oranges",
    "SO2": "PuRd",
    "NH3": "BuGn",
    "PM10": "Greys",
    "PM25": "RdPu",
}


def pollutant_key(pollutant: str) -> str:
    p = str(pollutant).strip()
    if p in ("PM2.5", "PM2_5", "pm2_5"):
        return "PM25"
    return p


def threshold_for(pollutant: str) -> float:
    return float(BACKGROUND_THRESHOLD[pollutant_key(pollutant)])


def default_threshold(pollutant: str) -> float:
    return threshold_for(pollutant)


def colormap_for(pollutant: str) -> str:
    return COLORMAP[pollutant_key(pollutant)]
