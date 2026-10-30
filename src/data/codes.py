"""Code mappings for categorical fields (data-auditor, 2026-10-30).

1) HQ_city -> Italian region / macro-area.
   HYPOTHESIS (evidence in reports/data/01_profile.md): HQ_city holds ISTAT province codes
   1..110 (the 2009-2016 province set). Evidence: exactly 110 distinct codes 1..110; the most
   frequent codes are 15 (Milano), 58 (Roma), 17 (Brescia), 1 (Torino), 63 (Napoli), 16 (Bergamo),
   37 (Bologna), matching where Italian firms > EUR 1.5MM assets concentrate; codes 105/106
   (Ogliastra, Medio Campidano, the smallest provinces) are the rarest. The code table below is
   ISTAT's province coding [UNVERIFIED: literature-scout to verify against an ISTAT source].
   Macro-areas follow ISTAT's NUTS-1 grouping (Nord-Ovest, Nord-Est, Centro, Sud, Isole).
   The city/province code is a LABEL, never a number.

2) ateco_sector (2-digit ATECO 2007 division == NACE Rev. 2 division) -> section letter A..U.
   Division ranges are the NACE Rev. 2 / ATECO 2007 structure [UNVERIFIED: literature-scout].
   Note: docs/2603_-CODES_ATECO_(003).pdf is a 2020 COVID list of non-suspended activities,
   not a full classification; it confirms the code format (2-digit divisions) only.
"""
from __future__ import annotations

PROVINCE_REGION = {}
_REG = {
    "Piemonte": [1, 2, 3, 4, 5, 6, 96, 103],
    "Valle d'Aosta": [7],
    "Liguria": [8, 9, 10, 11],
    "Lombardia": [12, 13, 14, 15, 16, 17, 18, 19, 20, 97, 98, 108],
    "Trentino-Alto Adige": [21, 22],
    "Veneto": [23, 24, 25, 26, 27, 28, 29],
    "Friuli-Venezia Giulia": [30, 31, 32, 93],
    "Emilia-Romagna": [33, 34, 35, 36, 37, 38, 39, 40, 99],
    "Marche": [41, 42, 43, 44, 109],
    "Toscana": [45, 46, 47, 48, 49, 50, 51, 52, 53, 100],
    "Umbria": [54, 55],
    "Lazio": [56, 57, 58, 59, 60],
    "Campania": [61, 62, 63, 64, 65],
    "Abruzzo": [66, 67, 68, 69],
    "Molise": [70, 94],
    "Puglia": [71, 72, 73, 74, 75, 110],
    "Basilicata": [76, 77],
    "Calabria": [78, 79, 80, 101, 102],
    "Sicilia": [81, 82, 83, 84, 85, 86, 87, 88, 89],
    "Sardegna": [90, 91, 92, 95, 104, 105, 106, 107],
}
for _r, _codes in _REG.items():
    for _c in _codes:
        PROVINCE_REGION[str(_c)] = _r

REGION_MACRO = {
    "Piemonte": "North-West", "Valle d'Aosta": "North-West", "Liguria": "North-West",
    "Lombardia": "North-West",
    "Trentino-Alto Adige": "North-East", "Veneto": "North-East",
    "Friuli-Venezia Giulia": "North-East", "Emilia-Romagna": "North-East",
    "Toscana": "Centre", "Umbria": "Centre", "Marche": "Centre", "Lazio": "Centre",
    "Abruzzo": "South", "Molise": "South", "Campania": "South", "Puglia": "South",
    "Basilicata": "South", "Calabria": "South",
    "Sicilia": "Islands", "Sardegna": "Islands",
}

# NACE Rev.2 / ATECO 2007 sections by 2-digit division
_SECTIONS = [
    ("A", 1, 3, "Agriculture, forestry, fishing"),
    ("B", 5, 9, "Mining and quarrying"),
    ("C", 10, 33, "Manufacturing"),
    ("D", 35, 35, "Electricity, gas, steam"),
    ("E", 36, 39, "Water supply, sewerage, waste"),
    ("F", 41, 43, "Construction"),
    ("G", 45, 47, "Wholesale and retail trade; motor vehicle repair"),
    ("H", 49, 53, "Transportation and storage"),
    ("I", 55, 56, "Accommodation and food service"),
    ("J", 58, 63, "Information and communication"),
    ("K", 64, 66, "Financial and insurance"),
    ("L", 68, 68, "Real estate"),
    ("M", 69, 75, "Professional, scientific, technical"),
    ("N", 77, 82, "Administrative and support services"),
    ("O", 84, 84, "Public administration"),
    ("P", 85, 85, "Education"),
    ("Q", 86, 88, "Health and social work"),
    ("R", 90, 93, "Arts, entertainment, recreation"),
    ("S", 94, 96, "Other services"),
    ("T", 97, 98, "Households as employers"),
    ("U", 99, 99, "Extraterritorial organisations"),
]
SECTION_NAME = {s: n for s, _, _, n in _SECTIONS}


def ateco_section(code) -> str | None:
    try:
        v = int(float(code))
    except (TypeError, ValueError):
        return None
    for s, lo, hi, _ in _SECTIONS:
        if lo <= v <= hi:
            return s
    return None


def province_region(code) -> str | None:
    if code is None:
        return None
    try:
        key = str(int(float(code)))
    except (TypeError, ValueError):
        return None
    return PROVINCE_REGION.get(key)
