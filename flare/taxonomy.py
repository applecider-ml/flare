"""Broad-class taxonomy: 10 spectroscopic subclasses mapped to 5 broad classes."""

BROAD_CLASSES = ["SNI", "SNII", "CV", "AGN", "TDE"]
ORIG2BROAD = {
    "SN Ia": "SNI", "SN Ib": "SNI", "SN Ic": "SNI",
    "SN II": "SNII", "SN IIP": "SNII", "SN IIn": "SNII", "SN IIb": "SNII",
    "Cataclysmic": "CV", "AGN": "AGN", "Tidal Disruption Event": "TDE",
}
SUBCLASS_ID2NAME = [
    "SN Ia", "SN Ib", "SN Ic",
    "SN II", "SN IIP", "SN IIn", "SN IIb",
    "Cataclysmic", "AGN", "Tidal Disruption Event",
]
NUM_CLASSES = len(BROAD_CLASSES)
BROAD2ID = {c: i for i, c in enumerate(BROAD_CLASSES)}
ID2BROAD_ID = {i: BROAD2ID[ORIG2BROAD[n]] for i, n in enumerate(SUBCLASS_ID2NAME)}
TDE_IDX = BROAD2ID["TDE"]
AGN_IDX = BROAD2ID["AGN"]

# The shipped BTS6 taxonomy
BTS6_CLASSES = ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"]
BTS6_TOP_CLASSES = ["SN_Ia", "SN_CC+", "AGN", "TDE", "CV"]
NUM_BTS6_CLASSES = len(BTS6_CLASSES)
