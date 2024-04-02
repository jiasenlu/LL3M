import seqio
from . import tasks

MixtureRegistry = seqio.MixtureRegistry

MixtureRegistry.add(
    "dolma_mixture",
    [
        ("cc_en", 46.23),
        ("stack", 21.37),
        ("c4", 16.19),
        ("reddit", 6.94),
        ("peS2o", 5.49),
        ("reddit", 2.10),
        ("wiki", 1.66),
    ],
    default_rate=1.0)
