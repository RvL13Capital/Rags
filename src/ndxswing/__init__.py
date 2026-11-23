"""NASDAQ-100 systematic swing-trading research stack.

Layering (signals never talk to a broker):

    data -> features -> signal (SignalProposal) -> risk (RiskApprovedOrder)
         -> backtest / execution -> analytics -> research (walk-forward, ablation)

Nothing in this package fabricates market data or performance numbers.  The
synthetic generator in :mod:`ndxswing.data.synthetic` exists only to verify the
pipeline mechanics and every artefact produced from it is labelled as such.
"""

__version__ = "0.1.0"
