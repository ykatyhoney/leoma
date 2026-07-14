"""Validator service (king-of-the-hill duel + weight-setter).

Deliberately re-exports nothing. Importing the ``main`` *function* here shadowed
the ``main`` *module*, so ``leoma.app.validator.main`` resolved to a function —
which made the module unreachable by attribute access (and unpatchable in tests).
Import the entry point from its module instead:

    from leoma.app.validator.main import main
"""
