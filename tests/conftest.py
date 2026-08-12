"""Shared test setup.

Selecting the Agg backend here (before any test imports pyplot) keeps the
plotting tests headless, so they run the same way in CI as they do locally.
"""

import matplotlib

matplotlib.use("Agg")
