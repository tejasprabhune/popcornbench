"""popcorn_world scenarios.

Importing this package side-effect-registers each scenario with ensemble's
scenario registry, the same way examples/plank/scenarios does.
"""

from . import single_problem  # noqa: F401
from . import judge_review  # noqa: F401
