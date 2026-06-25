"""API route modules for Leoma."""

from leoma.delivery.http.routes.blacklist import router as blacklist_router
from leoma.delivery.http.routes.health import router as health_router
from leoma.delivery.http.routes.miners import router as miners_router
from leoma.delivery.http.routes.overview import router as overview_router
from leoma.delivery.http.routes.rotation import router as rotation_router
from leoma.delivery.http.routes.samples import router as samples_router
from leoma.delivery.http.routes.scores import router as scores_router
from leoma.delivery.http.routes.tasks import router as tasks_router
from leoma.delivery.http.routes.validators import router as validators_router
from leoma.delivery.http.routes.weights import router as weights_router

__all__ = [
    "miners_router",
    "overview_router",
    "rotation_router",
    "samples_router",
    "scores_router",
    "blacklist_router",
    "health_router",
    "tasks_router",
    "validators_router",
    "weights_router",
]
