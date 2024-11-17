from collections.abc import Generator

import gurobipy as gp
import pytest

from xplor.gurobi import XplorGurobi


@pytest.fixture
def xmodel() -> Generator[XplorGurobi]:
    env = gp.Env()
    model = gp.Model(env=env)
    xplor = XplorGurobi(model)
    yield xplor
    model.close()
    env.close()
