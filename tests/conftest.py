from __future__ import annotations

import pytest

from living_tabletop.engine import GameEngine
from living_tabletop.llm import LLMSettings, OpenAICompatibleLLM
from living_tabletop.scenario import create_initial_state, load_scenario


@pytest.fixture(scope="session")
def scenario():
    return load_scenario()


@pytest.fixture
def offline_llm():
    return OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None))


@pytest.fixture
def engine(scenario, offline_llm):
    return GameEngine(scenario, offline_llm)


@pytest.fixture
def state(scenario):
    return create_initial_state(scenario, player_name="测试员", seed=19, session_id="test-session")

