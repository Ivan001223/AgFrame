from app.infrastructure.database.models import Base


def test_harness_tables_are_registered():
    tables = Base.metadata.tables

    assert "harness_run" in tables
    assert "harness_agent_project" in tables
    assert "harness_model_provider" in tables
    assert "harness_approval" in tables
    assert "harness_verification" in tables
    assert "harness_event" in tables
