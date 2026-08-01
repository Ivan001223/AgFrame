from app.platform.runtime.worker_adapter import RuntimeWorkerAdapter


def test_runtime_worker_adapter_separates_governance_and_execution_steps():
    adapter = RuntimeWorkerAdapter()

    plan = adapter.build_execution_plan(run_id="hr-1", task_type="agent_orchestration")

    assert plan["governance_phase"] == "load_run_and_authorize"
    assert plan["runtime_phase"] == "execute_runtime_command"


def test_worker_adapter_plan_includes_completion_phase():
    adapter = RuntimeWorkerAdapter()

    plan = adapter.build_execution_plan(run_id="hr-1", task_type="document_ingest")

    assert plan["completion_phase"] == "report_result_to_governance"


def test_worker_adapter_plan_has_ordered_phases():
    adapter = RuntimeWorkerAdapter()

    plan = adapter.build_execution_plan(run_id="hr-1", task_type="agent_orchestration")
    phases = plan["phases"]

    assert len(phases) == 3
    names = [phase["name"] for phase in phases]
    assert names == ["load_run_and_authorize", "execute_runtime_command", "report_result_to_governance"]


def test_worker_adapter_phase_names_match_top_level_fields():
    adapter = RuntimeWorkerAdapter()

    plan = adapter.build_execution_plan(run_id="hr-1", task_type="session_resume_approval")

    assert plan["phases"][0]["name"] == plan["governance_phase"]
    assert plan["phases"][1]["name"] == plan["runtime_phase"]
    assert plan["phases"][2]["name"] == plan["completion_phase"]


def test_worker_adapter_plan_carries_run_metadata():
    adapter = RuntimeWorkerAdapter()

    plan = adapter.build_execution_plan(run_id="hr-42", task_type="agent_orchestration", command_type="resume")

    assert plan["run_id"] == "hr-42"
    assert plan["task_type"] == "agent_orchestration"
    assert plan["command_type"] == "resume"


def test_worker_adapter_phases_have_descriptions():
    adapter = RuntimeWorkerAdapter()

    plan = adapter.build_execution_plan(run_id="hr-1", task_type="agent_orchestration")

    for phase in plan["phases"]:
        assert isinstance(phase["description"], str)
        assert len(phase["description"]) > 0
