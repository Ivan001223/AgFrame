from app.platform.runtime.worker_adapter import RuntimeWorkerAdapter


def test_runtime_worker_adapter_separates_governance_and_execution_steps():
    adapter = RuntimeWorkerAdapter()

    plan = adapter.build_execution_plan(run_id="hr-1", task_type="agent_orchestration")

    assert plan["governance_phase"] == "load_run_and_authorize"
    assert plan["runtime_phase"] == "execute_runtime_command"
