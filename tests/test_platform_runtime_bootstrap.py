from app.platform.contracts.runtime_protocol import RuntimeCommandV1
from app.platform.runtime.bootstrap import build_runtime_command_for_run, build_runtime_execution_plan


def test_build_runtime_execution_plan_delegates_to_worker_adapter():
    plan = build_runtime_execution_plan(run_id="hr-1", task_type="agent_orchestration")

    assert plan["run_id"] == "hr-1"
    assert plan["runtime_phase"] == "execute_runtime_command"


def test_build_runtime_command_for_run_shapes_start_command():
    command = build_runtime_command_for_run(run_id="hr-1", task_type="agent_orchestration")

    assert isinstance(command, RuntimeCommandV1)
    assert command.run_id == "hr-1"
    assert command.command_type == "start"
