from app.platform.contracts.runtime_protocol import RuntimeCommandV1
from app.platform.runtime.service import RuntimeApplicationService


def test_runtime_application_service_accepts_canonical_commands():
    service = RuntimeApplicationService()
    command = RuntimeCommandV1(
        command_id="cmd-1",
        run_id="hr-1",
        command_type="start",
        payload={"task": "Coordinate work"},
    )

    result = service.accept(command)

    assert result.command_id == "cmd-1"
    assert result.result_type == "accepted"
