from __future__ import annotations

from app.platform.contracts.runtime_protocol import RuntimeCommandV1, RuntimeResultV1


class RuntimeApplicationService:
    def accept(self, command: RuntimeCommandV1) -> RuntimeResultV1:
        return RuntimeResultV1(
            command_id=command.command_id,
            run_id=command.run_id,
            result_type="accepted",
            payload={"command_type": command.command_type},
        )
