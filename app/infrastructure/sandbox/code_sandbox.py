import asyncio
import docker
import json
import os
import tempfile
from typing import Optional, Dict, Any
from datetime import datetime


class CodeSandbox:
    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        memory_limit: str = "256m",
        cpu_limit: float = 0.5,
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self._client: Optional[docker.DockerClient] = None

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _create_container(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            container = self.client.containers.run(
                image=self.image,
                command=f"python3 {os.path.basename(temp_path)}",
                volumes={temp_path: {"bind": f"/code/{os.path.basename(temp_path)}", "mode": "ro"}},
                working_dir="/code",
                detach=True,
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=int(self.cpu_limit * 100000),
                network_disabled=True,
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
            )
            return container.id
        finally:
            os.unlink(temp_path)

    async def execute(self, code: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        try:
            container_id = await loop.run_in_executor(
                None, lambda: self._create_container(code)
            )
            container = self.client.containers.get(container_id)

            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: container.wait()),
                    timeout=self.timeout,
                )
                logs = container.logs(stdout=True, stderr=True, tail=100).decode("utf-8")
                exit_code = container.attrs["State"]["ExitCode"]

                return {
                    "success": exit_code == 0,
                    "output": logs,
                    "exit_code": exit_code,
                    "container_id": container_id,
                    "executed_at": datetime.utcnow().isoformat(),
                }
            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        except asyncio.TimeoutError:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {self.timeout}s",
                "exit_code": -1,
                "timeout": True,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "exit_code": -1,
            }


code_sandbox = CodeSandbox()


async def execute_code(code: str) -> Dict[str, Any]:
    return await code_sandbox.execute(code)
