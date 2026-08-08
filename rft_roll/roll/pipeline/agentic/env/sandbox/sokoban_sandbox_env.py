from gem import Env
from xrl.sdk.sandbox.config import SandboxConfig
from xrl.sdk.sandbox.client import Sandbox
from xrl.sdk.sandbox.request import CreateBashSessionRequest
from xrl.sdk.sandbox.request import Action
import asyncio
import time
import json
import shlex
from typing import Optional
import datetime

from roll.utils.logging import get_logger

logger = get_logger()

class SokobanSandboxEnv(Env):
    """
    An environment for the Sokoban game that runs inside a secure, isolated sandbox.

    This class manages the entire lifecycle of a sandboxed environment, including:
    - Starting a dedicated cloud container (sandbox).
    - Managing separate, isolated shell sessions for the server and client.
    - Starting the game server as a background process.
    - Sending client commands to interact with the server.
    - Parsing and returning game state.
    - Cleaning up all resources on close.
    """
    def __init__(self,
                config=None,
                server_start_timeout: int = 60):
        """
        Args:
            config: An SDK SandboxConfig object. If None, a default is created.
            server_start_timeout: Seconds to wait for the game server to become healthy.
        """
        if config is None or not hasattr(config, 'base_url'):
            config = SandboxConfig()

        config.image = 'hub.docker.alibaba-inc.com/chatos/sandbox-sokoban:0.0.26'
        config.auto_clear_seconds=60 * 120 # Auto-delete after 2 hours of inactivity
        
        self.sandbox_config = config
        self.server_start_timeout = server_start_timeout
        self.server_port = 8001
        
        self.sandbox = None
        self.session = None
        self.server_session = None        
        self.last_known_map_str: str = "Game has not been reset yet."
        
        self._initialized: bool = False
        
    def reset(self, seed=None, **kwargs):
        """Resets the environment to a new game."""
        self._lazy_init()
        
        command = f"python client.py reset --seed {seed} --json" if seed else "python client.py reset --json"
        
        result = self._execute_command(command)
        
        if isinstance(result, dict):
            game_state = result
            obs = game_state.get('observation', 'Error: Observation not found in reset response.')
            info = game_state.get('info', {})
            
            suffix = info.get('suffix', 'Map not available on reset.')
            map_part = suffix.split('\n', 1)[-1] if '\n' in suffix else suffix
            
            self.last_known_map_str = map_part.strip()
            return obs, info
        else:
            raw_output = result
            obs = "Error: Failed to decode JSON from sandbox on reset."
            info = {"error": "JSONDecodeError", "raw_output": raw_output}
            return obs, info

    def step(self, action: str):
        """Executes one step in the environment."""
        # The quotes around '{action}' handle actions that might contain special characters.
        sanitized_action = action.replace('\n', ' ').replace('\r', '') 
        command = f"python client.py action {shlex.quote(sanitized_action)} --json"
        
        result = self._execute_command(command)
        
        if isinstance(result, dict):
            game_state = result
            obs = game_state.get('observation', 'Error: Observation not found in step response.')
            reward = game_state.get('reward', 0)
            terminated = game_state.get('terminated', True)
            truncated = game_state.get('truncated', False)
            info = game_state.get('info', {})
            
            suffix = info.get('suffix', f"No map update. Last known map:\n{self.last_known_map_str}")
            map_part = suffix.split('\n', 1)[-1] if '\n' in suffix else suffix
            self.last_known_map_str = map_part.strip()
            
            return obs, reward, terminated, truncated, info
            
        else:
            raw_output = result
            obs = "Error: Failed to decode JSON from sandbox on step."
            reward = 0 
            terminated = True 
            truncated = False
            info = {"error": "JSONDecodeError", "raw_output": raw_output}
            return obs, reward, terminated, truncated, info
    
    def render(self) -> str:
        return self.last_known_map_str
    
    def close(self):
        """
        Stops the underlying sandbox instance and releases all cloud resources.
        This is a critical cleanup step.
        """
        if self._initialized and self.sandbox:
            try:
                # The stop() method of the SandboxClient is likely asynchronous,
                # just like start(), so we must run it with asyncio.run().
                asyncio.run(self.sandbox.stop())
            except Exception as e:
                logger.exception("An error occurred while stopping the sandbox.")

    def _wait_for_server(self, timeout: int):
        """
        Periodically polls the server inside the sandbox until it's ready.
        """
        start_time = time.time()
        # This curl command attempts to connect, but discards output (-s -o /dev/null).
        # It writes a custom string with the HTTP status code (-w "..."), which we can check.
        health_check_command = f'curl -s -o /dev/null -w "HC_CODE_%{{http_code}}_HC_CODE" http://localhost:{self.server_port}/'
        
        while time.time() - start_time < timeout:
            try:
                response = asyncio.run(self.sandbox.run_in_session(Action(session="default", command=health_check_command)))
                # The response from curl is in resp.output.
                if "HC_CODE_200_HC_CODE" in response.output:
                    return
            except Exception as e:
                pass
            
            time.sleep(1) # Wait 1 second before retrying.

        raise RuntimeError(f"Server did not start within the {timeout}s timeout.")
    
    def _lazy_init(self):
        """
        Performs one-time, heavy initialization of the sandbox environment.
        
        This method is called "lazily" on the first `reset()` to
        avoid the "thundering herd" problem in distributed settings, where
        multiple workers would try to initialize simultaneously. It handles
        sandbox creation, session setup, and game server startup.
        """
        if self._initialized:
            return

        try:
            # 1. Start the Sandbox Container
            # This provisions the underlying container resource via the sandbox service.
            self.sandbox = Sandbox(self.sandbox_config)
            
            asyncio.run(self.sandbox.start())
            logger.info("Sandbox %s created on host %s (%s)",
                        self.sandbox._sandbox_id, self.sandbox._host_name, self.sandbox._host_ip)
            
            # 2. Create Isolated Execution Sessions
            # Two separate bash sessions are created to prevent the game server's
            # logs from interfering with the clean JSON output of client commands.
            # The 'default' session is used for executing client commands (reset, step).
            self.session = asyncio.run(self.sandbox.create_session(CreateBashSessionRequest(session="default")))
            # The 'server_session' is to contain the noisy server process.
            self.server_session = asyncio.run(self.sandbox.create_session(CreateBashSessionRequest(session="server_session")))
            
            # 3. Launch the Game Server as a Background Process
            # The '&' runs the command as a background process, allowing the run_in_session call to return immediately.
            start_command = "python server.py &"
            asyncio.run(self.sandbox.run_in_session(Action(session="server_session", command=start_command)))

            # 4. Wait for the Server to Become Responsive
            self._wait_for_server(timeout=self.server_start_timeout)
            
            self._initialized = True

        except Exception as e:
            logger.exception("A critical error occurred during LAZY initialization.")
            # In a distributed environment, a failed initialization is fatal for this
            # worker. Raising an exception ensures the framework will handle the
            # failed worker gracefully (e.g., by replacing it).
            raise RuntimeError("Failed to initialize sandbox environment.") from e
                
    def _execute_command(self, command: str) -> Optional[dict]:
        """
        A low-level helper to execute a command and parse its JSON output.
        This is the part that is truly duplicated between reset and step.
        """
        raw_output = ""
        try:
            response = asyncio.run(self.sandbox.run_in_session(Action(session="default", command=command)))
            raw_output = response.output.strip()
            return json.loads(raw_output)
        except json.JSONDecodeError:
            sandbox_info = self._get_sandbox_info_str()
            logger.error("%s Failed to decode JSON from sandbox. Raw output: >>>%s<<<", sandbox_info, raw_output)
            return raw_output
        except Exception as e:
            sandbox_info = self._get_sandbox_info_str()
            logger.exception("%s An unexpected error occurred executing command: %s. Error: %s. ", sandbox_info, command, e)
            return f"{sandbox_info} Unexpected sandbox execution error: {e}"   
        
    def _get_sandbox_info_str(self) -> str:
        """
        Generates a standardized string with sandbox context for logging.

        Returns:
            A formatted string containing the current timestamp, sandbox ID,
            and host information, suitable for prefixing log messages.
            e.g., "Timestamp: ... | Context: [Sandbox: ... on Host: ...]".
        """
        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        
        sandbox_info = "[Sandbox: Not Initialized]"
        if self.sandbox and self.sandbox.sandbox_id:
            sandbox_info = (
                f"[Sandbox: {self.sandbox.sandbox_id} "
                f"on Host: {self.sandbox.host_name} ({self.sandbox.host_ip})]"
            )
            
        return f"Timestamp: {current_time_str} | Context: {sandbox_info}" 