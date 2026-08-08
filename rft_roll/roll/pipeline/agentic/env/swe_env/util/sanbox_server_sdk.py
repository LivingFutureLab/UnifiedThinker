"""
与sanbox交互的底层client写在这里
sdk交互方式。
"""

import time
import uuid
from numpy import True_
from numpy.testing import break_cycles
import requests
import asyncio
import aiohttp
import os
import requests
import uuid
import mimetypes
import concurrent.futures
import re

from xrl.sdk.sandbox.config import SandboxConfig
from xrl.sdk.sandbox.client import Sandbox
from xrl.sdk.sandbox.request import Command, CreateBashSessionRequest, Action
from xrl.sdk.sandbox.response import CommandResponse, FileUploadResponse, Observation
from roll.pipeline.agentic.env.swe_env.utils import pretty_print
from roll.pipeline.agentic.env.swe_env.util.define.log import get_logger

DEBUG = True


class SWERexClientSDK:
    def __init__(self, host: str = "https://xrl-aliyun.alibaba-inc.com/swe-rex/docker", logger=None):
        self.host = host
        self.sandbox = None
        self.sanbox_id = None
        self.host_write = "https://xrl-sandbox.alibaba-inc.com/apis/envs/sandbox/v1"  # 写集群，支持start，适合短命令
        self.host_read = "https://xrl-sandbox-read.alibaba-inc.com/apis/envs/sandbox/v1"  # 读集群，支持arun，适合长期命令，不支持start
        if logger == None:
            self.logger = get_logger("SWERexClinet")
        else:
            self.logger = logger
        pass

    def start_session(self, route_key: str):
        session = requests.Session()
        session.headers.update({"ROUTE-KEY": route_key, "USER-ID": "374702"})
        return session

    def start_container(
        self,
        session: requests.Session,
        route_key: str,
        docker_image: str,
        clear_time: int = 60,
        timeout=180,
        max_execute_time: float = 300.0,
        max_execute_retry: int = 10,
        update_route_key_interval: int = 11,
    ):
        """请求远程服务init_env, pull docker image, 并返回container_name。
        @input:
            - route_key: if None, will be auto-generated.
            - docker_image:
            - clear_time: (min)
            - timeout: (s)
            - update_route_key_interval: 更新route_key的间隔(次)
        @output:
            - return_info:
                - state: "success" or "error" # Attention
                - error_message: error_message if state is "error".
                - route_key: final route_key.
                - container_name: only one container_name.
                - retry_times: final retry_times.
                - session: final session.
        """
        st, execute_time, retry_times = time.time(), 0, 1

        return_info = {
            "state": "error",
            "container_name": "",
            "retry_times": retry_times,
            "error_message": "",
            "route_key": route_key,
            "session": session,
        }
        error_message = ""

        if DEBUG:
            pretty_print("docker_image: ", docker_image)
        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                config = SandboxConfig(
                    image=docker_image,
                    auto_clear_seconds=clear_time * 60,
                    startup_timeout=timeout,
                    xrl_authorization="t-k8ki94jgmi75lolo",  # FIXME: change dynamic author_key
                )
                self.sandbox = Sandbox(config)
                # self.sandbox._url = self.host
                if DEBUG:
                    pretty_print(
                        f"[SANBOX SDK]",
                        f"sandbox created config success, auto_clear_seconds: {clear_time*60} seconds, sandbox_id: {self.sanbox_id}",
                    )
                break
            except Exception as e:
                error_message = repr(e)
                return_info["error_message"] = error_message
                return_info["state"] = "error"
                if DEBUG:
                    pretty_print(
                        "[SANBOX SDK][START CONTAINER ERROR]",
                        f"(retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, docker_image: {docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, sandbox_id: {self.sanbox_id}, error: {error_message}",
                    )
                time.sleep(1)
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                asyncio.run(self.sandbox.start())
                asyncio.run(self.sandbox.create_session(CreateBashSessionRequest(session="bash-1")))
                return_info["state"] = "success"
                return_info["container_name"] = self.sandbox.sandbox_id
                return_info["retry_times"] = retry_times
                self.sanbox_id = self.sandbox.sandbox_id
                if DEBUG:
                    pretty_print(
                        "[SANBOX SDK][START CONTAINER SUCCESS]",
                        f"(retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, docker_image: {docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, sandbox_id: {self.sanbox_id}",
                    )
                self.logger.info(
                    f"[SANBOX SDK][START CONTAINER SUCCESS](retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, docker_image: {docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, sandbox_id: {self.sanbox_id}"
                )
                return return_info
            except Exception as e:
                error_message = repr(e)
                return_info["error_message"] = error_message
                return_info["state"] = "error"
                if DEBUG:
                    pretty_print(
                        "[SANBOX SDK][START CONTAINER ERROR]",
                        f"(retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, docker_image: {docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, sandbox_id: {self.sanbox_id}, error: {error_message}",
                    )
                self.logger.info(
                    f"[SANBOX SDK][START CONTAINER ERROR](retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, docker_image: {docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, sandbox_id: {self.sanbox_id}, error: {error_message}"
                )
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)

        if DEBUG:
            pretty_print(
                "[SANBOX SDK][START CONTAINER ERROR]",
                f"(retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, docker_image: {docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, sandbox_id: {self.sanbox_id}, error: {error_message}",
            )
        self.logger.info(
            f"[SANBOX SDK][START CONTAINER ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, docker_image: {docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, sandbox_id: {self.sanbox_id}, error: {error_message}"
        )
        return return_info

    async def stop_container(self, route_key: str, container_name: str):
        """
        @input:
            - container_name
            - route_key
        """
        try:
            if container_name:
                asyncio.create_task(self.sandbox.stop())
        except Exception as e:
            if DEBUG:
                pretty_print(
                    "[SANBOX SDK][STOP CONTAINNER ERROR]",
                    f"[{os.getpid()}]route_key: {route_key}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, error message: {repr(e)}",
                )
            self.logger.info(
                f"[SANBOX SDK][STOP CONTAINNER ERROR]([{os.getpid()}])route_key: {route_key}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, error message: {repr(e)}"
            )

    def run_in_session(
        self,
        session: requests.Session,
        command: str,
        workdir: str = None,
        environment: dict = None,  # 这两个在之前的sanbox里
        container_name: str = None,
        timeout: int = 180,
        max_execute_time: float = 300.0,
        max_execute_retry: int = 10,
        route_key: str = None,
        session_name: str = None,
        is_interactive: bool = False,
        expect: list[str] = None,
        check: str = "silent",
        action_type: str = "bash",
        arun=True,
    ):
        """ """
        retry_times, execute_time = 0, 0
        st = time.time()
        error_message = ""
        response = ""
        if not session_name:
            session_name = route_key
        return_info = {
            "state": "error",
            "route_key": route_key,
            "session_name": session_name,
            "container_name": container_name,
            "retry_times": retry_times,
            "error_message": error_message,
            "session": session,
        }
        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                # TODO：arun, nohup，循环访问PID，重定向结果。
                if arun:
                    self.sandbox._url = self.host_read
                    response: Observation = asyncio.run(self.sandbox.arun(command, mode="nohup"))
                else:
                    self.sandbox._url = self.host_write
                    response: Observation = asyncio.run(
                        self.sandbox.run_in_session(
                            Action(session="bash-1", command=command, acition_type=action_type, check=check)
                        )
                    )
                # if 'chmod' in command:
                #     if DEBUG: pretty_print('[SANBOX SDK]', '[chmod in command]')
                if DEBUG:
                    pretty_print(
                        f"[SANBOX SDK][RUN IN SESSION SUCCESS]",
                        f"route_key: {route_key}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, command: {command}, response: {response}",
                    )
                self.logger.info(
                    f"[SANBOX SDK][RUN IN SESSION SUCCESS]route_key: {route_key}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, command: {command}, response: {response}"
                )
                return {"stdout": response.output, "stderr": "", "exit_code": response.exit_code}
            except Exception as e:
                error_message = repr(e)
                if DEBUG:
                    pretty_print(
                        "[SANBOX SDK][RUN IN SESSION ERROR]",
                        f"route_key: {route_key}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, error message: {repr(e)}, command: {command}, response: {response}",
                    )
                self.logger.info(
                    f"[SANBOX SDK][RUN IN SESSION ERROR]route_key: {route_key}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, error message: {repr(e)}, command: {command}, response: {response}"
                )
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
            return_info["error_message"] = error_message
            return_info["retry_times"] = retry_times

        if DEBUG:
            pretty_print(
                "[SANBOX SDK][RUN IN SESSION ERROR]",
                f"(retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}, command: {command}",
            )
        self.logger.info(
            f"[SANBOX SDK][RUN IN SESSION ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}, command: {command}"
        )
        return {"stdout": "Error: network connection timeout, no response", "stderr": "", "exit_code": -1}

    def execute(
        self,
        session: requests.Session,
        command: str | list[str],
        workdir: str = "",
        environment: dict = None,
        container_name: str = None,
        timeout: int = 180,
        max_execute_time: float = 300.0,
        max_execute_retry: int = 10,
        route_key: str = "",
    ) -> dict:
        """
        @执行成功的返回
            {
                'stdout': "Error executing command:\\n\\n[STDOUT]\\n\\n \\n\\n[STDERR]\\n\\npython: can't open file '/testbed/reproduce_issue.py': [Errno 2] No such file or directory\\n",
                'stderr': '',
                'exit_code': 2
            }
        @执行失败的返回(一般是hang住没有返回)
            {
                "stdout": "Error: network connection timeout, no response",
                "stderr": "",
                "exit_code": -1
            }
        """
        data = {
            "command": command,
            "cwd": workdir,
            "env": environment,
            "container_name": container_name,
            "timeout": timeout,
        }

        st, execute_time, retry_times = time.time(), 0, 0
        error_message = ""
        # logger.info(f'[START EXCUTE]execute command: {command}, container_name: {container_name}, timeout: {timeout}')

        response = None
        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                response: CommandResponse = asyncio.run(self.sandbox.execute(Command(command=command)))
                self.logger.info(
                    f"[SANBOX SDK][EXCUTE SUCCESS](RETRY_times:{retry_times}) execute command: {command}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, timeout: {timeout}, response: {response.stdout + response.stderr}"
                )
                if DEBUG:
                    pretty_print(
                        "[SANBOX SDK][EXCUTE SUCCESS]",
                        f"(RETRY_times:{retry_times}) execute command: {command}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, timeout: {timeout}, response: {response.stdout + response.stderr}",
                    )
                return {"stdout": response.stdout, "stderr": response.stderr, "exit_code": response.exit_code}
                # else:
                # self.logger.info(f'[EXCUTE UNKOWN ERROR](RETRY_times:{retry_times}) execute command: {command}, container_name: {container_name}, timeout: {timeout}, response: \n{response.stdout + response.stderr}')
            except Exception as e:
                error_message = repr(e)
                self.logger.info(
                    f"[SANBOX SDK][EXCUTE ERROR](RETRY_times:{retry_times}) execute command: {command}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, timeout: {timeout}, error: {error_message}, response: {response}"
                )
                if DEBUG:
                    pretty_print(
                        "[SANBOX SDK][EXCUTE ERROR]",
                        f"(RETRY_times:{retry_times}) execute command: {command}, container_name: {container_name}, sandbox_id: {self.sanbox_id}, timeout: {timeout}, error: {error_message}, response: {response}",
                    )
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        if DEBUG:
            pretty_print(
                "[SANBOX SDK][EXECUTE FINAL ERROR]",
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}(retry_times:{retry_times})(execute_time: {execute_time})(timeout: {timeout}) route_key: {route_key}, sandbox_id: {self.sanbox_id}, input: {[data]}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}, response: {response}',
            )
        self.logger.info(
            f"[SANBOX SDK][EXECUTE FINAL ERROR](retry_times:{retry_times})(execute_time: {execute_time})(timeout: {timeout})"
            f"route_key: {route_key}, sandbox_id: {self.sanbox_id}, input: {[data]}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}, response: {response}"
        )
        return {"stdout": "Error: network connection timeout, no response", "stderr": "", "exit_code": -1}

    def copy_to_container(
        self,
        src_path: str,
        dest_path: str,
        session: requests.Session = None,
        container_name: str = None,
        max_execute_time: float = 300.0,
        max_execute_retry: int = 10,
        route_key: str = None,
    ):
        """
        Copies a file or directory from the host to the Docker container.

        Args:
            src_path: Path to the file or directory on the host.
            dest_path: Destination path inside the container.
        """
        session = session if session else self.session
        container_name = container_name if container_name else self.container_name
        route_key = route_key if route_key else self.route_key
        max_execute_time = max_execute_time if max_execute_time else self.max_execute_time
        max_execute_retry = max_execute_retry if max_execute_retry else self.max_execute_retry

        content_type, _ = mimetypes.guess_type(src_path)
        if content_type is None:
            content_type = "application/octet-stream"
        data = {
            "target_path": dest_path,
            "container_name": container_name,
        }

        st, execute_time, retry_times = time.time(), 0, 0
        timeout, error_message = 180, ""
        files = None  # 初始化 files 变量

        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                with open(src_path, "rb") as local_file:
                    files = {"file": (os.path.basename(dest_path), local_file, content_type)}
                    response: FileUploadResponse = asyncio.run(self.sandbox.upload(src_path, dest_path))
                if response.success:
                    # print(f"[SANBOX SDK][COPY TO CONTAINER SUCCESS](retry_times:{retry_times})(execute_time: {execute_time})file: {src_path}, dest_path: {dest_path}, container_name: {container_name}")
                    self.logger.info(
                        f"[SANBOX SDK][COPY TO CONTAINER SUCCESS](retry_times:{retry_times})(execute_time: {execute_time})file: {src_path}, dest_path: {dest_path}, container_name: {container_name}"
                    )
                    return "SUCCESS"
            except Exception as e:
                error_message = repr(e)
                self.logger.info(
                    f"[SANBOX SDK][COPY TO CONTAINER ERROR](retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, sandbox_id: {self.sanbox_id}, data: {data}, files: {files}, error_message: {error_message}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}"
                )
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        if DEBUG:
            pretty_print(
                "[SANBOX SDK][COPY TO CONTAINER ERROR]",
                f"(retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, sandbox_id: {self.sanbox_id}, data: {data}, files: {files}, error_message: {error_message}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}",
            )
        self.logger.info(
            f"[SANBOX SDK][COPY TO CONTAINER ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, sandbox_id: {self.sanbox_id}, data: {data}, files: {files}, error_message: {error_message}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}"
        )
        return "ERROR"

    def create_file(
        self,
        file_path: str,
        content: str,
        session: requests.Session = None,
        container_name: str = None,
        max_execute_time: float = None,
        max_execute_retry: int = None,
        route_key: str = None,
    ):

        route_key = route_key if route_key else self.route_key
        container_name = container_name if container_name else self.container_name
        max_execute_time = max_execute_time if max_execute_time else self.max_execute_time
        max_execute_retry = max_execute_retry if max_execute_retry else self.max_execute_retry
        session = session if session else self.session

        # create a local file with the content
        data = {
            "content": content,
            # "container_name": container_name,
            "sandbox_id": self.sanbox_id,  # attention
            "path": f"{file_path}",
        }
        st, execute_time, retry_times = time.time(), 0, 0
        timeout, error_message = 180, ""

        while retry_times < max_execute_retry and execute_time < max_execute_time:
            try:
                # FIX ME: use sdk when sdk support wrire_file
                response = asyncio.run(self.sandbox.write_file(content=content, path=file_path))
                if response.success:
                    response_data = response.message
                    if DEBUG:
                        pretty_print("[SANBOX SDK][CREATE FILE SUCCESS]", f"response_data: {response_data}")
                    self.logger.info(f"[SANBOX SDK][CREATE FILE SUCCESS]response: {response_data}, data: {data}")
                    return response.message
                else:
                    if DEBUG:
                        pretty_print("[SANBOX SDK][CREATE FILE ERROR]", f"response: {response}")
                    self.logger.info(f"[SANBOX SDK][CREATE FILE ERROR]response: {response}, data: {data}")
            # except requests.exceptions.RequestException as e:
            except Exception as e:
                error_message = repr(e)
                self.logger.info(
                    f'[SANBOX SDK][CREATE FILE ERROR](retry_times:{retry_times})(execute_time: {execute_time}) \
                    f"route_key: {route_key}, sandbox_id: {self.sanbox_id}, input: {[data]}, timeout: {timeout}, error: {error_message}, link: {f"{self.host}/write_file"}'
                )
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        self.logger.info(
            f"[SANBOX SDK][CREATE FILE ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, sandbox_id: {self.sanbox_id}, input: {[data]}, timeout: {timeout}, error: {error_message}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}"
        )
        if DEBUG:
            pretty_print(
                "[SANBOX SDK][CREATE FILE ERROR]",
                f"(retry_times:{retry_times})(execute_time: {execute_time}) route_key: {route_key}, sandbox_id: {self.sanbox_id}, input: {[data]}, timeout: {timeout}, error: {error_message}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}",
            )

        return "ERROR"

    def clean_stdout(self, stdout):
        # 清理空的STDOUT和STDERR标签
        # 删除空的STDOUT标签（如果后面没有内容）
        stdout = re.sub(r"\[STDOUT\]\n\n \n\n\[STDERR\]", "[STDERR]", stdout)
        stdout = re.sub(r"\[STDOUT\]\n\n\[STDERR\]", "[STDERR]", stdout)
        stdout = re.sub(r"\[STDOUT\]\n\n \n", "", stdout)
        stdout = re.sub(r"\[STDOUT\]\n\n", "", stdout)
        # 清理多余的换行符
        stdout = re.sub(r"\n\n\[STDERR\]\n\n", "\n[STDERR]\n", stdout)
        stdout = re.sub(r"\n\n\[STDERR\]", "\n[STDERR]", stdout)
        # 如果STDERR也是空的，删除整个标签
        stdout = re.sub(r"\[STDERR\]\n\n$", "", stdout)
        stdout = re.sub(r"\[STDERR\]\n$", "", stdout)
        return stdout

    def setup_run(
        self,
        workdir=None,
        session: requests.Session = None,
        container_name: str = None,
        max_execute_time: float = 300.0,
        max_execute_retry: int = 10,
        timeout: int = 180,
        route_key: str = None,
    ):
        self.workdir = workdir
        self.session = session
        self.container_name = container_name
        self.max_execute_time = max_execute_time
        self.max_execute_retry = max_execute_retry
        self.route_key = route_key
        self.timeout = timeout

    def _should_use_nohup(self, command: str) -> bool:
        """
        判断命令是否需要使用 nohup 执行

        Args:
            command: 要执行的命令

        Returns:
            True if the command should use nohup, False otherwise
        """
        # 需要 nohup 的命令模式
        nohup_patterns = [
            # 下载
            r"pip\s+install",
            r"uv\s+pip\s+install",
            r"python\s+-m\s+pip\s+install",
            r"poetry\s+install",
            r"pipenv\s+install",
            r"setup\.py\s+install",
            r"python\s+setup\.py",
            r"conda\s+activate",
            r"source\s+.*activate",
            # 运行单测
            r"run_tests\.sh",
        ]
        command_lower = command.lower().strip()
        for pattern in nohup_patterns:
            if re.search(pattern, command_lower):
                return True
        return False

    def run(
        self, code: str, args: str = "", timeout: int = None, max_execute_time: int = 60 * 4, arun=True
    ) -> tuple[str, str]:
        """
        General method to execute code or commands in the container, with a timeout.

        :param code: The code or command to execute.
        :param args: Arguments to pass to the code/script.
        :param workdir: The working directory inside the container (optional).
        :return: A tuple containing (stdout, error_ecode). If no error, error_message is the exit code (str).
            - "124": post timeout
            - "-1": post fialed
            - "0": execute success
            - "1": post success, but get execute error
        """
        if not timeout:
            timeout = self.timeout
        DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        # command = f"timeout {timeout} export PATH={DOCKER_PATH} && {code} {args}"
        command = f"timeout {timeout} {code} {args}"

        # try:
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Notice we do NOT set tty=True here
        # future = executor.submit(
        exec_result = self.run_in_session(
            session=self.session,
            command=command,
            workdir=self.workdir,
            environment={"PATH": DOCKER_PATH},
            container_name=self.container_name,
            timeout=timeout,
            max_execute_time=max_execute_time,
            max_execute_retry=self.max_execute_retry,
            route_key=self.route_key,
            arun=arun,
        )
        if DEBUG:
            pretty_print("[SANBOX SDK][RUN]exec_result: ", f"{exec_result}")
        self.logger.info(f"[SANBOX SDK][RUN]exec_result: {exec_result}")
        # exec_result = future.result(timeout=timeout)
        # Retrieve output and exit code
        if isinstance(exec_result, dict):
            stdout = self.clean_stdout(exec_result["stdout"] + exec_result["stderr"])
            exit_code = exec_result["exit_code"]
        else:
            if DEBUG:
                pretty_print(
                    "[SANBOX SDK][RUN]",
                    f"exec_result type error: {type(exec_result)}, exec_result: {exec_result}, sandbox_id: {self.sanbox_id}",
                )
            self.logger.info(
                f"[SANBOX SDK][RUN]exec_result type error: {type(exec_result)}, exec_result: {exec_result}, sandbox_id: {self.sanbox_id}"
            )
            raise ValueError(f"type(exec_result): {type(exec_result)}, exec_result: {exec_result}")

        # return stdout & return exit_code
        return_stdout, return_code = "", ""
        if exit_code == 124:
            return_stdout, return_code = f"The command took too long to execute (>{timeout}s)", "-1"
        # elif exit_code == -100:
        #     return_stdout, return_code = f"Error: network connection timeout, no response", "-1"
        elif exit_code != 0 and exit_code != 1:
            # output: [{'stdout': '', 'stderr': '/bin/sh: 1: cannot open /parameter: No such file\n', 'exit_code': 2}]  # TODO：这里可以设计不同的reward
            return_stdout, return_code = stdout, str(exit_code)
        else:
            # success post: Remove ANSI escape codes and \r characters
            return_stdout, return_code = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout), str(exit_code)
        if exit_code == 0 and return_stdout.strip() == "":
            return_stdout = "success"

        self.logger.info(
            f"[SANBOX执行命令的返回]command: {[command]}, sandbox_id: {self.sanbox_id}\n"
            f"exit_code: {exit_code}, exec_result: {exec_result}\n"
            f"return_code: {return_code}, return_stdout: {return_stdout}\n"
        )
        return return_stdout, return_code


if __name__ == "__main__":
    from tests.agentic.sweenv.utils.utils_file import load_data_json

    ds = load_data_json("/home/lixing/workspace/future_update/ROLL_version/ScaleAligner/data/swe_verified_iflow.json")
    if DEBUG:
        pretty_print("==== ds keys: ", f"{ds.keys()}")
    docker_image = ds["docker_image"]
    docker_image = (
        "rex-registry.cn-hangzhou.cr.aliyuncs.com/slimshetty/swebench-verified:sweb.eval.x86_64.astropy__astropy-12907"
    )
    if DEBUG:
        pretty_print("==== docker_image: ", f"{docker_image}")

    # client = SWERexClientSDK(host ='https://xrl-aliyun.alibaba-inc.com/swe-rex/docker')
    client = SWERexClientSDK(host="https://xrl-sandbox.alibaba-inc.com/apis/envs/sandbox/v1")
    session = client.start_session(route_key="test")

    # test sdk start
    return_info = client.start_container(session, "test", docker_image)
    if DEBUG:
        pretty_print("==== return_info: ", f"{return_info}")
    client.setup_run()

    # test sdk execute
    exec_result = client.run("ls", timeout=300)

    # print('正在初始化conda环境... ')
    # client.run("conda init bash", timeout=300)
    # client.run("source ~/.bashrc", timeout=300)
    # client.run("eval \"$(conda shell.bash hook)\"", timeout=300)
    # client.run("conda activate testbed", timeout=300)
    # print('conda环境初始化完成')

    # test sdk upload
    if DEBUG:
        pretty_print("[SANBOX SDK]", "正在拷贝文件...")
    file = "/home/lixing/linxin/ScaleAligner/roll/agentic/env/swe_env/util/sanbox_server_sdk.py"
    client.copy_to_container(file, f"/usr/local/bin/test")
    if DEBUG:
        pretty_print("[SANBOX SDK]", "拷贝文件完成...")

    # test sdk write_file
    if DEBUG:
        pretty_print("[SANBOX SDK]", "正在写入文件...")
    file = "/home/lixing/linxin/ScaleAligner/roll/agentic/env/swe_env/util/sanbox_server_sdk.py"
    temp_content = "print('hello world')"
    escaped_content = temp_content.replace('"', '\\"').replace("$", "\\$")
    client.run(f'echo "{escaped_content}" > /test.py', timeout=300)
    if DEBUG:
        pretty_print("[SANBOX SDK]", "写入文件完成...")

    if DEBUG:
        pretty_print("[SANBOX SDK]", "\n**运行写入的文件**")
    client.run(f"python3 /test.py", timeout=300)
    if DEBUG:
        pretty_print("[SANBOX SDK]", "运行写入的文件完成...")

    if DEBUG:
        pretty_print("[SANBOX SDK]", "\n**正在创建文件**")
    client.create_file("./test_lixing.py", "print('hello world')", session, container_name=None)
    if DEBUG:
        pretty_print("[SANBOX SDK]", "创建文件完成...")

    print("查看写入的文件：")
    client.run(f"cat ./test_lixing.py", timeout=300)
    print("查看写入的文件完成...")

    asyncio.run(client.stop_container("test", "test"))
