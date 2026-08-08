"""
与sanbox交互的底层client写在这里.
因为环境变量问题，改成session交互方式。
"""

import time
import uuid
import requests
import asyncio
import aiohttp
import os
import requests
import uuid
import mimetypes
import concurrent.futures
import re

from roll.pipeline.agentic.env.swe_env.util.define.log import get_logger


class SWERexClientSession:
    def __init__(self, host: str = "https://xrl-aliyun.alibaba-inc.com/swe-rex/docker", logger=None):
        self.host = host
        if logger == None:
            self.logger = get_logger("SWERexClinet")
        else:
            self.logger = logger
        pass

    def start_session(self, route_key: str):
        session = requests.Session()
        session.headers.update({"ROUTE-KEY": route_key})
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

        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                response = session.post(
                    f"{self.host}/init_env",
                    json={"image": docker_image, "auto_clear_time": clear_time},  # default 60 min
                    timeout=(10, timeout),
                )
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success":
                        return_info["state"] = "success"
                        return_info["container_name"] = response_data.get("result").get("container_name")
                        return_info["retry_times"] = retry_times
                        return_info["route_key"] = route_key
                        self.create_session(
                            route_key,
                            return_info["container_name"],
                            session_name=None,
                            timeout=timeout,
                            max_execute_time=max_execute_time,
                            max_execute_retry=max_execute_retry,
                        )
                        return return_info
            except Exception as e:
                error_message = repr(e)
                return_info["error_message"] = error_message

            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)

            if retry_times % update_route_key_interval == 0:
                route_key = uuid.uuid4().hex
                session = requests.Session()
                session.headers.update({"ROUTE-KEY": route_key})
                return_info["session"] = session
        print(
            f"[ERROR SANBOX SERVER][START CONTAINER](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, docker_image: {docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}"
        )
        self.logger.info(
            f"[ERROR SANBOX SERVER][START CONTAINER](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, docker_image: {docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}"
        )
        return return_info

    def create_session(
        self,
        route_key: str,
        container_name: str,
        session_name: str = None,
        timeout: int = 180,
        max_execute_time: float = 300.0,
        max_execute_retry: int = 10,
        max_read_size: int = 2000,
    ):
        session = requests.Session()
        if not session_name:
            session_name = route_key
        retry_times, execute_time = 0, 0
        st = time.time()
        error_message = ""
        return_info = {
            "state": "error",
            "container_name": container_name,
            "retry_times": retry_times,
            "error_message": error_message,
            "route_key": route_key,
            "session": session,
        }
        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                response = session.post(
                    f"{self.host}/create_session",
                    headers={"ROUTE-KEY": route_key},
                    json={
                        "session": session_name,
                        "container_name": container_name,
                        "session_type": "bash",  # 目前仅支持bash
                        "startup_timeout": timeout,
                        # "max_read_size": max_read_size, # 默认2000, 若命令需要输出很多需设置大些，不然会导致命令卡住）
                        "env": {"HOME": "/root"},
                        "env_enable": True,  # 是否继承主进程的环境变量（默认为True）
                        # startup_source, 执行命令前source的文件（字符串列表，选填）
                    },
                    # timeout = timeout
                )
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success":
                        return_info["state"] = "success"
                        return_info["session"] = session
                        return return_info
                    else:
                        error_message = response_data
            except Exception as e:
                error_message = repr(e)
                print(
                    f"[ERROR SANBOX SERVER][CREATE SESSION ERROR] route_key: {route_key}, container_name: {container_name}, error message: {error_message}"
                )
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        print(
            f"[ERROR SANBOX SERVER][CREATE SESSION ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, container_name: {container_name}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}"
        )
        self.logger.info(
            f"[ERROR SANBOX SERVER][CREATE SESSION ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, container_name: {container_name}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}"
        )
        return session

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
    ):
        """ """
        retry_times, execute_time = 0, 0
        st = time.time()
        error_message = ""
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
                response = session.post(
                    f"{self.host}/run_in_session",
                    json={
                        "container_name": container_name,
                        "command": command,
                        "timeout": timeout,
                        "session": session_name,  # 这个的必要性？
                        "timeout": timeout,
                        "action_type": action_type,
                        "is_interactive": is_interactive,  # 这个是什么
                        "check": check,
                        # expect：交互式命令时期待的输出（list[str]，字符串可为正则表达式，若交互式命令的输出没有匹配到会一直卡到timeout结束）
                        # check：退出模式，raise（默认值，返回exit code，若非0则抛出异常NonZeroExitCodeError）、silent（返回exit code，非0不抛异常）、ignore（不返回exit code）
                    },
                )
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success":
                        return {
                            "stdout": response_data.get("result").get("output"),
                            "stderr": "",
                            "exit_code": response_data.get("result").get("exit_code"),
                        }
                    else:
                        error_message = response_data
                print("[run_in_session][error][response]", response.json())
            except Exception as e:
                error_message = repr(e)
                print(
                    f"[ERROR SANBOX SERVER][RUN IN SESSION ERROR] route_key: {route_key}, container_name: {container_name}, error message: {repr(e)}, command: {command}"
                )
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
            print(
                f"[ERROR SANBOX SERVER][RUN IN SESSION ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
                f"route_key: {route_key}, container_name: {container_name}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}, command: {command}"
            )
            self.logger.info(
                f"[ERROR SANBOX SERVER][RUN IN SESSION ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
                f"route_key: {route_key}, container_name: {container_name}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}, command: {command}"
            )
            return_info["error_message"] = error_message
            return_info["retry_times"] = retry_times
        return {"stdout": "Error: network connection timeout, no response", "stderr": "", "exit_code": -1}

    async def stop_container(self, route_key: str, container_name: str):
        """
        @input:
            - container_name
            - route_key
        """
        try:
            if container_name:
                asyncio.create_task(self.stop_container_async(route_key, container_name))
        except Exception as e:
            print(
                f"\n[ERROR][{os.getpid()}][STOP CONTAINNER ERROR] route_key: {route_key}, container_name: {container_name}, error message: {repr(e)}"
            )
            self.logger.info(
                f"[ERROR][{os.getpid()}][STOP CONTAINNER ERROR] route_key: {route_key}, container_name: {container_name}, error message: {repr(e)}"
            )

    async def stop_container_async(self, route_key, container_name):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.host}/stop",
                headers={"ROUTE-KEY": f"{route_key}"},
                json={"container_name": f"{container_name}"},
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    if response_data.get("status") == "Success":
                        print(f"Container {container_name} stopped successfully, route_key: {route_key}")
                        return
                print(f"Container stop request failed with status: {response}")

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

        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                response = session.post(f"{self.host}/execute", json=data, timeout=timeout)
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success":
                        self.logger.info(
                            f'[EXCUTE SUCCESS](RETRY_times:{retry_times}) execute command: {command}, container_name: {container_name}, timeout: {timeout}, response: {response_data.get("result")}'
                        )
                        return response_data.get("result")
                else:
                    self.logger.info(
                        f"[EXCUTE UNKOWN ERROR](RETRY_times:{retry_times}) execute command: {command}, container_name: {container_name}, timeout: {timeout}, response: \n{response.json()}"
                    )
            except Exception as e:
                error_message = repr(e)
                self.logger.info(
                    f"[EXCUTE ERROR](RETRY_times:{retry_times}) execute command: {command}, container_name: {container_name}, timeout: {timeout}, error: {error_message}"
                )
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[ERROR][EXECUTE ERROR](retry_times:{retry_times})(execute_time: {execute_time})(timeout: {timeout})"
            f"route_key: {route_key}, input: {[data]}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}, response: {response}"
        )
        self.logger.info(
            f"[ERROR][EXECUTE ERROR](retry_times:{retry_times})(execute_time: {execute_time})(timeout: {timeout})"
            f"route_key: {route_key}, input: {[data]}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, error: {error_message}, response: {response}"
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

        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                with open(src_path, "rb") as local_file:
                    files = {"file": (os.path.basename(dest_path), local_file, content_type)}
                    response = session.post(
                        f"{self.host}/upload",
                        headers={"ROUTE-KEY": route_key},
                        data=data,
                        files=files,
                        timeout=timeout,
                    )
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success":
                        return "SUCCESS"
            except requests.exceptions.RequestException as e:
                error_message = repr(e)
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        print(
            f"[ERROR][COPY TO CONTAINER ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, data: {data}, files: {files}, error_message: {error_message}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}"
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
        data = {"content": content, "container_name": container_name, "path": f"{file_path}"}
        st, execute_time, retry_times = time.time(), 0, 0
        timeout, error_message = 180, ""
        while retry_times < max_execute_retry and execute_time < max_execute_time:
            try:
                response = session.post(
                    f"{self.host}/write_file", headers={"ROUTE-KEY": route_key}, json=data, timeout=timeout
                )
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success":
                        return "SUCCESS"
            except requests.exceptions.RequestException as e:
                error_message = repr(e)
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        self.logger.info(
            f"[ERROR][CREATE FILE ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {route_key}, input: {[data]}, timeout: {timeout}, error: {error_message}, max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}"
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

    def run(self, code: str, args: str = "", timeout: int = None, max_execute_time: int = 60 * 4) -> tuple[str, str]:
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
        command = f"timeout {timeout} {code} {args}"
        DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

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
        )
        print("[exec_result]", exec_result)
        # exec_result = future.result(timeout=timeout)
        # Retrieve output and exit code
        if isinstance(exec_result, dict):
            stdout = self.clean_stdout(exec_result["stdout"] + exec_result["stderr"])
            exit_code = exec_result["exit_code"]
        else:
            print(f"[SANBOX][RUN]exec_result type error: {type(exec_result)}, exec_result: {exec_result}")
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
            f"[SANBOX执行命令的返回]command: {[command]}\n"
            f"exit_code: {exit_code}, exec_result: {exec_result}\n"
            f"return_code: {return_code}, return_stdout: {return_stdout}\n"
        )
        return return_stdout, return_code

        # ## timeout
        # except concurrent.futures.TimeoutError:
        #     return_code = "-1"
        #     return_stdout = f"The command took too long to execute (>{timeout}s)"
        #     self.logger.info(f"[SANBOX][RUN]command: {[command]}\n" \
        #                 f"exit_code: {exit_code}, exec_result: {exec_result}\n" \
        #                 f"return_code: {return_code}, return_stdout: {return_stdout}\n"
        #     )
        #     return return_stdout, return_code

        # except Exception as e:
        #     return_code = "-1"
        #     return_stdout = f"Error: {repr(e)}"
        #     self.logger.info(f"[SANBOX][RUN]command: {[command]}\n" \
        #                 f"exit_code: {exit_code}, exec_result: {exec_result}\n" \
        #                 f"return_code: {return_code}, return_stdout: {return_stdout}"
        #     )
        #     return return_stdout, return_code

    # def read_file(self, rel_file_path: str) -> str:
    #     output, _ = self.run(f"cat /{self.alt_path}/{rel_file_path}")
    #     return output

    # def run_tests(self) -> str:
    #     output, _= self.run(f"bash {self.alt_path}/run_tests.sh", timeout=300)
    #     # Remove ANSI escape codes and \r characters
    #     output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
    #     return output
