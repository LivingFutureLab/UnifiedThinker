import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import gym
from requests import session
import uuid
import re

from roll.pipeline.agentic.env.swe_env.util.define.action import Action
from roll.pipeline.agentic.env.swe_env.util.define.observation import Observation
from roll.pipeline.agentic.env.swe_env.util.define.log import get_logger
from roll.pipeline.agentic.env.swe_env.util.define.commands import ParseCommandBash
from roll.pipeline.agentic.env.swe_env.util.define.action import Action
from roll.pipeline.agentic.env.swe_env.util.get_requirment.swebench_test_spec import make_test_spec
from roll.pipeline.agentic.env.swe_env.util.spec.config import SKIP_FILES_NEW
from roll.pipeline.agentic.env.swe_env.util.spec.swe_reward import SweReward
from tests.agentic.sweenv.utils.utils_file import load_data_txt

DEBUG = False

"""
调用sanbox_server执行命令，init_env
"""


class RepoClient(gym.Env):
    def __init__(
        self,
        logger=None,
        max_execute_time=300.0,
        max_execute_retry=10,
        timeout=180,
        max_env_time=60 * 60,
        sanbox_mode="http",
        swe_rex_host="https://xrl-aliyun.alibaba-inc.com/swe-rex/docker",
    ):
        # Get the logger
        if logger is None:
            self.logger = get_logger("RepoClient")  # Pass the module name for clarity
        else:
            self.logger = logger

        # init base path
        self.repo_path = "/testbed"
        self.alt_path = "/root"
        self.cmd_parser = ParseCommandBash()

        # init sanbox server
        self.sanbox_host = (
            swe_rex_host
            if swe_rex_host
            else os.getenv("SWE_REX_HOST", "https://xrl-aliyun.alibaba-inc.com/swe-rex/docker")
        )

        print("[RepoEnv][注意]sanbox_mode:", sanbox_mode)
        self.sanbox_mode = sanbox_mode
        if self.sanbox_mode == "sdk":
            from roll.pipeline.agentic.env.swe_env.util.sanbox_server_sdk import SWERexClientSDK as SWERexClient
        elif self.sanbox_mode == "http":
            from roll.pipeline.agentic.env.swe_env.util.sanbox_server import SWERexClient
        else:
            from roll.pipeline.agentic.env.swe_env.util.sanbox_server import SWERexClient
        self.runtime = SWERexClient(host=self.sanbox_host, logger=self.logger)

        # init swe reward
        self.swe_reward = SweReward(logger=self.logger)

        # init for spec
        self.session = None
        self.container_name = None
        self.route_key = None
        self.max_execute_time = max_execute_time
        self.max_execute_retry = max_execute_retry
        self.timeout = timeout
        #
        self.retry_times = None
        self.observation = None
        self.done = False
        self.state = "init"
        self.commands = []  # 初始化 commands 属性为空列表
        self.max_env_time = max_env_time  # env交互时间
        self.env_start_time = time.time()  # 记录最开始服务时间
        self.available_tools_name = []

    def init_data(self, ds):
        self.ds = ds
        self.docker_image = self.ds["docker_image"]
        if self.sanbox_mode == "sdk":
            self.docker_image = self.docker_image.replace("rex-registry-vpc", "rex-registry")
        self.swebench_verified = "swebench" in self.docker_image
        self.repo_name = self.ds["repo"] if self.ds.get("repo") else self.ds["repo_name"]
        # self.parsed_commit = self.ds['parsed_commit'] if self.ds.get('parsed_commit') else self.ds['parsed_commit_content']
        self.run_tests_regression = self.ds.get("run_tests_regression", None)
        if self.swebench_verified:
            # also create a test spec for swebench verified dockers (useful for grading)
            self.test_spec = make_test_spec(self.ds)

    def read_file(self, rel_file_path: str) -> str:
        # alt_path = /root
        output, _ = self.runtime.run(f"cat {rel_file_path}")
        return output

    def reset(
        self,
        ds,
        clear_time: int = 60,  # min
        timeout: int = 180,
        max_execute_time: float = 400.0,
        max_execute_retry: int = 10,
        update_route_key_interval: int = 11,
    ) -> Dict[str, Any]:
        """
        Resets the environment and returns an initial observation.
        return:
            {
                - state: "success" or "error" # Attention
                - error_message: error_message if state is "error".
                - route_key: final route_key.
                - container_name: only one container_name.
                - retry_times: final retry_times.
                - session: final session.
            }
        """
        # init data: docker_image, swebench_verified, repo_name, parsed_commit, expected_json, run_tests_regression
        self.init_data(ds)

        # get http session
        self.route_key = self.ds.get("route_key", uuid.uuid4().hex)
        self.task_idx = self.ds.get("task_idx", None)
        self.session = self.runtime.start_session(route_key=self.route_key)

        # post for start container
        self.logger.info(
            f"[RepoEnv]开始重置环境 ... route_key: {self.route_key}, task_idx: {self.task_idx}, used_time: {round(time.time() - self.env_start_time,2)} (current_time: {round(time.time(),2)}, env_start_time: {round(self.env_start_time,2)}, max_env_time: {round(self.max_env_time,2)})"
        )
        if DEBUG:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} [RepoEnv]开始重置环境 ... route_key: {self.route_key}, task_idx: {self.task_idx}, used_time: {round(time.time() - self.env_start_time,2)} (current_time: {round(time.time(),2)}, env_start_time: {round(self.env_start_time,2)}, max_env_time: {round(self.max_env_time,2)})"
            )
        reset_info = self.runtime.start_container(
            session=self.session,
            route_key=self.route_key,
            docker_image=self.docker_image,
            clear_time=clear_time,  # attention
            timeout=timeout,
            max_execute_time=min(
                self.max_env_time - (time.time() - self.env_start_time), max_execute_time
            ),  # 最多执行时间
            max_execute_retry=max_execute_retry,
            update_route_key_interval=update_route_key_interval,
        )
        self.container_name = reset_info.get("container_name", None)
        self.route_key = reset_info.get("route_key", None)
        self.retry_times = reset_info.get("retry_times", 1)
        self.error_message = reset_info.get("error_message", None)
        self.state = reset_info.get("state", "error")
        self.session = reset_info.get("session", self.session)
        if DEBUG:
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [RepoEnv]成功重置环境, route_key: {self.route_key}, container_name: {self.container_name}, retry_times: {self.retry_times}, state: {self.state}, error_message: {self.error_message}, used_time: {time.time() - self.env_start_time} (current_time: {time.time()}, env_start_time: {self.env_start_time}), max_env_time: {self.max_env_time}'
            )
        self.logger.info(
            f" [RepoClient]成功重置环境, route_key: {self.route_key}, container_name: {self.container_name}, retry_times: {self.retry_times}, state: {self.state}, error_message: {self.error_message}, used_time: {time.time() - self.env_start_time} (current_time: {time.time()}, env_start_time: {self.env_start_time}), max_env_time: {self.max_env_time}"
        )

        # setup env
        if self.state != "error":
            self.setup_env()
        return reset_info

    def setup_env(self):
        self.runtime.setup_run(
            workdir=self.repo_path,
            session=self.session,
            container_name=self.container_name,
            max_execute_time=self.max_execute_time,
            max_execute_retry=self.max_execute_retry,
            timeout=self.timeout,
            route_key=self.route_key,
        )

        if self.swebench_verified:
            self.setup_env_swebench()
        else:
            self.setup_env_trainset()

    def setup_env_swebench(self):
        self.alt_path = "/"  # the run_test is in the "/" directory for swebench dockers
        self.logger.info(f"[RepoEnv]开始准备swebench的环境 ...")
        self._execute_command(f"chmod +x /run_tests.sh")
        self._execute_command(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
        self._execute_command(f"python -m pip install chardet -i https://mirrors.aliyun.com/pypi/simple/")
        self.logger.info(f"[RepoEnv]swebench的环境准备完成 ...")
        return "SUCCESS"

    def debug_export(self):
        print("当前环境变量PATH：")
        self.runtime.run("echo $PATH")
        print("\n重新export后的环境变量PATH：")
        # self.runtime.run("export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        self.runtime.run(
            "bash -c 'export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'"
        )
        self.runtime.run("echo $PATH")

    def setup_env_trainset(self):
        # 调试：检查 PATH 环境变量和文件状态
        # self._execute_command('echo "Current PATH: $PATH"')
        # self._execute_command("ls -la /usr/local/bin/")
        # self._execute_command("ls -la /root/")
        # 确保 /root/.local/bin 目录存在（为后续的符号链接做准备）
        # self._execute_command("mkdir -p /root/.local/bin")
        self.alt_path = "/root"

        self.logger.info(f"[RepoEnv]开始准备r2e trainset的环境 ...")
        self.logger.info(f"[RepoEnv][setup_env_trainset]repo_path: {self.repo_path}, alt_path: {self.alt_path}")

        # create a symlink from repo_path/.venv to /root/.venv
        # '/testbed' -> '/root'
        self._execute_command(f"ln -s /testbed/.venv /root/.venv")
        self._execute_command(f"ln -s /testbed/.venv/bin/python /root/.local/bin/python")
        self._execute_command(f"ln -s /testbed/.venv/bin/python /root/.local/bin/python3")

        # self.logger.info('检查/testbed/.venv/bin/中的文件 ...')
        # self._execute_command(f"ls -la /testbed/.venv/bin/")
        # install required packages
        # self._execute_command("uv pip install chardet -i https://mirrors.aliyun.com/pypi/simple/") # 默认安装了chardetect
        self._execute_command(f"ln -sf /testbed/.venv/bin/chardetect /usr/local/bin/chardet")

        # clean cache file. also delete pycache and pyc.
        self._execute_command(f"find /testbed/.venv/bin -type f -executable -exec ln -sf {{}} /root/.local/bin/ \\;")
        self._execute_command("find . -name '*.pyc' -delete")
        self._execute_command("find . -name '__pycache__' -exec rm -rf {} +")
        self._execute_command("find /testbed/tests -name '*.pyc' -delete")
        self._execute_command("find /testbed/tests -name '__pycache__' -exec rm -rf {} +")

        # move all skip files (if present) to /root
        # for skip_file in SKIP_FILES_NEW:
        # self._execute_command(f"scp -r /testbed/{skip_file} /root/{skip_file}")
        # r2e_tests are in the / directory, move them to /root
        self._execute_command("mv /r2e_tests /root/r2e_tests")
        self._execute_command("mv /testbed/run_tests.sh /root/run_tests.sh")
        self._execute_command("ln -s /root/run_tests.sh /testbed/run_tests.sh")
        self._execute_command(
            "sed -i 's|\.venv/bin/python|/testbed/.venv/bin/python|g' /root/run_tests.sh"
        )  # 修改为绝对路径
        self._execute_command("sed -i 's|r2e_tests|/root/r2e_tests|g' /root/run_tests.sh")  # 修改为绝对路径
        self._execute_command("chmod +x /root/run_tests.sh")

        # self.logger.info('检查当前位置 ...')
        # self._execute_command("pwd")
        # self.logger.info('检查/root/r2e_tests中的文件 ...')
        # self._execute_command("ls -a /root/r2e_tests/")
        # self.logger.info('检查/root/tests中的文件 ...')
        # self._execute_command("ls -a /root/tests/")
        # self.logger.info(f"检查/root/run_tests.sh中的文件 ... ")
        # self._execute_command('cat /root/run_tests.sh')
        # self.logger.info('检查chardet是否安装成功 ...')
        # self._execute_command("chardet --help")
        # self.logger.info('检查/root/run_tests.sh的运行 ...')
        # self._execute_command("/root/run_tests.sh")
        self.logger.info(f"[RepoEnv]r2e trainset的环境准备完成 ...")

        return "SUCCESS"

    def calculate_reward(self) -> int:
        """
        Basic reward calculation based on command success.
        """
        if self.swebench_verified:
            reward, output = self._calculate_reward_swebench()
        else:
            reward, output = self._calculate_reward_r2e()
        print(f"[RepoEnv]计算reward: {reward}, output: {[output]}")
        self.logger.info(f"[RepoEnv]计算reward: {reward}")
        return reward, output

    def _calculate_reward_r2e(self) -> int:
        self.logger.info(f"[RepoEnv]计算r2e trainset的reward start ...")
        self.expected_json = self.ds.get("expected_output_json", "")
        # self.read_file("/root/r2e_tests/expected_test_output.json"))
        # self.expected_json = json.loads(self.expected_json)
        # run_tests.sh
        output, _ = self.runtime.run(f"/root/run_tests.sh", timeout=300)
        # Remove ANSI escape codes and \r characters
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

        reward = self.swe_reward.calculate_reward_r2e(
            output=output,
            ds=self.ds,
            repo_name=self.repo_name,
            expected_json=self.expected_json,
            get_test_output=False,
            route_key=self.route_key,
        )
        self.logger.info(f"[RepoEnv]计算r2e trainset的reward: {reward}")
        return reward, output

    def _calculate_reward_swebench(self) -> int:
        # run_tests.sh
        self.logger.info(f"[RepoEnv]计算swebench的reward start ...")
        self._rm_conda_in_swebench()

        # 先初始化conda环境
        # self.logger.info("正在初始化conda环境...")
        # print('正在初始化conda环境... ')
        # self.runtime.run("conda init bash", timeout=300)
        # self.runtime.run("source ~/.bashrc", timeout=300)

        # 确保conda环境可用
        # self.runtime.run("eval \"$(conda shell.bash hook)\"", timeout=300)
        # self.runtime.run("conda activate testbed", timeout=300)
        # self.logger.info("conda环境初始化完成")
        # print('conda环境初始化完成')

        # out, _ = self.runtime.run("cat /run_tests.sh", timeout=1800)  # run the tests after applying the patch
        # out, _ = self.runtime.run("pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && /run_tests.sh", timeout=1800)  # run the tests after applying the patch
        out, _ = self.runtime.run("/run_tests.sh", timeout=1800)  # run the tests after applying the patch

        # parse eval logs & calculate reward
        reward = self.swe_reward.calculate_reward_swebench(self.ds, out, get_test_output=False)
        self.logger.info(f"[RepoEnv]计算swebench的reward: {reward}")
        return reward, out

    def get_system_prompt_in_iflow(self):
        """ """
        stdout, exit_code = self._execute_command("iflow --sysinfo 目录下有哪些文件")
        text = stdout
        try:
            stdout = stdout.split("You are iFlow CLI, ")[-1].split("query is completely resolved")[0]
            text = f"You are iFlow CLI, {stdout}query is completely resolved"
        except Exception as e:
            print(f"[RepoEnv][ERROR][get_system_prompt_in_iflow error]: {e}")
        return text

    def add_commands(self, cmd_files: list[str]):
        """
        Adds command files and tool names to the environment.

        Elements ending with '.py' are treated as file paths and parsed normally. copying them to the Docker container, and making them executable or sourced.
        Elements not ending with '.py' are treated as tool names and registered directly.

        Args:
            cmd_files: List of paths to command files or tool names.
        """
        from roll.pipeline.agentic.env.swe_env.util.define.commands import Command

        cmds = []
        for cmd_file in cmd_files:
            # Check if it's a tool name (not ending with .py)
            if not cmd_file.endswith(".py"):
                # Register as a tool name directly
                name = cmd_file.split("/")[-1]
                tool_cmd = Command(
                    name=name,
                    code=f"# Tool: {name}",
                    docstring=f"Tool command: {name}",
                    arguments=None,
                    signature=None,
                )
                cmds.append(tool_cmd)
                self.logger.info(f"Registered tool: {cmd_file}")
                continue

            # Process as a file path
            current_file_path = Path(__file__).resolve()
            cmd_file = cmd_file.replace("./", f"{current_file_path.parent.parent.parent.parent.parent}/")
            # Parse commands from file
            parsed_commands = self.cmd_parser.parse_command_file(cmd_file)
            cmds.extend(parsed_commands)
            # Process the command file
            self._process_command_file(cmd_file)

        # Add to existing commands
        self.commands = cmds  # name, code,
        # print(f'cmds: {cmds}')
        self.logger.info(f"Added {len(cmds)} commands to the environment.")

        # 为 r2e trainset 环境创建 execute_bash 的符号链接
        # if not self.swebench_verified:
        #     self._create_execute_bash_symlinks()

        self.available_tools_name = self.get_available_cmds()

    def get_available_cmds(self) -> list[str]:
        # print('[get_available_cmds]', [x.name for x in self.commands])
        return [x.name for x in self.commands]

    def _create_execute_bash_symlinks(self):
        """
        为 r2e trainset 环境创建 execute_bash 的符号链接
        """
        try:
            # command = f"export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

            # 检查 /usr/local/bin/execute_bash 是否存在
            self._execute_command("ls -la /usr/local/bin/execute_bash")

            # 创建符号链接到 /root/.local/bin/
            self._execute_command("ln -sf /usr/local/bin/execute_bash /root/.local/bin/execute_bash")

            # 验证符号链接是否创建成功
            self._execute_command("ls -la /root/.local/bin/execute_bash")

            # 测试 execute_bash 是否可以直接执行
            self._execute_command("/usr/local/bin/execute_bash --help || echo 'execute_bash not executable'")
            self._execute_command("/root/.local/bin/execute_bash --help || echo 'execute_bash symlink not executable'")
            self._execute_command("/root/.local/bin/execute_bash --cmd 'ls'")
            self._execute_command("/usr/local/bin/execute_bash --cmd 'ls'")

            #     command = f"export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && {command}"
            self.runtime.run(
                "bash -c 'export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'",
                timeout=self.timeout,
                max_execute_time=self.max_execute_time,
            )
            self.runtime.run("echo $PATH")

            self.logger.info("Successfully created execute_bash symlinks for r2e trainset environment")

        except Exception as e:
            self.logger.error(f"Failed to create execute_bash symlinks: {e}")

    def _process_command_file(self, cmd_file: str):
        """
        Process a single command file by copying it to the container and setting appropriate permissions.

        Args:
            cmd_file: Path to the command file to process.
        """
        # Determine the file extension and get base name
        _, ext = os.path.splitext(cmd_file)
        cmd_name = os.path.basename(cmd_file)

        # Determine container command name and path
        if ext == ".py" or self._is_shebang_script(cmd_file):
            # Python script or shebang script: strip .py extension if applicable
            container_cmd_name = cmd_name[:-3] if ext == ".py" else cmd_name
            container_path = f"/usr/local/bin/{container_cmd_name}"
            self.runtime.copy_to_container(cmd_file, container_path)
            self._execute_command(f"chmod +x {container_path}")
            print(f"[RepoEnv][_process_command_file]copied {cmd_file} to {container_path}")
        else:
            # Bash script: copy, chmod, and source it
            container_cmd_name = cmd_name
            container_path = f"/usr/local/bin/{container_cmd_name}"
            self.runtime.copy_to_container(cmd_file, container_path)
            self._execute_command(f"chmod +x {container_path}")
            # Source the script inside the container
            self.runtime.run(f"bash -c 'source {container_path}'")
        # self._execute_command(f"cat /usr/local/bin/execute_bash")

    def _is_shebang_script(self, cmd_file: str) -> bool:
        """
        Checks if the given file starts with a shebang (#!).

        Args:
            cmd_file: Path to the command file.

        Returns:
            True if the file starts with a shebang, False otherwise.
        """
        with open(cmd_file, "r") as file:
            first_line = file.readline().strip()
        return first_line.startswith("#!")

    def _should_use_nohup(self, command: str) -> bool:
        """
        判断命令是否需要使用 nohup 执行

        Args:
            command: 要执行的命令

        Returns:
            True if the command should use nohup, False otherwise
        """
        # 排除不应该使用 nohup 的命令
        exclude_patterns = [
            r"^export\s+",  # export 命令
            r"^mkdir\s+",  # mkdir 命令
            r"^chmod\s+",  # chmod 命令
            r"^ln\s+-s",  # 符号链接命令
            r"^mv\s+",  # 移动文件命令
            r"^echo\s+",  # echo 命令
            r"^bash\s+-c",  # bash -c 命令
        ]

        command_lower = command.lower().strip()
        for pattern in exclude_patterns:
            if re.search(pattern, command_lower):
                return False

        # 需要 nohup 的命令模式
        nohup_patterns = [
            r"pip\s+install",
            r"uv\s+pip\s+install",
            r"python\s+-m\s+pip\s+install",
            r"poetry\s+install",
            r"pipenv\s+install",
            r"setup\.py\s+install",
            r"python\s+setup\.py",
            r"conda\s+activate",
            r"source\s+.*activate",
        ]

        command_lower = command.lower().strip()
        for pattern in nohup_patterns:
            if re.search(pattern, command_lower):
                return True
        return False

    def source_export(self):
        """
        读取 bashrc，增加 export PATH 设置，然后 source bashrc
        """
        print("[RepoEnv]source_export ============")
        try:
            # 读取当前的 ~/.bashrc 文件
            out_file, _ = self.runtime.run("cat ~/.bashrc", 300)
            if not out_file:
                self.logger.error("[ERROR]无法读取 ~/.bashrc 文件")
                return False

            # 定义要添加的 PATH 导出语句
            path_export = "export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

            lines = out_file.split("\n")
            modified_lines = []
            path_already_exists = False

            # 检查是否已经存在相同的 PATH 导出
            for line in lines:
                if path_export in line:
                    path_already_exists = True
                    self.logger.info("PATH 导出语句已存在，跳过添加")
                    break

            if not path_already_exists:
                # 添加 PATH 导出语句到文件末尾
                modified_lines = lines + ["", "# Added by source_export function", path_export]

                # 创建修改后的内容
                modified_content = "\n".join(modified_lines)

                # 写入文件
                escaped_content = modified_content.replace('"', '\\"').replace("$", "\\$")
                self.runtime.run(f'echo "{escaped_content}" > ~/.bashrc', timeout=300)

                self.logger.info("成功添加 PATH 导出语句到 ~/.bashrc")
            else:
                modified_content = out_file

            # source bashrc 使更改生效
            self.runtime.run("bash -c 'source ~/.bashrc'", timeout=300)
            self.logger.info("成功执行 source ~/.bashrc")

            self.logger.info("检查是否写入成功 ~/.bashrc")
            self.runtime.run("bash -c 'cat ~/.bashrc'", timeout=300)

            # 验证 PATH 是否设置成功
            path_output, _ = self.runtime.run("echo $PATH", timeout=300)
            self.logger.info(f"当前 PATH: {path_output}")
            print("[RepoEnv]source_export end ============")
            return True

        except Exception as e:
            self.logger.error(f"[ERROR]source_export 执行时出错: {repr(e)}")
            return False

    def extract_pytest_results(self, output: str) -> dict:
        """
        从 pytest 输出中提取测试结果信息

        Args:
            output: pytest 的完整输出文本

        Returns:
            dict: 包含 passed_num, error_num, warning_num 的字典
        """
        import re

        # 初始化结果字典
        result = {"passed_num": 0, "error_num": 0, "warning_num": 0}

        try:
            # 匹配 pytest 结果行的模式
            # 例如: "======================= 1 passed, 111 warnings in 0.03s ======================="
            pattern = r"=+\s*(\d+)\s+passed(?:,\s*(\d+)\s+errors?)?(?:,\s*(\d+)\s+warnings?)?\s+in\s+[\d.]+\s*=+"
            match = re.search(pattern, output)

            if match:
                result["passed_num"] = int(match.group(1))
                if match.group(2):
                    result["error_num"] = int(match.group(2))
                if match.group(3):
                    result["warning_num"] = int(match.group(3))
            else:
                # 如果没有匹配到完整模式，尝试分别匹配各个部分
                # 匹配 passed 数量
                passed_match = re.search(r"(\d+)\s+passed", output)
                if passed_match:
                    result["passed_num"] = int(passed_match.group(1))

                # 匹配 error 数量
                error_match = re.search(r"(\d+)\s+errors?", output)
                if error_match:
                    result["error_num"] = int(error_match.group(1))

                # 匹配 warning 数量
                warning_match = re.search(r"(\d+)\s+warnings?", output)
                if warning_match:
                    result["warning_num"] = int(warning_match.group(1))

            self.logger.info(f"提取的测试结果: {result}")
            return result

        except Exception as e:
            self.logger.error(f"提取 pytest 结果时出错: {repr(e)}")
            return result

    def _execute_command(self, command: str, timeout: int = None) -> tuple[str, str]:
        """
        执行命令，根据命令类型决定是否使用 nohup

        Args:
            command: 要执行的命令
            timeout: 超时时间

        Returns:
            (stdout, exit_code) 元组
        """
        if timeout is None:
            timeout = self.timeout

        return_stdout, return_code = "Error: Timeout", "-1"
        if time.time() - self.env_start_time > self.max_env_time:  # 超过了整个环境的交互时间
            return return_stdout, return_code
        max_execute_time = min(
            self.max_env_time - (time.time() - self.env_start_time), self.max_execute_time
        )  # 单步执行时间

        # 为 r2e trainset环境设置执行前缀
        self.logger.info(f"[RepoEnv]self.available_tools_name: {self.available_tools_name}")
        if command.split(" ")[0] in self.available_tools_name and not self.swebench_verified:
            command = "/usr/local/bin/" + command

        # 检查是否是 execute_bash 命令且包含 cd
        if command.startswith("execute_bash") and "cd " in command:
            # 提取 --cmd 参数中的实际命令
            import re

            cmd_match = re.search(r"--cmd\s+'([^']+)'", command)
            if cmd_match:
                actual_cmd = cmd_match.group(1)
                # 直接使用 bash -c 执行包含 cd 的命令，避免 execute_bash 工具的限制
                bash_command = f"bash -c '{actual_cmd}'"
                self.logger.info(f"[针对cd的命令做了转换]之前:{command}, 之后:{bash_command}")
                return self.runtime.run(bash_command, timeout=timeout, max_execute_time=max_execute_time)

        # # 为 r2e trainset 环境设置正确的 PATH
        # if not self.swebench_verified and command.startswith('execute_bash'):
        #     # 使用 export 设置 PATH 环境变量
        #     command = f"export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && {command}"

        # # 转换包含 cd 的命令为 bash -c 形式
        # command = self._convert_cd_command(command)

        if self._should_use_nohup(command):
            # 使用 nohup 执行命令，设置较短的超时时间
            nohup_command = f"nohup {command} > /dev/null 2>&1 &"
            # nohup 命令会立即返回，所以使用较短的超时时间
            nohup_timeout = min(30, timeout) if timeout else 30
            return self.runtime.run(nohup_command, timeout=nohup_timeout, max_execute_time=max_execute_time)
        else:
            # 直接执行命令
            return self.runtime.run(command, timeout=timeout, max_execute_time=max_execute_time)

    # def _convert_cd_command(self, command: str) -> str:
    # """
    # 将包含 cd 的命令转换为 bash -c 形式，避免 execute_bash 工具的限制
    #     cd /testbed && python reproduce_issue.py → bash -c 'cd /testbed && python reproduce_issue.py'
    #     cd /testbed → bash -c 'cd /testbed'
    # Args:
    #     command: 原始命令
    # Returns:
    #     转换后的命令
    # """
    # # 检查是否包含 cd 命令
    # if 'cd ' in command and ('&&' in command or ';' in command or '|' in command):
    #     # 将命令包装在 bash -c 中，这样 cd 命令可以正常工作
    #     # 使用单引号避免参数解析问题
    #     return f"bash -c '{command}'"
    # elif command.strip().startswith('cd '):
    #     # 单独的 cd 命令也转换为 bash -c
    #     return f"bash -c '{command}'"
    # else:
    #     # 不包含 cd 的命令直接返回
    #     return command

    def run_action(self, action: Action, timeout: int, base_agent: str = "swe"):
        """
        @input:
            - action: 模型的输入
        @output:
            - bash_output, exit_code: 环境执行反馈
        """
        start_time = time.time()
        bash_cmd, exit_code, bash_output = action.function_name, "-100", ""

        # if base_agent == 'swe':
        allowed_cmds = [x.name for x in self.commands]
        # if base_agent == 'iflow':
        # allowed_cmds = allowed_cmds + ['list_directory','read_file','write_file','replace','multi_edit','search_file_content','web_search','todo_write','todo_read','glob','run_shell_command','web_fetch']
        # check for empty or no function call / action
        if not action.function_name:
            bash_output = (
                f"Invalid Action: input action must be one of allowed actions \n Allowed actions: {allowed_cmds}\n."
            )
            exit_code = "-100"
        # Check if action is in allowed actions/commands
        elif action.function_name not in allowed_cmds:
            bash_output = f"Invalid Action: input action must be one of allowed actions \n Allowed actions: {allowed_cmds}\n. Input action: {action.function_name}\t"
            exit_code = "-100"
        # Run action and return
        elif base_agent == "swe":
            bash_cmd = action.to_bashcmd()
            bash_output, exit_code = self._execute_command(bash_cmd, timeout=timeout)
        elif base_agent == "iflow":
            bash_cmd = action.to_iflowcmd(tool_id=self.route_key + "_" + str(time.time()))
            print(f"[run_action: iflow]bash_cmd: {bash_cmd}")
            bash_output, exit_code = self._execute_command(bash_cmd, timeout=timeout)
            try:
                bash_output_json = json.loads(bash_output)
                print(f"[run_action: iflow]bash_output_json: {bash_output_json}")
                self.logger.info(f"[run_action: iflow]bash_output_json: {bash_output_json}")
                bash_output = bash_output_json["tool_results"][0]["full_response"]
            except Exception as e:
                print(f"[ERROR][RepoEnv][run_action error]: {e}")
                self.logger.error(f"[ERROR][RepoEnv][run_action error]: {e}")
        self.logger.info(
            f"[RepoEnv][执行命令]{[bash_cmd]}\n执行反馈:exit_code: {exit_code}\nbash_output: {bash_output}"
        )
        print(
            f"[RepoEnv][执行命令]{[bash_cmd]}\n[RepoEnv][执行反馈]exit_code: {exit_code}\nbash_output: {[bash_output]}"
        )
        return bash_output, exit_code, time.time() - start_time

    def step(self, action: Action, timeout: int) -> Tuple[Observation, int, bool, Dict[str, Any]]:
        """
        Executes an action (command) in the Docker container.
        Runs an action proposed by the agent in the environment and returns the corresponding output.

        Args:
            action: command to run in bash shell

        Returns:
            observation:  output from container
            reward: Always set to 0
            done: whether task is over
            info: additional information (e.g. debugging information)
        """
        reward = 0
        bash_output, error_code, total_time = self.run_action(action, timeout=timeout)
        self.observation = Observation(bash_output, error_code, action)
        if "finish" in action.function_name.lower() or self.done:
            self.done = True
        info = {"total_time": total_time}
        return self.observation, reward, self.done, info

    def close(self):
        try:
            self.runtime.stop_container(route_key=self.route_key, container_name=self.container_name)
        except Exception as e:
            self.logger.warning(f"关闭容器时出错: {e}")

    def _rm_conda_in_swebench(self):
        """
        注释掉 /run_tests.sh 中与环境创建相关的命令
        包括 conda activate、source activate、pip install 等
        """
        try:
            # 读取当前的 /run_tests.sh 文件
            out_file, _ = self.runtime.run("cat /run_tests.sh", 300)
            if not out_file:
                self.logger.error("[ERROR]无法读取 /run_tests.sh 文件")
                return False

            # 定义需要注释的环境相关命令
            env_commands_to_comment = [
                # conda 相关
                r"^conda activate",
                r"^source /opt/miniconda3/bin/activate",
                r"^source.*activate",
                # pip install 相关
                r"^python -m pip install",
                r"^pip install",
                r"^uv pip install",
                r"^poetry install",
                r"^pipenv install",
                # setup.py install 相关
                r"^python setup\.py install",
                r"^setup\.py install",
                r"^python.*setup\.py.*install",
                # 错误命令
                r"^cat: 300: No such file or directory",
            ]

            lines = out_file.split("\n")
            modified_lines = []
            commented_count = 0

            for line in lines:
                should_comment = False

                # 检查是否需要注释
                for pattern in env_commands_to_comment:
                    if re.search(pattern, line.strip()):
                        should_comment = True
                        break

                if should_comment and line.strip() and not line.strip().startswith("#"):
                    # 添加注释符号
                    modified_lines.append(f"# {line}")
                    commented_count += 1
                    # self.logger.info(f"注释环境命令: {line.strip()}")
                else:
                    modified_lines.append(line)

            modified_lines.insert(1, "export GIT_PAGER=cat")
            # 写回文件
            modified_content = "\n".join(modified_lines)
            # 创建临时文件
            temp_content = f"""{modified_content}"""
            # if self.sanbox_mode=='sdk':
            # 使用 cat 和 heredoc 语法避免 bash 历史扩展问题
            # self.runtime.run(f"cat > /run_tests.sh << 'EOF'\n{temp_content}\nEOF", timeout=300)
            # else:
            self.runtime.create_file(file_path="/tmp/run_tests_modified.sh", content=temp_content)
            self.runtime.run("cp /tmp/run_tests_modified.sh /run_tests.sh", timeout=300)
            self._execute_command("chmod +x /run_tests.sh", timeout=300)
            self.logger.info(f"成功注释环境中的耗时命令，耗时命令数量: {commented_count}, 当前的单测文件：\n")
            out_file, _ = self.runtime.run("cat /run_tests.sh", timeout=300)
            return True

        except Exception as e:
            self.logger.info(f"[ERROR]注释环境命令时出错: {repr(e)}")
            return False


if __name__ == "__main__":

    # content = load_data_txt('/home/lixing/workspace/future_update/ROLL_version/ScaleAligner/roll/agentic/env/swe_env/util/tools/execute_bash.py')
    # print([content])
    # exit()
    repo_env = RepoClient(swe_rex_host="https://xrl-aliyun.alibaba-inc.com/swe-rex/docker", sanbox_mode="sdk")
    from tests.agentic.sweenv.utils.utils_file import load_data_json

    # ds=load_data_json('/home/lixing/workspace/future_update/ROLL_version/ScaleAligner/data/swe_verified_iflow.json')
    ds = load_data_json("/home/lixing/workspace/future_update/ROLL_version/ScaleAligner/data/swe_trainset_0.json")
    # ds=load_data_json('/home/lixing/workspace/future_update/ROLL_version/ScaleAligner/data/swe_verified.json')
    # reset docker & setup env
    reset_info = repo_env.reset(ds=ds, max_execute_time=180, max_execute_retry=10, timeout=180)
    # repo_env.debug_export()

    current_file_path = Path(__file__).resolve()
    tools = [
        "/home/lixing/workspace/future_update/ROLL_version/ScaleAligner/roll/agentic/env/swe_env/util/tools/execute_bash.py",
        "/home/lixing/workspace/future_update/ROLL_version/ScaleAligner/roll/agentic/env/swe_env/util/tools/str_replace_editor.py",
        "/home/lixing/workspace/future_update/ROLL_version/ScaleAligner/roll/agentic/env/swe_env/util/tools/submit.py",
    ]
    # tools = ["list_directory","read_file","write_file","replace","multi_edit","search_file_content","glob","web_search","web_fetch","todo_write","todo_read","run_shell_command"]
    # tools = [f'{current_file_path.parent.parent}/{data_path}' for data_path in tools]
    repo_env.add_commands(tools)
    print(f"[DEBUG]repo_env.commands: {repo_env.commands}")

    repo_env.runtime.create_file(
        file_path="/home/lixing/linxin/ScaleAligner/roll/agentic/env/swe_env/util/sanbox_server_sdk.py",
        # file_path="/tmp/test.txt",
        content="test hello word",
        session=reset_info["session"],
        container_name=reset_info["container_name"],
    )
    # repo_env.runtime.stop_container(route_key=reset_info['route_key'], container_name=reset_info['container_name'])

    reward, output = repo_env.calculate_reward()
    print(f"[DEBUG]reward: {reward}")

    repo_env.close()
    print(f"[DEBUG]repo_env.closed")
