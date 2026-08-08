import re
from typing import Dict
import shlex


class Action:
    """
    Represents an action with:
      - function_name (e.g. 'file_editor')
      - parameters    (a dictionary of parameter_name -> value)

    Provides methods:
      - from_string(...) -> create Action from XML-like string
      - to_dict()        -> returns a JSON-like dict
      - to_bashcmd()     -> returns a string representing an equivalent bash command
    """

    def __init__(self, function_name: str, parameters: Dict[str, str], function_id: str = None):
        self.function_name = function_name
        self.parameters = parameters
        # self.function_id = function_id

    @classmethod
    def from_string(cls, action_str: str) -> "Action":
        """
        Parses a string of the form:

          <function=FUNCTION_NAME>
            <parameter=KEY>VALUE</parameter>
            ...
          </function>

        and returns an Action object.

        For example:
          <function=file_editor>
            <parameter=command>view</parameter>
            <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
            <parameter=concise>True</parameter>
          </function>

        yields an Action with:
          function_name = "file_editor"
          parameters = {
            "command":  "view",
            "path":     "./sympy/tensor/array/dense_ndim_array.py",
            "concise":  "True"
          }
        """
        # Extract the function name: <function=...>
        fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
        function_name = fn_match.group(1).strip() if fn_match else ""

        # Extract parameters of the form: <parameter=KEY>VALUE</parameter>
        # DOTALL allows the captured VALUE to span multiple lines
        pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
        param_matches = re.findall(pattern, action_str, flags=re.DOTALL)

        params = {}
        for param_key, param_value in param_matches:
            param_key = param_key.strip()
            param_value = param_value.strip()
            params[param_key] = param_value

        return cls(function_name, params)

    def __str__(self) -> str:
        return self.to_xml_string()

    def to_xml_string(self) -> str:
        """
        Returns an XML-like string representation of this action.

        Example:
          <function=file_editor>
            <parameter=command>view</parameter>
            <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
            <parameter=concise>True</parameter>
          </function>
        """
        # Start with the function name
        xml_str = f"<function={self.function_name}>\n"

        # Add each parameter as <parameter=KEY>VALUE</parameter>
        for param_key, param_value in self.parameters.items():
            xml_str += f"  <parameter={param_key}>{param_value}</parameter>\n"

        xml_str += "</function>"
        return xml_str

    def to_dict(self) -> Dict[str, object]:
        """
        Returns a JSON-like dictionary representation of this action.

        Example:
          {
            "function": "file_editor",
            "parameters": {
              "command": "view",
              "path": "./sympy/tensor/array/dense_ndim_array.py",
              "concise": "True"
            }
          }
        """
        return {"function": self.function_name, "parameters": self.parameters}

    def to_iflowcmd(self, tool_id: str = None) -> str:
        """
        Converts this action into a iflow command string.

        Args:
            tool_id: Optional tool call ID. If not provided, will use function_name + timestamp

        Returns:
            iflow command string in the format: iflow -t '{"tool_calls":[...]}'
        """
        import json
        import time

        if tool_id is None:
            tool_id = f"{self.function_name}_{int(time.time())}"

        # Create tool call structure
        tool_call = {
            "id": tool_id,
            "function": {"name": self.function_name, "arguments": json.dumps(self.parameters)},
            "type": "function",
        }

        # Create the complete tool calls structure
        tool_calls = {"tool_calls": [tool_call]}

        # Convert to JSON string
        json_str = json.dumps(tool_calls)

        # Return iflow command with proper escaping
        command = f"iflow -t '{json_str}'"
        print(f"[iflow command转换后]{command}")
        return command

    def quote_bash_command(self, cmd: str) -> str:
        """
        Safely quote a bash command string with proper handling of mixed quotes.
        """
        if not cmd:
            return "''"

        # 对于包含复杂引号的命令，使用双引号包围并转义内部双引号
        if '"' in cmd and "'" in cmd:
            # 混合引号：转义双引号，用双引号包围
            escaped_cmd = cmd.replace('"', '\\"')
            return f'"{escaped_cmd}"'
        elif '"' in cmd:
            # 只有双引号：用单引号包围
            return f"'{cmd}'"
        elif "'" in cmd:
            # 只有单引号：用双引号包围
            return f'"{cmd}"'
        else:
            # 没有引号：用单引号包围
            return f"'{cmd}'"

    def to_bashcmd(self) -> str:
        """
        Converts this action into a Bash command string with proper quoting.
        Handles nested quotes safely for execute_bash.
        """
        if not self.function_name:
            return ""
        elif self.function_name == "finish":
            return "echo '<<<Finished>>>'"

        # Start building the command
        cmd_parts = [shlex.quote(self.function_name)]

        if self.function_name == "execute_bash":
            base_command = self.parameters.get("command")
            if base_command is not None:
                cmd_parts.append("--cmd")
                cmd_parts.append(self.quote_bash_command(base_command))

            # 处理其它参数
            for param_key, param_value in self.parameters.items():
                if param_key == "command":
                    continue
                cmd_parts.append(f"--{param_key}")
                cmd_parts.append(shlex.quote(str(param_value)))
        else:
            # 通用逻辑
            base_command = self.parameters.get("command")
            if base_command is not None:
                cmd_parts.append(shlex.quote(base_command))

            for param_key, param_value in self.parameters.items():
                if param_key == "command":
                    continue
                cmd_parts.append(f"--{param_key}")
                cmd_parts.append(shlex.quote(str(param_value)))

        return " ".join(cmd_parts)


def test_to_bashcmd():
    cases = [
        # 最简单的命令
        ("execute_bash", {"command": "ls -l"}),
        # 带双引号的命令
        ("execute_bash", {"command": 'python -c "print(\\"hello world\\")"'}),
        # 带单引号的命令
        ("execute_bash", {"command": "echo 'hello'"}),
        # 混合单双引号
        ("execute_bash", {"command": "python -c \"print('hello')\""}),
        # 带额外参数
        ("execute_bash", {"command": "ls", "path": "/tmp/test dir"}),
        # 非 execute_bash 的命令
        ("file_editor", {"command": "view", "path": "./main.py", "concise": True}),
        # finish 特殊情况
        ("finish", {}),
        # 边缘 cases - 复杂引号处理
        # 只有单引号的简单命令
        ("execute_bash", {"command": "python -c \"print('Test with single quotes: hello')\""}),
        # 混合引号的复杂命令
        ("execute_bash", {"command": 'python -c "print(\'Test with mixed quotes: \\"hello\\" and \\"world\\"\')"'}),
        # 包含 cd 和复杂引号的命令
        (
            "execute_bash",
            {
                "command": "cd /home/lixing/workspace/future_update/ROLL_version/ScaleAligner && python -c \"print('Hello from Python!')\""
            },
        ),
        # 原始问题命令 - astropy 相关
        (
            "execute_bash",
            {
                "command": "cd /testbed && python -c \"from astropy.modeling.models import Pix2Sky_TAN, Linear1D; print('Pix2Sky_TAN:', Pix2Sky_TAN().n_inputs, Pix2Sky_TAN().n_outputs); print('Linear1D:', Linear1D(10).n_inputs, Linear1D(10).n_outputs)\""
            },
        ),
        # 包含特殊字符的命令
        ("execute_bash", {"command": 'grep -n "separable" /testbed/astropy/modeling/models.py'}),
        # 空命令
        ("execute_bash", {"command": ""}),
        # 只有空格的命令
        ("execute_bash", {"command": "   "}),
    ]

    # Case 1: {'command': 'ls -l'}
    # execute_bash --cmd 'ls -l'

    # Case 2: {'command': 'python -c "print(\\"hello world\\")"'}
    # execute_bash --cmd 'python -c "print(\\"hello world\\")"'

    # Case 3: {'command': "echo 'hello'"}
    # execute_bash --cmd 'echo '"'"'hello'"'"''

    # Case 4: {'command': 'python -c "print(\'hello\')"'}
    # execute_bash --cmd 'python -c "print('"'"'hello'"'"')"'

    # Case 5: {'command': 'ls', 'path': '/tmp/test dir'}
    # execute_bash --cmd 'ls' --path '/tmp/test dir'

    # Case 6: {'command': 'view', 'path': './main.py', 'concise': True}
    # file_editor view --path './main.py' --concise 'True'

    # Case 7: {}
    # echo '<<<Finished>>>'
    for i, (fname, params) in enumerate(cases, 1):
        action = Action(fname, params)
        print(f"Case {i}: {params}")
        print(action.to_bashcmd())
        print("-" * 80)


def test_edge_cases():
    """
    专门测试边缘 cases 的函数
    """
    print("=" * 80)
    print("测试边缘 cases")
    print("=" * 80)

    edge_cases = [
        # 空命令和空白命令
        ("execute_bash", {"command": ""}),
        ("execute_bash", {"command": "   "}),
        # 复杂引号组合
        ("execute_bash", {"command": "python -c \"print('Test with single quotes: hello')\""}),
        ("execute_bash", {"command": 'python -c "print(\'Test with mixed quotes: \\"hello\\" and \\"world\\"\')"'}),
        # 包含 cd 的复杂命令
        (
            "execute_bash",
            {
                "command": "cd /home/lixing/workspace/future_update/ROLL_version/ScaleAligner && python -c \"print('Hello from Python!')\""
            },
        ),
        # 原始问题命令
        (
            "execute_bash",
            {
                "command": "cd /testbed && python -c \"from astropy.modeling.models import Pix2Sky_TAN, Linear1D; print('Pix2Sky_TAN:', Pix2Sky_TAN().n_inputs, Pix2Sky_TAN().n_outputs); print('Linear1D:', Linear1D(10).n_inputs, Linear1D(10).n_outputs)\""
            },
        ),
        # 特殊字符命令
        ("execute_bash", {"command": 'grep -n "separable" /testbed/astropy/modeling/models.py'}),
        # 包含特殊符号的命令
        ("execute_bash", {"command": "find . -name '*.py' -exec grep -l 'import' {} \\;"}),
        # 包含管道和重定向的命令
        ("execute_bash", {"command": "ls -la | grep 'test' > /tmp/output.txt"}),
    ]

    for i, (fname, params) in enumerate(edge_cases, 1):
        action = Action(fname, params)
        print(f"Edge Case {i}: {params}")
        print(f"转换结果: {action.to_bashcmd()}")
        print("-" * 80)


if __name__ == "__main__":
    # Sample usage
    test_to_bashcmd()

    # 测试边缘 cases
    test_edge_cases()

    # Example 1
    xml_1 = """
    <function=file_editor>
      <parameter=command>view</parameter>
      <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
      <parameter=concise>True</parameter>
    </function>
    """
    action1 = Action.from_string(xml_1)
    print("[Example 1] Action as dict:", action1.to_dict())
    print("[Example 1] Action as bashcmd:", action1.to_bashcmd(), "\n")

    # Example 2
    xml_2 = """
    <function>execute_bash>
      <parameter=command>search_dir</parameter>
      <parameter=search_term>class ImmutableDenseNDimArray</parameter>
    </function>
    """
    action2 = Action.from_string(xml_2)
    print("[Example 2] Action as dict:", action2.to_dict())
    print("[Example 2] Action as bashcmd:", action2.to_bashcmd())

    # xml_3 = """<function=search>\n  <parameter=search_term>return_future\n  <parameter=path>/testbed\n</function>"""
    xml_3 = """<function=search>
    <parameter=search_term>return_future</parameter>
    <parameter=path>/testbed</parameter>
    </function>"""
    action3 = Action.from_string(xml_3)
    print("\n\n[Example 3] Action as dict:", action3.to_dict())
    print("[Example 3] Action as bashcmd:", action3.to_bashcmd())

    xml_4 = """<function=execute_bash>
  <parameter=command>grep -n "separable" /testbed/astropy/modeling/models.py</parameter>
</function>"""
    action4 = Action.from_string(xml_4)
    print("[Example 4] Action as dict:", action4.to_dict())
    print("[Example 4] Action as bashcmd:", action4.to_bashcmd())

    xml_5 = """<function=execute_bash>
  <parameter=command>cd /testbed && python -c "from astropy.modeling import models as m; print('Linear1D separable:', m.Linear1D(10).separable); print('Pix2Sky_TAN separable:', m.Pix2Sky_TAN().separable)"</parameter>
</function>
"""
    action5 = Action.from_string(xml_5)
    print("[Example 5] Action as dict:", action5.to_dict())
    print("[Example 5] Action as bashcmd:", action5.to_bashcmd())
