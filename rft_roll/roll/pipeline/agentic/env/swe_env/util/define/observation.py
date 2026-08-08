import re
from typing import Dict
from roll.pipeline.agentic.env.swe_env.util.define.action import Action
from roll.pipeline.agentic.env.swe_env.util.spec.config import CONTINUE_MSG


class Observation:
    def __init__(self, bash_output, error_code, action: Action, num_lines: int = 40):
        self.bash_output = bash_output
        self.error_code = error_code
        self.action = action
        self.num_lines = num_lines

    def __str__(self):
        # empty or no function call
        if not self.action.function_name:
            return CONTINUE_MSG
        elif self.action.function_name == "finish":
            return "<<< Finished >>>"
        else:
            if self.action.function_name == "execute_bash":
                if self.error_code == "-1":
                    output = f"Error occurred: {self.bash_output}"
                else:
                    lines = self.bash_output.splitlines() if self.bash_output else []
                    if len(lines) > 2 * self.num_lines:
                        top_lines = "\n".join(lines[: self.num_lines])
                        bottom_lines = "\n".join(lines[-self.num_lines :])
                        divider = "-" * 50
                        truncated_output = (
                            f"{top_lines}\n"
                            f"{divider}\n"
                            f"<Observation truncated in middle for saving context>\n"
                            f"{divider}\n"
                            f"{bottom_lines}"
                        )
                    else:
                        truncated_output = self.bash_output
                    output = (
                        f"Exit code: {self.error_code}\n"
                        f"Execution output of [{self.action.function_name}]:\n"
                        f"{truncated_output}"
                    )
                # output = f"Exit code: {self.error_code}\nExecution output of [{self.action.function_name}]:\n{self.bash_output}"
            else:
                output = f"Execution output of [{self.action.function_name}]:\n{self.bash_output}"

            # print(f'\n\n[Observation] error_code: {self.error_code}, output: {output}')
            return output
