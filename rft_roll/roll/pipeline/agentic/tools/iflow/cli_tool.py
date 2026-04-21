import ast
import json
import os
import re
import time
import shlex
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

from roll.utils.logging import get_logger

class IFlowCLITool:
    """
    A tool class that provides iflow-cli specific functionality for parsing model responses,
    executing tool calls, and processing tool responses.
    """
    def __init__(self, model_name: str, api_key: str, base_url: str, auth_type: str, 
                 search_api_key: str, debug: bool = False, logger: Optional[Any] = None):
        """Initialize the IFlowCLITool"""
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.auth_type = auth_type
        self.search_api_key = search_api_key
        self.debug = debug
        self.logger = logger
        self.tools = self.get_tools()

    def _parse_action(self, response: str):
        """
        Parse the model response to extract tool calls.
        """
        self.logger.info(f"[ACTION_PARSE] START - Parsing tool calls from response")
        
        try:
            actions = []
            if "<function" in response:
                action_pattern = r'<function[^>]*>.*?</function>'
                action_matches = re.findall(action_pattern, response, re.DOTALL)
                
                for i, action_str in enumerate(action_matches):
                    try:
                        fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
                        function_name = fn_match.group(1).strip() if fn_match else ""
                        pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
                        param_matches = re.findall(pattern, action_str, flags=re.DOTALL)

                        params = {}
                        for param_key, param_value in param_matches:
                            param_key = param_key.strip()
                            param_value = param_value.strip()
                            params[param_key] = param_value
                        cur = {
                            "type": "function",
                            "id": f"{function_name}_{int(time.time())}_{i}",
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps(params, ensure_ascii=False)
                            }
                        }
                        actions.append(cur)
                        self.logger.debug(f"[ACTION_PARSE] Parsed function action {i+1}: {function_name}")
                    except Exception as e:
                        self.logger.warning(f"[ACTION_PARSE] Failed to parse function action {i+1}: {e}")
                        continue
            elif "<tool_call>" in response:
                if "<tool_call>" in response and "</tool_call>" not in response:
                    response = response + "</tool_call>"
                tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
                tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)
                
                for i, tool_call_str in enumerate(tool_call_matches):
                    try:
                        tool_call_str = tool_call_str.strip()
                        tool_call_json = json.loads(tool_call_str)
                        function_name = tool_call_json.get("name", "")
                        arguments = tool_call_json.get("arguments", {})
                        cur = {
                            "type": "function",
                            "id": f"{function_name}_{int(time.time())}_{i}",
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False)
                            }
                        }
                        actions.append(cur)
                        self.logger.debug(f"[ACTION_PARSE] Parsed tool_call action {i+1}: {function_name}")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"[ACTION_PARSE] Failed to parse JSON in tool_call {i+1}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"[ACTION_PARSE] Failed to parse tool_call action {i+1}: {e}")
                        continue
            if actions:
                self.logger.info(f"[ACTION_PARSE] Success! - Parsed {len(actions)} tool calls")
            else:
                self.logger.info(f"[ACTION_PARSE] No tool calls found in response")
            
            return True, actions
            
        except Exception as e:
            self.logger.error(f"[ACTION_PARSE] Failed! - Error parsing action: {e}")
            return False, []

    def execute_action(self, response: str, session_name: str) -> Tuple[bool, bool, str]:
        """
        Execute tool calls from the model response using iflow-cli.
        """
        self.logger.info(f"[ACTION_EXECUTE] START - Executing tool calls")
        
        is_parsed, actions = self._parse_action(response)
        
        if not is_parsed:
            self.logger.error(f"[ACTION_EXECUTE] Failed! - Could not parse actions from response")
            return False, ""
        
        if not actions:
            self.logger.info(f"[ACTION_EXECUTE] No actions to execute")
            return True, ""
        
        try:
            tool_calls = {"tool_calls": actions}
            tool_calls = json.dumps(tool_calls, ensure_ascii=False)
            tool_calls = shlex.quote(tool_calls)
            iflow_cmd = f"iflow -t {tool_calls}"
            self.logger.info(f"[ACTION_EXECUTE] Success! - Generated iflow command with {len(actions)} tool calls")
            return True, iflow_cmd
        except Exception as e:
            error_msg = f"Error executing tool calls: {e}"
            self.logger.error(f"[ACTION_EXECUTE] Failed! - {error_msg}")
            sleep(1000)
            return True, ""

    def _process_tool_response(self, tool_output: str) -> Tuple[List[str], bool]:
        """
        Process the tool execution response from iflow-cli.
        
        Args:
            tool_output: The raw output from iflow tool execution
            
        Returns:
            Tuple of (observations, is_effective)
        """
        self.logger.info(f"[TOOL_RESPONSE] START - Processing tool execution response")

        with open("tool_output.json", "w") as f:
            f.write(json.dumps(tool_output, ensure_ascii=False))
        
        if "Show help  [boolean]" in tool_output:
            tool_output = tool_output.split("Show help  [boolean]")[-1].strip()
        if "{\r\n  \"success\"" in tool_output:
            tool_output = tool_output.split("{\r\n  \"success\"")[-1]
            tool_output = "{\r\n  \"success\"" + tool_output
            
        if "< /dev/null > command.txt 2>&1" in tool_output:
            tool_output = tool_output.split("\n}\r\n")[0].strip() + "\n}"
    
        if not tool_output:
            return "No tool output received", False
        
        parse_success = False
        try:
            try:
                if isinstance(tool_output, str):
                    try:
                        tool_output = json.loads(tool_output)
                    except Exception as e:
                        print(e)
                        tool_output = ast.literal_eval(tool_output)
                else:
                    tool_output = tool_output
                if isinstance(tool_output, dict) and "tool_results" in tool_output:
                    tool_results = tool_output["tool_results"]
                    observations = []
                    for i, tool_result in enumerate(tool_results):
                        if isinstance(tool_result, str):
                            tool_result = json.loads(tool_result)
                        tool_response = ""
                        if "full_response" in tool_result:
                            tool_response = tool_result.get("full_response", "")
                            tool_response = tool_response.replace("bash: warning: setlocale: LC_ALL: cannot change locale (zh_CN.UTF-8)", "")
                            if tool_response == "":
                                tool_response = ""
                        elif "error" in tool_result:
                            error = tool_result.get("error", "")
                            tool_response = f"ERROR: {error}"
                        else:
                            tool_response = json.dumps(tool_result, ensure_ascii=False)
                        cur_observation = f"<tool_response>\n{tool_response}\n</tool_response>"
                        observations.append(cur_observation)
                        self.logger.debug(f"[TOOL_RESPONSE] Processed tool result {i+1}")
                    
                    observation = "\n".join(observations)
                    self.logger.info(f"[TOOL_RESPONSE] Success! - Processed {len(tool_results)} tool results")
                    return observation, True
                else:
                    self.logger.warning(f" [TOOL_RESPONSE] No tool_results found in response")
                    observation = tool_output
                    parse_success = False

            except json.JSONDecodeError:
                self.logger.warning(f" [TOOL_RESPONSE] Failed to parse JSON, returning raw output")
                observation = tool_output
                parse_success = False
                
        except Exception as e:
            self.logger.error(f"[TOOL_RESPONSE] Failed! - Error processing tool response: {e}")
            return tool_output, False


    def get_tools(self) -> Dict[str, Any]:
        """Get available tools"""
        self.logger.info(f"[GET_TOOLS] START - Loading available tools")
        
        try:
            if os.path.exists("roll/pipeline/agentic/tools/iflow/iflow_tools.json"):
                with open("roll/pipeline/agentic/tools/iflow/iflow_tools.json") as f:
                    tools = json.load(f)
                self.logger.info(f"[GET_TOOLS] Success! - Loaded {len(tools)} tools from iflow_tools.json")
            else:
                tools = []
                self.logger.warning(f" [GET_TOOLS] Tools file not found, using empty tools list")
            return tools
        except Exception as e:
            self.logger.error(f"[GET_TOOLS] Failed! - Error loading tools: {e}")
            return []

