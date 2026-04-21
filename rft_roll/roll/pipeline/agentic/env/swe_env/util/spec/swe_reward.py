import json
import re
from datasets.info import get_logger
from swebench.harness.log_parsers.javascript import MAP_REPO_TO_PARSER_JS
from swebench.harness.log_parsers.python import MAP_REPO_TO_PARSER_PY
from swebench.harness.log_parsers.utils import get_eval_type
from swebench.harness.test_spec.test_spec import TestSpec

from tests.agentic.sweenv.utils.utils_file import *

MAP_REPO_TO_PARSER = {
    **MAP_REPO_TO_PARSER_JS,
    **MAP_REPO_TO_PARSER_PY,
}
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)

#
from roll.pipeline.agentic.env.swe_env.util.spec.config import SUPPORTED_REPOS, SKIP_FILES, SKIP_FILES_NEW, CMD_TIMEOUT
from roll.pipeline.agentic.env.swe_env.util.spec.execution_log_parser import parse_log_fn, decolor_dict_keys
from roll.pipeline.agentic.env.swe_env.util.spec.grading import get_eval_tests_report, get_resolution_status
from roll.pipeline.agentic.env.swe_env.util.get_requirment.swebench_test_spec import make_test_spec


class SweReward:
    def __init__(self, logger=None):
        if not logger:
            self.logger = get_logger("SWEReward")
        else:
            self.logger = logger

    def calculate_reward_r2e(
        self,
        output: str,
        ds: dict = None,
        repo_name: str = None,
        expected_json: json = None,
        get_test_output=False,
        route_key: str = None,
    ) -> float:
        """
        Calculate the reward for the r2e task.
        """
        # parsed
        parse = parse_log_fn(f"{repo_name}")(output)
        parse = decolor_dict_keys(parse)
        # excepted
        expected: dict = json.loads(expected_json)
        expected = decolor_dict_keys(expected)
        # 过滤掉空键，避免 KeyError
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys()) if k.strip()}
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys()) if k.strip()}
        # Compare
        if not parse:
            print(
                f"[ATTENTION][计算reward可能存在问题-trainset]parse is null. parse: {parse}, expected: {expected}, output: {[output]}, route_key: {route_key}"
            )
            self.logger.error(
                f"[ATTENTION][计算reward可能存在问题-trainset]parse is null. parse: {parse}, expected: {expected}, output: {[output]}, route_key: {route_key}"
            )
            reward = 0.0
        elif not expected:
            print(
                f"[ATTENTION][计算reward存在问题! trainset]expected is null. parse: {parse}, expected: {expected}, output: {[output]}, route_key: {route_key}"
            )
            self.logger.error(
                f"[ATTENTION][计算reward存在问题! trainset]expected is null. parse: {parse}, expected: {expected}, output: {[output]}, route_key: {route_key}"
            )
            reward = 0.0
        elif len(parse) != len(expected):
            reward = 0.0
        else:
            # If ANY mismatch, reward = 0.0, else = 1.0
            match = True
            for k in parse.keys():
                if k not in expected:
                    print(
                        f"[ATTENTION][计算reward可能存在问题-trainset]k not in expected. k: {k}, parse: {parse}, expected: {expected}, output: {[output]}, route_key: {route_key}"
                    )
                    self.logger.error(
                        f"[ATTENTION][计算reward可能存在问题-trainset]k not in expected. k: {k}, parse: {parse}, expected: {expected}, output: {[output]}, route_key: {route_key}"
                    )
                    continue
                elif parse[k] != expected[k] and parse[k] == "FAILED":
                    match = False
                    break
            reward = 1.0 if match else 0.0
        # If the caller wants the test output as well, return (reward, output)
        self.logger.info(
            "========== 计算reward in trainset ==========\n"
            "\n************* parse **********\n"
            f"{parse}"
            "\n************* expected **********\n"
            f"{expected}"
            f"\n************* reward = {reward} **********\n"
        )
        # if get_test_output:
        # return reward, output
        return reward

    def calculate_reward_swebench(self, ds, out: str, get_test_output=False) -> float:
        """
        Calculate the reward for the swebench task.
        """
        test_spec = make_test_spec(ds)
        # get eval logs
        eval_status_map, found = self.get_logs_eval(test_spec, out)
        eval_ref = {
            KEY_INSTANCE_ID: test_spec.instance_id,
            FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=get_eval_type(test_spec))
        success = get_resolution_status(report) == ResolvedStatus.FULL.value

        self.logger.info(
            "========== 计算reward in swebench ==========\n"
            "\n************* eval_status_map **********\n"
            f"{eval_status_map}"
            "\n************* eval_ref **********\n"
            f"{eval_ref}"
            "\n************* report **********\n"
            f"{report}"
            f"\n************* reward = {int(success)} **********\n"
        )
        if get_test_output:
            return int(success), out
        return int(success)

    def get_logs_eval(self, test_spec: TestSpec, content: str) -> tuple[dict[str, str], bool]:
        """
        Retrieve evaluation results for a task instance from its corresponding log file

        Args:
            log_fp (str): path to log file
        Returns:
            bool: whether the patch applied successfully
            dict: status map

        modified from swebench/harness/grading.py
        """
        repo = test_spec.repo
        version = test_spec.version
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        # print(f"[DEBUG]repo: {repo}")
        # print(f"[DEBUG]version: {version}")
        # print(f"[DEBUG]MAP_REPO_TO_PARSER keys: {list(MAP_REPO_TO_PARSER.keys())}")
        # print(f"[DEBUG]MAP_REPO_VERSION_TO_SPECS keys: {list(MAP_REPO_VERSION_TO_SPECS.keys())}")

        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]
        # print(f"[DEBUG]test_cmd: {test_cmd}")
        # print(f"[DEBUG]content contains test_cmd: {test_cmd in content}")

        # with open(log_fp) as f:
        # # TODO fix constant here
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            print(f"[ATTENTION][Reward计算-BadCodeFound]Bad code found in log: {bad_codes}")
            self.logger.error(f"[ATTENTION][Reward计算-BadCodeFound]Bad code found in log: {bad_codes}")
            return {}, False

        # elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
        #     # Test patch did not apply (should not happen at all)
        #     print("Test patch did not apply")
        #     return {}, False

        # Get status map of evaluation results
        # content = content.split(test_cmd)[-1]

        try:
            return log_parser(content, test_spec), True
        except Exception as e:
            print(
                f"[ATTENTION][Reward计算-解析log存在问题]Error parsing logs for repo {repo}: {repr(e)}"
                f"Content preview: {content[:500]}..."
            )
            self.logger.error(
                f"[ATTENTION][Reward计算-解析log存在问题]Error parsing logs for repo {repo}: {repr(e)}"
                f"Content preview: {content[:500]}..."
            )
            return {}, False


if __name__ == "__main__":
    reward = SweReward()

    """测试trainset"""
    data = load_data(
        "/home/lixing/workspace/swe_rele/dataset/2_docker_file/250814_trainset_v1_r2e_lite_vpc_4578_split100/part_0.jsonl"
    )
    ds = data[0]
    out = """============================= test session starts ==============================
platform linux -- Python 3.9.21, pytest-8.3.4, pluggy-1.5.0
rootdir: /testbed
plugins: asyncio-0.25.2, cov-6.0.0, mock-3.14.0
asyncio: mode=strict, asyncio_default_fixture_loop_scope=None
collected 64 items

r2e_tests/test_1.py .....F.............................................. [ 81%]
............                                                             [100%]

=================================== FAILURES ===================================
_______________ TestUrlDispatcher.test_add_route_with_invalid_re _______________

self = <r2e_tests.test_1.TestUrlDispatcher testMethod=test_add_route_with_invalid_re>

    def test_add_route_with_invalid_re(self):
        handler = self.make_handler()
        with self.assertRaises(ValueError) as ctx:
            self.router.add_route('GET', r'/handler/{to:+++}', handler)
        s = str(ctx.exception)
>       self.assertTrue(s.startswith(
            "Bad pattern '\/handler\/(?P<to>+++)': nothing to repeat"), s)
E       AssertionError: False is not true : Bad pattern '/handler/(?P<to>+++)': nothing to repeat at position 17

r2e_tests/test_1.py:402: AssertionError
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_invalid_path
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_route_invalid_method
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_route_not_started_with_slash
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_route_root
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_route_simple
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_route_with_re
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_route_with_re_and_slashes
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_route_with_re_including_slashes
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_route_with_re_not_match
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_static
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_url_escaping
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_url_invalid1
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_url_invalid2
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_url_invalid3
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_url_invalid4
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_with_matchdict
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_with_name
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_add_with_tailing_slash
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_any_method
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_contains
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_custom_expect_handler_dynamic
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_custom_expect_handler_plain
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_default_expect_handler
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_double_add_url_with_the_same_name
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_dynamic_match_non_ascii
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_dynamic_match_two_part2
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_dynamic_match_unquoted_path
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_dynamic_match_with_static_part
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_dynamic_not_match
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_dynamic_repr
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_dynamic_with_trailing_slash
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_expect_handler_non_coroutine
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_iter
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_len
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_match_second_result_in_table
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_not_allowed_repr
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_not_found_repr
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_plain_not_match
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_plain_repr
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_raise_method_not_allowed
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_raise_method_not_found
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_register_route
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_register_route_checks
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_regular_match_info
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_route_dynamic
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_route_dynamic_with_regex
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_route_dynamic_with_regex_spec
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_route_dynamic_with_regex_spec_and_trailing_slash
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_route_plain
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_route_unknown_route_name
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_route_with_qs
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_routes_abc
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_routes_view_contains
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_routes_view_iter
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_routes_view_len
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_static_adds_slash
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_static_dont_add_trailing_slash
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_static_handle_again
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_static_handle_eof
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_static_handle_exception
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_static_not_match
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_static_repr
PASSED r2e_tests/test_1.py::TestUrlDispatcher::test_system_route
FAILED r2e_tests/test_1.py::TestUrlDispatcher::test_add_route_with_invalid_re
========================= 1 failed, 63 passed in 0.11s =========================
/root/.venv/lib/python3.9/site-packages/pytest_asyncio/plugin.py:207: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
"""

    # print(reward.calculate_reward_r2e(out, ds, repo_name=ds['repo_name'], expected_json=ds['expected_output_json'], get_test_output=False, route_key='8faede633f444304993264e11ae2a544'))
    # print('\n[expected_json]\n',ds['expected_output_json'])
    # write_data_json(ds,'/home/lixing/workspace/process/swe/log/eval_local/piplinev6_cleanswe/trainset_idx_0_expected_output_json.json')

    """测试swe_bench"""
    data = load_data(
        "/home/lixing/workspace/swe_rele/dataset/2_docker_file/250814_valset_v1_swe_bench_verified_vpc_500_split100/part_0.jsonl"
    )
    ds = data[0]
    out = """Writing to /root/.config/pip/pip.conf
On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   pyproject.toml

no changes added to commit (use "git add" and/or "git commit -a")
commit d16bfe05a744909de4b27f5875fe0d4ed41ce607
Merge: a4f25a2ced 95f3d4da59
Author: William Jamieson <wjamieson@stsci.edu>
Date:   Thu Mar 3 13:21:56 2022 -0500

    Merge pull request #12900 from Cadair/custom_compound_model
    
    Allow a model to override calculation of it's separability matrix

diff --git a/pyproject.toml b/pyproject.toml
index 3364d30740..02dddbe713 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1,5 +1,5 @@
 [build-system]
-requires = ["setuptools",
+requires = ["setuptools==68.0.0",
             "setuptools_scm>=6.2",
             "wheel",
             "cython==0.29.22",
============================= test session starts ==============================
platform linux -- Python 3.9.21, pytest-7.4.0, pluggy-1.3.0

Running tests with Astropy version 5.1.dev623+gd16bfe05a7.d20250104.
Running tests in astropy/modeling/tests/test_separable.py.

Date: 2025-09-07T16:32:23

Platform: Linux-5.10.134-18.al8.x86_64-x86_64-with-glibc2.35

Executable: /opt/miniconda3/envs/testbed/bin/python

Full Python Version: 
3.9.21 (main, Dec 11 2024, 16:24:11) 
[GCC 11.2.0]

encodings: sys: utf-8, locale: UTF-8, filesystem: utf-8
byteorder: little
float info: dig: 15, mant_dig: 15

Package versions: 
Numpy: 1.25.2
Scipy: not available
Matplotlib: not available
h5py: not available
Pandas: not available
PyERFA: 2.0.0.3
Cython: not available
Scikit-image: not available
asdf: not available
pyarrow: not available

Using Astropy options: remote_data: none.

ARCH_ON_CI: undefined
IS_CRON: undefined

rootdir: /testbed
configfile: setup.cfg
plugins: hypothesis-6.82.6, arraydiff-0.5.0, astropy-0.10.0, astropy-header-0.2.2, cov-4.1.0, doctestplus-1.0.0, filter-subpackage-0.1.2, mock-3.11.1, openfiles-0.5.0, remotedata-0.4.0, xdist-3.3.1
collected 15 items

astropy/modeling/tests/test_separable.py ..........F..F.                 [100%]

=================================== FAILURES ===================================
___________________ test_separable[compound_model6-result6] ____________________

compound_model = <CompoundModel(angle_0=2., offset_1=1., offset_2=2.)>
result = (array([False, False,  True,  True]), array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True, False],
       [False, False, False,  True]]))

    @pytest.mark.parametrize(('compound_model', 'result'), compound_models.values())
    def test_separable(compound_model, result):
>       assert_allclose(is_separable(compound_model), result[0])

astropy/modeling/tests/test_separable.py:151: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<function assert_allclose.<locals>.compare at 0x7fe94ba305e0>, array([False, False, False, False]), array([False, False,  True,  True]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=1e-07, atol=0', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=1e-07, atol=0
E           
E           Mismatched elements: 2 / 4 (50%)
E            x: array([False, False, False, False])
E            y: array([False, False,  True,  True])

/opt/miniconda3/envs/testbed/lib/python3.9/contextlib.py:79: AssertionError
___________________ test_separable[compound_model9-result9] ____________________

compound_model = <CompoundModel(angle_0=2., offset_1=1., factor_2=1., factor_3=2.)>
result = (array([False, False,  True,  True,  True]), array([[ True,  True, False, False, False],
       [ True,  True, False, ... False,  True, False, False],
       [False, False, False,  True, False],
       [False, False, False, False,  True]]))

    @pytest.mark.parametrize(('compound_model', 'result'), compound_models.values())
    def test_separable(compound_model, result):
>       assert_allclose(is_separable(compound_model), result[0])

astropy/modeling/tests/test_separable.py:151: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<function assert_allclose.<locals>.compare at 0x7fe94ba30a60>, array([False, False,  True, False, False]), array([False, False,  True,  True,  True]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=1e-07, atol=0', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=1e-07, atol=0
E           
E           Mismatched elements: 2 / 5 (40%)
E            x: array([False, False,  True, False, False])
E            y: array([False, False,  True,  True,  True])

/opt/miniconda3/envs/testbed/lib/python3.9/contextlib.py:79: AssertionError
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED astropy/modeling/tests/test_separable.py::test_coord_matrix
PASSED astropy/modeling/tests/test_separable.py::test_cdot
PASSED astropy/modeling/tests/test_separable.py::test_cstack
PASSED astropy/modeling/tests/test_separable.py::test_arith_oper
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model0-result0]
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model1-result1]
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model2-result2]
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model3-result3]
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model4-result4]
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model5-result5]
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model7-result7]
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model8-result8]
PASSED astropy/modeling/tests/test_separable.py::test_custom_model_separable
FAILED astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]
FAILED astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9]
========================= 2 failed, 13 passed in 0.29s =========================
+ conda activate testbed
/run_tests.sh: line 5: conda: command not found
+ cd /testbed
+ git config --global --add safe.directory /testbed
fatal: $HOME not set
+ cd /testbed
+ git status
+ git show
+ git diff d16bfe05a744909de4b27f5875fe0d4ed41ce607
+ conda activate testbed
/run_tests.sh: line 13: conda: command not found
+ git checkout d16bfe05a744909de4b27f5875fe0d4ed41ce607 astropy/modeling/tests/test_separable.py
Updated 0 paths from 4d9ea46e57
+ git apply -v -
Checking patch astropy/modeling/tests/test_separable.py...
Applied patch astropy/modeling/tests/test_separable.py cleanly.
+ pytest -rA astropy/modeling/tests/test_separable.py
<frozen importlib._bootstrap>:228: RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility. Expected 80 from C header, got 96 from PyObject
+ git checkout d16bfe05a744909de4b27f5875fe0d4ed41ce607 astropy/modeling/tests/test_separable.py
Updated 1 path from 4d9ea46e57
+ cat: 300: No such file or directory
/run_tests.sh: line 57: cat:: command not found
    """
    print(reward.calculate_reward_swebench(ds, out, get_test_output=False))
