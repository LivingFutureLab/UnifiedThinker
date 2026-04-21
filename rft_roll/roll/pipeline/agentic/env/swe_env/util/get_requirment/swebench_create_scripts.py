from swebench.harness.test_spec.javascript import (
    make_repo_script_list_js,
    make_env_script_list_js,
    make_eval_script_list_js,
)

# from swebench.harness.test_spec.python import (
#     make_repo_script_list_py,
#     make_env_script_list_py,
#     make_eval_script_list_py,
# )

from roll.pipeline.agentic.env.swe_env.util.get_requirment.swebench_python_inswebench import (
    make_repo_script_list_py,
    make_env_script_list_py,
    make_eval_script_list_py,
)

from swebench.harness.constants import MAP_REPO_TO_EXT


def make_repo_script_list(specs, repo, repo_directory, base_commit, env_name) -> list:
    """
    Create a list of bash commands to set up the repository for testing.
    This is the setup script for the instance image.
    """
    ext = MAP_REPO_TO_EXT[repo]
    func = {
        "js": make_repo_script_list_js,
        "py": make_repo_script_list_py,
    }[ext]
    try:
        return func(specs, repo, repo_directory, base_commit, env_name)
    except Exception as e:
        print(
            f"[DEBUG][make_repo_script_list]specs: {specs}, repo: {repo}, repo_directory: {repo_directory}, base_commit: {base_commit}, env_name: {env_name}, error: {e}"
        )
        # raise e
        return []


def make_env_script_list(instance, specs, env_name) -> list:
    """
    Creates the list of commands to set up the environment for testing.
    This is the setup script for the environment image.
    """
    ext = MAP_REPO_TO_EXT[instance["repo"]]
    func = {
        "js": make_env_script_list_js,
        "py": make_env_script_list_py,
    }[ext]
    try:
        return func(instance, specs, env_name)  # 死在这里
    except Exception as e:
        print(f"[DEBUG][make_env_script_list]env_name: {env_name}, error: {e}")
        # raise e
        return []


def make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch) -> list:
    """
    Applies the test patch and runs the tests.
    """
    ext = MAP_REPO_TO_EXT[instance["repo"]]
    func = {
        "js": make_eval_script_list_js,
        "py": make_eval_script_list_py,
    }[ext]
    return func(instance, specs, env_name, repo_directory, base_commit, test_patch)
