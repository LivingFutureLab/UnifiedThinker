# -*- coding: utf-8 -*-
"""HTTP client for MTShopSimulatorEnv."""
import logging
import random
import time
from typing import Any, Dict

import requests


class MTShopSimulatorClient:
    """One-shot HTTP client (线程安全：无状态、无共享连接池)."""

    def __init__(self, base_url: str, authorization: str, server_key: str, *, timeout: float, max_retries: int, backoff: float):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.max_retries, self.backoff = max_retries, backoff
        self.authorization = authorization
        self.server_key = server_key
        self.logger = logging.getLogger("MTShopSimulatorClient")
        self.keys = [
            "d380658e-40a4-4ea0-9b5f-13a2e28d6271",
            "e7adf378-7759-4253-8c3c-9a38f8506b25",
            "9d2eab4b-99e1-417c-9c7f-90fba94687d1",
            "220d4447-dbe9-489f-aa14-e0e85626af55",
            "9f143c4d-a40c-4a74-9bce-aa0b2c39f86b",
            "a03af7bf-df1e-45f1-b869-ee54a38e1ce7",
            "a6e91da6-aec4-40be-9e01-83d8a26789e3",
            "1cc2a54d-be8c-49d2-8ac0-037cdad1622b",
            "afbdc5c8-b2be-4880-b27d-6bd190dd9953",
            "37363531-e0bc-41e9-91d0-27cc4d72acd3",
            "54dbef1d-62d4-4062-b60b-5b820111997b",
            "80e89db3-dc2a-481b-8a60-63d3197a450a",
            "5df9fa20-e5eb-45ea-9fb4-72d046d335a7",
            "dba4a931-31d1-47f1-a53b-33935d0513ca",
            "550d9e21-f1d2-4cf9-87b1-c7c4e969adf7",
            "875bb764-76da-452e-b43a-da5af93f43df",
            "6ffc2583-ff17-45ef-bfe9-1de09e848baf",
            "1ba4bbc6-3034-46c5-9085-c5b3214a71f5",
            "78dcbee5-f8fc-45bc-9532-6828ece5d9c7",
            "ab2add2f-fe71-47e4-a138-5a0f93c0766c"
    ]

    # -------- internal -------- #
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        retry = 0
        while True:
            try:
                if self.server_key:
                    # 如果是reset，则随机选择一个server_key
                    if payload["action"] == "reset":
                        key_idx = random.choice(self.server_key)
                        server_key = self.keys[key_idx]
                    # 如果是其他行为，则需要解析env_idx和server_key
                    elif payload["action"] == "release_one" or payload["action"] == "interact": 
                        env_info = payload["env_idx"]
                        payload["env_idx"] = int(env_info.split("_")[0])
                        server_key = int(env_info.split("_")[1])
                        server_key = self.keys[server_key]
                    else:
                        # 主动报错，防止后续的逻辑错误
                        assert False, "Invalid API call action"
                    r = requests.post(
                        f"{self.base_url}/api/shop_agent",  # 修改为正确的API端点
                        json=payload, 
                        timeout=self.timeout,
                        headers={
                            'BusinessType': 'inner',
                            'Content-Type': 'application/json',
                            "XRL-Authorization": self.authorization,
                            "CHATOS-MCP-SERVER-ROUTE-KEY": server_key
                        }
                    )
                    result = r.json()["result"]
                    # 如果是reset，则需要返回env_idx
                    if payload["action"] == "reset":
                        result['env_idx'] = f"{result['env_idx']}_{key_idx}"
                else:
                    r = requests.post(
                        f"{self.base_url}/api/shop_agent",  # 修改为正确的API端点
                        json=payload, 
                        timeout=self.timeout,
                        headers={
                            'Content-Type': 'application/json',
                            "Authorization": self.authorization
                        }
                    )
                    result = r.json()["result"]
                r.raise_for_status()
                return result
            except requests.RequestException as e:
                retry += 1
                if retry > self.max_retries:
                    self.logger.error("HTTP failed after %d retries: %s", retry - 1, e)
                    raise
                self.logger.warning("HTTP error, retry %d: %s", retry, e)
                time.sleep(self.backoff * retry)

    # -------- public API (每次都带唯一 call_id) -------- #
    def reset(self, idx: int) -> Dict[str, Any]:
        return self._post({
            "action": "reset", 
            "idx": idx
        })

    def interact(self, env_idx: int, response: str) -> Dict[str, Any]:
        return self._post({
            "action": "interact", 
            "env_idx": env_idx,
            "response": response
        })

    def release_one(self, env_idx: int):
        # 后端若无 release_one 接口可忽略异常
        try:
            self._post({
                "action": "release_one", 
                "env_idx": env_idx
            })
        except Exception:  # noqa
            self.logger.debug("release_one failed (ignored)")

    def release_all(self):
        try:
            self._post({
                "action": "release_all"
            })
        except Exception:  # noqa
            self.logger.debug("release_lst failed (ignored)") 