# -*- coding: utf-8 -*-
"""HTTP client for ShopSimulatorEnv."""
import logging
import random
import time
from typing import Any, Dict

import requests


class ShopSimulatorClient:
    """One-shot HTTP client (线程安全：无状态、无共享连接池)."""

    def __init__(self, base_url: str, authorization: str, server_key: str, *, timeout: float, max_retries: int, backoff: float):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.max_retries, self.backoff = max_retries, backoff
        self.authorization = authorization
        self.server_key = server_key
        self.logger = logging.getLogger("ShopSimulatorClient")
        self.keys = [
            # 20组[0,19] inner集群， 'BusinessType: inner'
            "d380658e-40a4-4ea0-9b5f-13a2e28d6271",
    ]

    # -------- internal -------- #
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        retry = 0
        print(f'[DEBUG][base_url]{self.base_url}/api/shop_agent')
        while True:
            try:
                if self.server_key:
                    # 如果是reset，则随机选择一个server_key
                    if payload["action"] == "reset":
                        key_idx = random.choice(self.server_key)
                        server_key = self.keys[key_idx]
                    # 如果是其他行为，则需要解析env_idx和server_key
                    elif payload["action"] == "release_one" or payload["action"] == "interact":
                        env_info = str(payload["env_idx"])
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
                            'BusinessType': 'nonpersona',
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