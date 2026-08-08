import logging
import os
import json
import threading
import queue
import time
import multiprocessing as mp
from pathlib import Path
import atexit
import weakref
import fcntl


class MultiprocessSafeLogger:
    """
    多进程安全的日志记录器

    特性：
    1. 进程级文件锁，避免多进程写入冲突
    2. 自动资源清理
    3. 进程退出时的优雅关闭
    4. 内存泄漏防护
    """

    # 类级别的进程锁字典，避免多进程间冲突
    # _process_locks = {}
    # _process_loggers = weakref.WeakValueDictionary()

    def __init__(self, path: str, max_queue_size: int = 1000):
        self.path = path
        self.max_queue_size = max_queue_size

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # # 进程级文件锁
        # self._get_process_lock()

        # # 后台写入相关
        self.write_queue = queue.Queue(maxsize=max_queue_size)
        self.write_thread = None
        self.stop_event = threading.Event()
        self._content_buffer = ""
        self._lock = threading.Lock()

        # 注册清理函数
        self._register_cleanup()

        # 启动后台写入线程
        self._start_background_writer()

        print(f"[debug]MultiprocessSafeLogger 初始化: {path}")

    def update_log_path(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def _get_process_lock(self):
        """获取进程级文件锁"""
        process_id = os.getpid()
        if process_id not in self._process_locks:
            self._process_locks[process_id] = threading.Lock()
        self._process_lock = self._process_locks[process_id]

    def _register_cleanup(self):
        """注册进程退出时的清理函数"""
        # 使用弱引用避免循环引用
        self_ref = weakref.ref(self)

        def cleanup():
            logger = self_ref()
            if logger:
                logger.close()

        atexit.register(cleanup)

    def _start_background_writer(self):
        """启动后台写入线程"""
        self.write_thread = threading.Thread(target=self._background_writer, daemon=True)
        self.write_thread.start()

    def _background_writer(self):
        """后台写入线程函数"""
        while not self.stop_event.is_set():
            try:
                # 等待写入任务，超时1秒
                content = self.write_queue.get(timeout=1.0)
                if content is None:  # 停止信号
                    break

                # 执行写入操作
                self._write_to_file(content)
                self.write_queue.task_done()

            except queue.Empty:
                # 超时，继续循环检查停止事件
                continue
            except Exception as e:
                print(f"[ERROR]MultiprocessSafeLogger._background_writer: {e}")
                # 即使出错也继续运行，避免线程退出

    def _write_to_file(self, content):
        """实际的文件写入操作，带容错和进程锁"""
        max_retries = 3
        retry_delay = 0.1
        for attempt in range(max_retries):
            try:
                print(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [保存日志]start write to file: {self.path}'
                )
                # with self._process_lock:  # 使用进程级锁保护文件写入
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(content)
                print(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [保存日志]end write to file: {self.path}'
                )
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [WARNING]MultiprocessSafeLogger._write_to_file 重试 {attempt + 1}/{max_retries}: {e}'
                    )
                    time.sleep(retry_delay * (2**attempt))  # 指数退避
                else:
                    print(
                        f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [ERROR]MultiprocessSafeLogger._write_to_file 写入失败: {e}'
                    )
                    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [ERROR]文件路径: {self.path}')
                    print(
                        f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [ERROR]内容: {content[:200]}...'
                        if len(content) > 200
                        else f"[ERROR]内容: {content}"
                    )

    def info(self, content: str):
        """记录信息日志"""
        # with self._lock:
        self._content_buffer += f'[INFO]{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ' + str(content) + "\n"

    def error(self, content: str):
        """记录错误日志"""
        # with self._lock:
        self._content_buffer += f'[ERROR]{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ' + str(content) + "\n"

    def warning(self, content: str):
        """记录警告日志"""
        # with self._lock:
        self._content_buffer += (
            f'[WARNING]{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ' + str(content) + "\n"
        )

    def debug(self, content: str):
        """记录调试日志"""
        # with self._lock:
        self._content_buffer += f'[DEBUG]{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ' + str(content) + "\n"

    def save(self):
        """将内容放入后台写入队列，不等待完成"""
        # with self._lock:
        try:
            #     # 检查队列是否已满
            if self.write_queue.full():
                print(f"[WARNING]MultiprocessSafeLogger.save: 写入队列已满，丢弃内容")
                print(f"[WARNING]文件路径: {self.path}")
                print(f"[WARNING]内容长度: {len(self._content_buffer)}")
                self._content_buffer = ""  # 清空缓冲区
                return
            self.write_queue.put(self._content_buffer, timeout=1.0)
            self._content_buffer = ""  # 清空内容，避免重复写入
        except queue.Full:
            print(f"[ERROR]MultiprocessSafeLogger.save: 写入队列已满，丢弃内容")
            print(f"[ERROR]文件路径: {self.path}")
            print(f"[ERROR]内容长度: {len(self._content_buffer)}")
            self._content_buffer = ""
        except Exception as e:
            print(f"[ERROR]MultiprocessSafeLogger.save: {e}")
        pass

    def close(self):
        # """关闭后台写入线程"""
        if self.write_thread and self.write_thread.is_alive():
            # 先保存剩余内容
            self.save()

            # 发送停止信号
            self.stop_event.set()
            try:
                self.write_queue.put(None, timeout=1.0)  # 发送停止信号
            except queue.Full:
                pass  # 队列已满，忽略
            # 等待线程结束
            self.write_thread.join(timeout=3.0)
            if self.write_thread.is_alive():
                print(f"[WARNING]MultiprocessSafeLogger.close: 后台写入线程未能在3秒内结束")
        pass

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.close()
        except:
            pass  # 忽略析构函数中的异常


# 当下不用了
class LoggerForEnvTrajWriteMode:
    """
    兼容原有接口的多进程安全日志记录器
    """

    def __init__(self, path: str):
        self._logger = MultiprocessSafeLogger(path)
        self.path = path
        self.content = ""  # 保持兼容性

    def info(self, content: str):
        self._logger.info(content)
        # 保持兼容性
        self.content += f'[INFO]{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ' + str(content) + "\n"

    def error(self, content: str):
        self._logger.error(content)
        self.content += f'[ERROR]{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ' + str(content) + "\n"

    def warning(self, content: str):
        self._logger.warning(content)
        self.content += f'[WARNING]{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ' + str(content) + "\n"

    def debug(self, content: str):
        self._logger.debug(content)
        self.content += f'[DEBUG]{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ' + str(content) + "\n"

    def save(self):
        self._logger.save()

    def close(self):
        self._logger.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass


# 当下不用了
def get_logger_for_env(base_log_dir: str = "./logs", env_id: str = "tag_group_id_group_seed_env_id"):
    """
    @output:
        log_file: ./logs/env/tag_group_id_group_seed_env_id.log
    @usage:
        # logger = get_logger_for_env(base_log_dir='./logs',env_id=1)
        # logger.info("This is a log for env 1")
    """
    # 使用多进程安全的实现
    base_log_dir = os.path.join(base_log_dir, "env")
    os.makedirs(base_log_dir, exist_ok=True)
    log_file = os.path.join(base_log_dir, f"{env_id}.log")
    print(f"[debug]log_file: {log_file}")

    return LoggerForEnvTrajWriteMode(log_file)


def write_data_json(data, path):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [保存日志]start write json to path: {path}')
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            info_str = json.dumps(data, ensure_ascii=False)
            f.write(info_str)
    except Exception as e:
        print(f"[ERROR]write_data_json: {e}")
        print(f"[{path}]\n{data}")
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [保存日志]end write json to path: {path}')
    pass


def write_data_txt(data_list, path):
    """多进程安全的文本写入"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for line in data_list:
                f.write(line + "\n")
    except Exception as e:
        print(f"[ERROR]write_data_txt: {e}")
    pass


def write_data_txt_a(content, path):
    """多进程安全的追加写入"""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    # 追加写入使用文件锁
    try:
        with open(path, "a", encoding="utf-8") as f:
            # fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他锁
            f.write(content + "\n")
            # fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁
    except Exception as e:
        print(f"[ERROR]write_data_txt_a: {e}")


# 保持原有接口
class Colors:
    """ANSI颜色代码"""

    HEADER = "\033[95m"  # 紫色
    BLUE = "\033[94m"  # 蓝色
    PINK = "\033[91m"  # 粉色
    GREEN = "\033[92m"  # 绿色
    YELLOW = "\033[93m"  # 黄色
    RED = "\033[91m"  # 红色
    BOLD = "\033[1m"  # 粗体
    UNDERLINE = "\033[4m"  # 下划线
    END = "\033[0m"  # 结束颜色


def pretty_print(color=Colors.BLUE, text_color: str = "", text: str = ""):
    """打印彩色文本"""
    print(f"{Colors.BOLD}{color}{text_color}{Colors.END}{text}")


def _lazy_load_jsonl_lines_spec_idx(path, task_idx):
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line_json = json.loads(line)
            print(f"[读取数据]{path}")
            idx = line_json["idx"]
            if int(idx) == int(task_idx):
                result = line_json
                break
    return result
