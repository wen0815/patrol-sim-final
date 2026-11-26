# -*- coding: utf-8 -*-
"""
10x10 网格巡检机器人仿真后端（标准版）

依赖安装:
    pip install flask flask-cors

运行:
    python app.py

接口:
    GET  /                托管前端页面 index.html
    GET  /api/state       获取地图和机器人状态
    POST /api/move        手动移动，JSON: {"direction": "up"}
    POST /api/auto_patrol 自动寻路，JSON: {"goal": [9, 9]}
    POST /api/reset       重置机器人到起点
"""

import copy
import heapq
import itertools
import os
import threading

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.json.ensure_ascii = False  # 让 JSON 中的中文信息可读

# 允许所有来源跨域访问 /api/* 接口
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# 地图：0 = 可通行，1 = 障碍物
# maze[y][x] 中 y 为行（上下），x 为列（左右）
# ---------------------------------------------------------------------------
MAZE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
]

ROWS = len(MAZE)
COLS = len(MAZE[0])

START_STATE = {"x": 0, "y": 0, "direction": "up"}
robot_state = dict(START_STATE)

# 方向 -> (dx, dy)；y 向下为正，所以 up 是 y-1
DIRECTION_DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Flask 开发服务器默认多线程，用锁保护全局状态
state_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 地图与 A* 算法
# ---------------------------------------------------------------------------
def in_bounds(x, y):
    """判断坐标是否位于 10x10 地图内。"""
    return 0 <= x < COLS and 0 <= y < ROWS


def is_walkable(x, y):
    """坐标在地图内且不是障碍物时，才可通行。"""
    return in_bounds(x, y) and MAZE[y][x] == 0


def manhattan(a, b):
    """A* 启发函数：曼哈顿距离。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_search(start, goal):
    """
    使用 A* 算法搜索从 start 到 goal 的最短路径。

    参数:
        start: (x, y) 起点
        goal:  (x, y) 终点

    返回:
        [(x1, y1), (x2, y2), ...]，包含起点和终点；
        如果终点是障碍物、越界或不可达，返回 None。
    """
    try:
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
    except (TypeError, ValueError, IndexError):
        return None

    if not is_walkable(*start) or not is_walkable(*goal):
        return None

    if start == goal:
        return [start]

    # 优先队列元素: (f 值, g 值, 入队序号, 坐标)
    # 入队序号用于避免 f、g 相同时比较坐标，保证行为稳定
    open_heap = []
    counter = itertools.count()
    heapq.heappush(
        open_heap,
        (manhattan(start, goal), 0, next(counter), start),
    )

    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_heap:
        _, current_g, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current == goal:
            # 从终点回溯到起点，再反转得到正向路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)

        for dx, dy in DIRECTION_DELTAS.values():
            neighbor = (current[0] + dx, current[1] + dy)

            if not is_walkable(*neighbor) or neighbor in closed:
                continue

            tentative_g = current_g + 1
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            f_score = tentative_g + manhattan(neighbor, goal)
            heapq.heappush(
                open_heap,
                (f_score, tentative_g, next(counter), neighbor),
            )

    return None


def direction_between(start, end):
    """根据路径中的相邻两点，计算机器人朝向。"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    if dx == 1:
        return "right"
    if dx == -1:
        return "left"
    if dy == 1:
        return "down"
    return "up"


def snapshot_state():
    """返回机器人状态和地图的副本，避免外部修改内部数据。"""
    return copy.deepcopy(robot_state), copy.deepcopy(MAZE)


def parse_goal(raw):
    """校验并解析前端传来的目标坐标，支持 [x, y] 或 {"x": x, "y": y}。"""
    if isinstance(raw, dict):
        raw = [raw.get("x"), raw.get("y")]

    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None

    x, y = raw
    if isinstance(x, bool) or isinstance(y, bool):
        return None
    if not isinstance(x, int) or not isinstance(y, int):
        return None
    if not in_bounds(x, y):
        return None

    return x, y


# ---------------------------------------------------------------------------
# 页面与 API
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """托管前端页面：访问 http://localhost:5000 即可打开。"""
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/state")
def get_state():
    """返回当前地图与机器人状态。"""
    with state_lock:
        robot, maze = snapshot_state()

    return jsonify(
        {
            "maze": maze,
            "robot": robot,
            "rows": ROWS,
            "cols": COLS,
        }
    )


@app.post("/api/move")
def move():
    """手动移动机器人，移动前进行边界检查和障碍物检查。"""
    data = request.get_json(silent=True) or {}
    direction = data.get("direction")

    if direction not in DIRECTION_DELTAS:
        with state_lock:
            robot, _ = snapshot_state()
        return (
            jsonify(
                {
                    "success": False,
                    "error": "无效方向，direction 必须是 up / down / left / right",
                    "robot": robot,
                }
            ),
            400,
        )

    dx, dy = DIRECTION_DELTAS[direction]

    with state_lock:
        new_x = robot_state["x"] + dx
        new_y = robot_state["y"] + dy

        if not in_bounds(new_x, new_y):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "移动失败：前方超出地图边界",
                        "robot": copy.deepcopy(robot_state),
                    }
                ),
                400,
            )

        if MAZE[new_y][new_x] == 1:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "移动失败：前方有障碍物",
                        "robot": copy.deepcopy(robot_state),
                    }
                ),
                409,
            )

        robot_state["x"] = new_x
        robot_state["y"] = new_y
        robot_state["direction"] = direction
        robot, maze = snapshot_state()

    return jsonify({"success": True, "robot": robot, "maze": maze})


@app.post("/api/auto_patrol")
def auto_patrol():
    """
    自动寻路接口。

    请求体:
        {"goal": [9, 9]}

    成功:
        {"status": "success", "path": [[0, 0], [0, 1], ...]}

    失败:
        {"status": "error", "message": "目标不可达，请重新选择"}
    """
    data = request.get_json(silent=True) or {}
    goal = parse_goal(data.get("goal", data.get("target")))

    if goal is None:
        return (
            jsonify({"status": "error", "message": "无效的目标坐标"}),
            400,
        )

    with state_lock:
        start = (robot_state["x"], robot_state["y"])
        path = a_star_search(start, goal)

        if path is None:
            return jsonify(
                {"status": "error", "message": "目标不可达，请重新选择"}
            )

        # 后端没有动画，直接同步机器人状态到目标点；
        # 前端拿到 path 后只负责逐格播放动画。
        robot_state["x"], robot_state["y"] = goal
        if len(path) >= 2:
            robot_state["direction"] = direction_between(path[-2], path[-1])

    return jsonify(
        {
            "status": "success",
            "path": [[x, y] for x, y in path],
        }
    )


@app.post("/api/reset")
def reset():
    """把机器人重置到起点（0, 0）。"""
    with state_lock:
        robot_state.update(START_STATE)
        robot, maze = snapshot_state()

    return jsonify({"success": True, "robot": robot, "maze": maze})


if __name__ == "__main__":
    # host=0.0.0.0 让容器外可以访问；生产环境请用 gunicorn/waitress 等
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=debug)
