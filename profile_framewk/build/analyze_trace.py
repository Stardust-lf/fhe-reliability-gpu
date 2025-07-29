#!/usr/bin/env python3
import re
import sys
from collections import defaultdict

class Node:
    def __init__(self, level, name):
        self.level    = level
        self.name     = name.strip()
        self.time     = 0
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __str__(self, indent=0):
        s  = "  " * indent + f"{self.level}:{self.name} — {self.time} µs\n"
        for c in self.children:
            s += c.__str__(indent + 1)
        return s

def parse_trace(filename):
    """
    将原始 trace 解析为一棵或者多棵原始调用树
    """
    pat = re.compile(
        r'^(?P<level>function|frontend|evaluator|dwthandler):\s*'
        r'(?P<name>[^\[]+?)'
        r'(?:\[(?P<time>\d+)\s+microseconds\])?$'
    )

    roots = []
    stack = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            m = pat.match(line)
            if not m:
                continue

            lvl  = m.group("level")
            name = m.group("name")
            t    = m.group("time")

            if t is None:
                # 开始一个新节点
                node = Node(lvl, name)
                if stack:
                    stack[-1].add_child(node)
                else:
                    roots.append(node)
                stack.append(node)

            else:
                # 结束一个节点，记录耗时
                elapsed = int(t)
                # 在 stack 里向上查找匹配的 start
                for i in range(len(stack)-1, -1, -1):
                    if stack[i].level == lvl and stack[i].name == name.strip():
                        node = stack.pop(i)
                        node.time = elapsed
                        break
                else:
                    # 找不到就当作孤立节点插入
                    node = Node(lvl, name)
                    node.time = elapsed
                    if stack:
                        stack[-1].add_child(node)
                    else:
                        roots.append(node)

    return roots

def aggregate_node(node):
    """
    对同级同名的 children 做归并：累加耗时、合并它们的子节点
    然后对子节点递归
    """
    buckets = defaultdict(list)
    for c in node.children:
        buckets[(c.level, c.name)].append(c)

    new_children = []
    for (lvl, name), group in buckets.items():
        merged = Node(lvl, name)
        # 累加所有同名节点的时间
        merged.time = sum(g.time for g in group)
        # 收集它们的所有子节点，放到一个列表里
        for g in group:
            merged.children.extend(g.children)
        # 递归合并
        merged = aggregate_node(merged)
        new_children.append(merged)

    node.children = new_children
    return node

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_trace.py <trace_file>")
        sys.exit(1)

    roots = parse_trace(sys.argv[1])
    # 只保留顶层的 function: 节点
    funcs = [r for r in roots if r.level == "function"]
    if not funcs:
        print("Error: no function: blocks found in trace.")
        sys.exit(1)

    # 将同名的 function 根节点归并（通常只有一个）
    buckets = defaultdict(list)
    for f in funcs:
        buckets[(f.level, f.name)].append(f)

    # 假设只有一个函数名，否则可改为 ROOT 包裹
    lvl, name = next(iter(buckets))
    merged = Node(lvl, name)
    merged.time = sum(f.time for f in buckets[(lvl,name)])
    for f in buckets[(lvl,name)]:
        merged.children.extend(f.children)

    # 递归合并子节点
    merged = aggregate_node(merged)

    # 打印最终树
    print(merged)

if __name__ == "__main__":
    main()
