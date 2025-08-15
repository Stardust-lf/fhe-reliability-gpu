#!/usr/bin/env python3
import re
import sys

def usage():
    print(f"Usage: {sys.argv[0]} <logfile>")
    sys.exit(1)

def main():
    if len(sys.argv) != 2:
        usage()
    logfile = sys.argv[1]

    # 正则
    start_re     = re.compile(r'^frontend: ROTATE$')
    end_re       = re.compile(r'^frontend: ROTATE\[\d+ microseconds\]$')
    ntt_re       = re.compile(r'^\[NTT\] total cost\s+(\d+)\s+µs$')
    cost_line_re = re.compile(r'^(.*total cost\s+)(\d+)(\s+µs.*)$')

    in_block   = False
    start_line = ""
    end_line   = ""
    out_lines  = []
    seq_sum    = 0
    in_seq     = False

    with open(logfile, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')

            # 区块开始
            if start_re.match(line):
                in_block = True
                start_line = line
                out_lines.clear()
                in_seq = False
                seq_sum = 0
                continue

            # 区块结束
            if in_block and end_re.match(line):
                # 如果正处于 NTT 序列中，先输出合并行
                if in_seq:
                    out_lines.append(f"[NTT] total cost {seq_sum} µs")
                    in_seq = False
                    seq_sum = 0
                end_line = line
                # 打印整个区块
                print(start_line)
                for l in out_lines:
                    print(l)
                print(end_line)
                in_block = False
                continue

            # 区块内处理
            if in_block:
                m_ntt = ntt_re.match(line)
                if m_ntt:
                    # 累加 NTT 耗时，不输出
                    cost = int(m_ntt.group(1))
                    if not in_seq:
                        in_seq = True
                        seq_sum = cost
                    else:
                        seq_sum += cost
                else:
                    # 碰到非 NTT 行
                    if in_seq:
                        # 先输出合并后的 NTT 行
                        out_lines.append(f"[NTT] total cost {seq_sum} µs")
                        # 再调整当前行的 cost
                        m_cost = cost_line_re.match(line)
                        if m_cost:
                            prefix, val, suffix = m_cost.group(1), int(m_cost.group(2)), m_cost.group(3)
                            adjusted = val - seq_sum
                            line = f"{prefix}{adjusted}{suffix}"
                        in_seq = False
                        seq_sum = 0
                    # 输出当前非 NTT 行
                    out_lines.append(line)

    # 如果文件结束还在区块中，则同样处理一次
    if in_block:
        if in_seq:
            out_lines.append(f"[NTT] total cost {seq_sum} µs")
        print(start_line)
        for l in out_lines:
            print(l)
        print(end_line)

if __name__ == '__main__':
    main()
