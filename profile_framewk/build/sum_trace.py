#!/usr/bin/env python3
import re
import sys
from collections import defaultdict

def usage():
    print(f"Usage: {sys.argv[0]} <logfile>")
    sys.exit(1)

def main():
    if len(sys.argv) != 2:
        usage()
    logfile = sys.argv[1]

    # Regex patterns
    start_re     = re.compile(r'^frontend: ROTATE$')
    end_re       = re.compile(r'^frontend: ROTATE\[(\d+)\s+microseconds\]$')
    ntt_re       = re.compile(r'^\[NTT\] total cost\s+(\d+)\s+µs$')
    cost_re      = re.compile(r'^\[([^\]]+)\] total cost\s+(\d+)\s+µs')

    # Accumulators
    costs = defaultdict(int)
    rotate_sum = 0

    in_block = False
    in_seq = False
    seq_sum = 0

    with open(logfile, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            # Enter ROI block
            if start_re.match(line):
                in_block = True
                in_seq = False
                seq_sum = 0
                continue

            # Exit ROI block
            m_end = end_re.match(line)
            if in_block and m_end:
                # Collapse any pending NTT sequence
                if in_seq:
                    costs['NTT'] += seq_sum
                    in_seq = False
                    seq_sum = 0
                # Accumulate rotate time
                rotate_sum += int(m_end.group(1))
                in_block = False
                break

            if not in_block:
                continue

            # Inside ROI block
            m_ntt = ntt_re.match(line)
            if m_ntt:
                # Accumulate NTT into sequence
                seq_sum += int(m_ntt.group(1))
                in_seq = True
                continue

            m_cost = cost_re.match(line)
            if m_cost:
                tag, val = m_cost.group(1), int(m_cost.group(2))
                if in_seq:
                    # Add the collapsed NTT time first
                    costs['NTT'] += seq_sum
                    # Self time for this tag
                    self_time = val - seq_sum
                    costs[tag] += self_time
                    # Reset sequence
                    in_seq = False
                    seq_sum = 0
                else:
                    costs[tag] += val
                continue

    # Compute proportions and output
    total_ops = sum(costs.values())
    if rotate_sum > 0:
        print(f"[frontend: ROTATE] total: {rotate_sum} µs")
        print(f"Overall operations total: {total_ops} µs ({total_ops/rotate_sum:.4f} of rotate)")
        print()
        for tag, total in sorted(costs.items()):
            proportion = total / rotate_sum
            print(f"[{tag}] total: {total} µs, proportion of rotate: {proportion:.4f} ({proportion*100:.2f}%)")
        # Also NTT share of operations
        if 'NTT' in costs:
            prop_ops = costs['NTT'] / total_ops if total_ops else 0
            print(f"\n[NTT] share of operations: {prop_ops:.4f} ({prop_ops*100:.2f}%)")
    else:
        print("No ROTATE block found or rotate time is zero.")

if __name__ == "__main__":
    main()

