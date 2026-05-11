#!/usr/bin/env python3
"""
perf annotate 输出分析脚本
用法：
  1. 服务器上采集数据：
     perf record -e instructions:u -c 5000 /tmp/main_prof
     perf annotate --stdio > /tmp/annotate.txt

  2. 分析：
     python3 analyze_perf.py /tmp/annotate.txt

  3. 或 pip 到 stdin：
     perf annotate --stdio | python3 analyze_perf.py
"""

import sys
import re
from collections import Counter, defaultdict

# ─── 指令分类规则 ───────────────────────────────────────────

def classify_normal(inst, operands):
    """通用指令分类（不考虑上下文）"""
    # 分支
    if inst.startswith('b.') or inst in ('b', 'bl', 'blr', 'br', 'ret', 'cbz', 'cbnz', 'tbz', 'tbnz'):
        return '分支/循环'
    if inst == 'nop':
        return '其他(nop)'

    # SVE gather: ld1w 带向量索引操作数 [base, vec_reg, uxtw #2]
    if inst.startswith('ld1w') and re.search(r'z\d+\.s.*uxtw', operands):
        return '查表(gather)'
    if inst.startswith('ld1d') and re.search(r'z\d+\.d.*uxtw', operands):
        return '查表(gather)'

    # SVE 向量 load（顺序）
    if inst.startswith('ld1') and 'gather' not in inst:
        if 'b' in inst:
            return '加载A/B矩阵(u8)'
        if 'h' in inst:
            return '加载A/B矩阵(fp16)'
        return 'SVE向量load'

    # SVE 向量 store
    if inst.startswith('st1'):
        return 'SVE向量store'

    # SVE 谓词管理
    if inst.startswith('whilelt') or inst.startswith('whilele') or \
       inst.startswith('whilelo') or inst.startswith('whilels') or \
       inst.startswith('ptrue') or inst.startswith('ptest'):
        return 'SVE谓词管理'

    # SVE 归约
    if inst.startswith('faddv') or inst.startswith('saddv') or \
       inst.startswith('uaddv') or inst.startswith('addv'):
        return 'SVE归约'

    # SVE 向量 ALU
    if inst.startswith('fadd') and ('z' in inst or 'm' in inst):
        return 'SVE累加(fadd)'
    if inst == 'fadd':
        return 'SVE累加(fadd)'
    if inst.startswith('fmla') or inst.startswith('fmls'):
        return 'SVE乘加(fmla)'
    if inst.startswith('sdot') or inst.startswith('udot'):
        return 'SVE点积(sdot)'
    if inst.startswith('smmmla') or inst.startswith('ummla'):
        return 'SVE矩阵乘(smmmla)'
    if inst.startswith('fmul') or inst.startswith('fmul'):
        return 'SVE向量乘'
    if inst.startswith('lsl_z') or inst.startswith('lsl_m') or \
       inst.startswith('lsr_z') or inst.startswith('lsr_m') or \
       inst.startswith('asr_z') or inst.startswith('asr_m'):
        return 'SVE移位'
    if inst.startswith('orr_z') or inst.startswith('orr_m') or \
       inst.startswith('and_z') or inst.startswith('and_m'):
        return 'SVE位运算'
    if inst.startswith('add_z') or inst.startswith('add_m') or \
       inst.startswith('sub_z') or inst.startswith('sub_m'):
        return 'SVE向量ALU'

    # SVE 其他
    if inst.startswith('cntw') or inst.startswith('cntb') or \
       inst.startswith('cntd') or inst.startswith('cnth') or \
       inst.startswith('addvl') or inst.startswith('addpl') or \
       inst.startswith('decw') or inst.startswith('decb') or \
       inst.startswith('inch') or inst.startswith('incw') or \
       inst.startswith('index') or inst.startswith('dup'):
        return 'SVE辅助'

    # SVE movprfx / mov
    if inst == 'movprfx':
        return 'SVE辅助'
    if inst.startswith('sel') or inst.startswith('splice') or \
       inst.startswith('compact') or inst.startswith('clast'):
        return 'SVE辅助'
    if inst.startswith('zip') or inst.startswith('uzp') or \
       inst.startswith('rev') or inst.startswith('svext'):
        return 'SVE辅助'
    if inst.startswith('uunpk') or inst.startswith('sunpk') or \
       inst.startswith('punpk'):
        return 'SVE辅助'

    # 标量 Load
    if inst.startswith('ldr') or inst.startswith('ldp') or inst.startswith('ldur'):
        return '标量load(地址/栈)'
    if inst.startswith('ldrb'):
        return '加载A/B矩阵(u8)'
    if inst.startswith('ldrh'):
        return '标量load'
    if inst.startswith('ldrsw'):
        return '标量load(地址/栈)'

    # 标量 Store
    if inst.startswith('str') or inst.startswith('stp') or inst.startswith('stur'):
        return '标量store'

    # 标量 ALU
    if inst in ('add', 'sub', 'adds', 'subs', 'cmp', 'cmn'):
        return '标量ALU(地址/循环)'
    if inst in ('lsl', 'lsr', 'asr'):
        return '标量移位'
    if inst.startswith('mul') or inst.startswith('smull') or inst.startswith('umull'):
        return '标量乘法'
    if inst.startswith('csel') or inst.startswith('csinc') or inst.startswith('cset') or \
       inst.startswith('cinc') or inst.startswith('ccmp'):
        return '标量条件运算'
    if inst.startswith('sxt') or inst.startswith('uxt') or \
       inst.startswith('sbf') or inst.startswith('ubf') or \
       inst.startswith('movk') or inst.startswith('movn'):
        return '标量位操作'

    # 标量 Move / 地址加载
    if inst == 'mov':
        return '标量mov'
    if inst == 'fmov':
        return '标量fmov'
    if inst.startswith('adrp') or inst.startswith('adr'):
        return '标量地址加载'

    # 标量 FP
    if inst.startswith('fcvt'):
        return '标量类型转换'
    if inst.startswith('fadd') or inst.startswith('fsub'):
        return '标量FP运算'
    if inst.startswith('fcmp'):
        return '标量FP比较'

    # 原子操作
    if inst.startswith('cas') or inst.startswith('ldadd') or \
       inst.startswith('stadd') or inst.startswith('swp'):
        return '原子操作'

    return f'其他({inst})'


def classify_optimized(inst, operands):
    """针对 matmul_fp8_lookup_sve_optimized 的精细分类"""
    base = classify_normal(inst, operands)

    # 优化版在预处理阶段有 ld1b+uunpk 等 A_shifted 构建逻辑
    if base == 'SVE辅助' and inst.startswith('uunpk'):
        return '预处理(A_shifted)'
    if base == '其他(uunpk)':
        return '预处理(A_shifted)'
    if inst.startswith('ld1b') and 'z0.h' in operands:
        return '预处理(A_shifted)'
    if base == '加载A/B矩阵(u8)' and inst == 'ld1w':
        return '加载A矩阵(预处理后)'
    return base


def classify_i8mm(inst, operands):
    """针对 matmul_int8_i8mm_complete 的精细分类"""
    base = classify_normal(inst, operands)
    # I8MM 中 smmla 是关键
    if inst.startswith('smmmla') or inst.startswith('ummla'):
        return 'I8MM矩阵乘(smmmla)'
    return base


# ─── 解析 perf annotate 输出 ──────────────────────────────

def parse_perf_annotate(lines):
    """解析 perf annotate --stdio 输出，返回函数列表"""
    functions = []
    current_func = None
    current_lines = []

    for line in lines:
        # 匹配函数起始：address <function_name(...)>:
        m = re.match(r'^([0-9a-f]+)\s+<(.+?)>:', line)
        if m:
            # 保存上一个函数
            if current_func and current_lines:
                functions.append((current_func, current_lines))
            name = m.group(2).split('(')[0]  # 去掉参数部分
            current_func = name
            current_lines = [line]
            continue

        if current_func is not None:
            current_lines.append(line)

    # 最后一个函数
    if current_func and current_lines:
        functions.append((current_func, current_lines))

    return functions


def extract_instructions(func_lines):
    """从函数汇编行中提取 (百分比, 指令, 操作数)"""
    instructions = []
    for line in func_lines:
        line = line.strip()
        # 匹配: 百分比 | 地址: 指令 操作数
        m = re.match(r'^\s*([\d.]+)\s*[│|]\s*[0-9a-f]+:\s+(\S+)\s+(.*)', line)
        if m:
            pct = float(m.group(1))
            inst = m.group(2)
            operands = m.group(3).strip()
            instructions.append((pct, inst, operands))
    return instructions


# ─── 分析引擎 ──────────────────────────────────────────────

def analyze_func(name, instructions, classifier=classify_normal):
    """分析一个函数的指令构成"""
    total_pct = sum(pct for pct, _, _ in instructions)
    if total_pct == 0:
        return None

    by_inst = Counter()
    by_category = defaultdict(float)
    details = []

    for pct, inst, operands in instructions:
        by_inst[inst] += pct
        cat = classifier(inst, operands)
        by_category[cat] += pct
        details.append((pct, inst, operands, cat))

    return {
        'name': name,
        'total_pct': total_pct,
        'total_insts': len(instructions),
        'by_inst': by_inst,
        'by_category': dict(by_category),
        'details': details,
    }


def print_report(results, top_n_inst=10):
    """打印分析报告"""
    print(f"{'='*100}")
    print(f"perf annotate 动态指令分析报告")
    print(f"{'='*100}")

    for r in results:
        if r is None:
            continue
        print(f"\n{'─'*80}")
        print(f"函数: {r['name']}")
        print(f"采样占比: {r['total_pct']:.1f}%")
        print(f"不同指令种类: {len(r['by_inst'])}")
        print(f"指令总条数(静态): {r['total_insts']}")
        print(f"{'─'*80}")

        # 按类别汇总
        print(f"\n  指令类别分布:")
        print(f"  {'类别':<30} {'采样占比':<10}")
        print(f"  {'-'*42}")
        for cat, pct in sorted(r['by_category'].items(), key=lambda x: -x[1]):
            bar = '█' * int(pct / max(r['by_category'].values()) * 20)
            print(f"  {cat:<30} {pct:>6.1f}%  {bar}")

        print(f"\n  前{top_n_inst}条高频指令:")
        print(f"  {'指令':<15} {'采样占比':<10} {'类别':<25} {'操作数'}")
        print(f"  {'-'*80}")
        inst_cat_map = {}
        for pct, inst, operands, cat in r['details']:
            if inst not in inst_cat_map:
                inst_cat_map[inst] = (cat, operands[:40])
        for inst, pct in r['by_inst'].most_common(top_n_inst):
            cat, ops = inst_cat_map.get(inst, ('?', '?'))
            print(f"  {inst:<15} {pct:>6.1f}%  {cat:<25} {ops}")


def print_comparison(results):
    """横向对比所有函数的关键指标"""
    print(f"\n{'='*100}")
    print(f"函数横向对比")
    print(f"{'='*100}")

    # 提取关键类别
    key_cats = [
        '查表(gather)', '加载A/B矩阵(u8)', '加载A矩阵(预处理后)',
        'SVE累加(fadd)', 'SVE乘加(fmla)', 'SVE点积(sdot)',
        'I8MM矩阵乘(smmmla)', 'SVE归约', 'SVE谓词管理',
        '标量ALU(地址/循环)', '标量mov', '标量load(地址/栈)',
        '标量store', '分支/循环', '预处理(A_shifted)',
        'SVE矩阵乘(smmmla)',
    ]

    header = f"{'函数':<40}"
    for cat in key_cats:
        header += f" {cat[:8]:>8}"
    print(header)
    print('-' * len(header))

    for r in results:
        if r is None:
            continue
        cats = r['by_category']
        line = f"{r['name'][:38]:40}"
        for cat in key_cats:
            pct = cats.get(cat, 0)
            line += f" {pct:>7.1f}%"
        print(line)


# ─── 主入口 ────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    functions = parse_perf_annotate(lines)
    print(f"发现 {len(functions)} 个函数")

    # 目标函数列表（按兴趣程度排序）
    targets = {
        'matmul_fp8_lookup_scalar': classify_normal,
        'matmul_fp8_lookup_sve': classify_normal,
        'matmul_fp8_lookup_sve_optimized': classify_optimized,
        'dotProductSVE': classify_normal,
        'matmul_sve_fp16': classify_normal,
        'matmul_scalar': classify_normal,
        'matmul_int8_sve': classify_normal,
        'matmul_int8_i8mm_complete': classify_i8mm,
    }

    results = []
    for name, classifier in targets.items():
        # 尝试精确匹配和包含匹配
        func_lines = None
        matched_name = name
        for fn, fl in functions:
            if fn == name or fn.startswith(name + '(') or name in fn:
                func_lines = fl
                matched_name = fn
                break

        if func_lines is None:
            print(f"  [未找到] {name}")
            continue

        insts = extract_instructions(func_lines)
        if not insts:
            print(f"  [无指令] {name}")
            continue

        result = analyze_func(matched_name, insts, classifier)
        if result:
            results.append(result)
            print(f"  [OK] {name}: {result['total_insts']} 条指令, "
                  f"采样占比 {result['total_pct']:.1f}%")
        else:
            print(f"  [无样本] {name}: 采样占比 0% (loopCnt 可能太小)")

    # 打印报告
    print_report(results)
    print_comparison(results)

    # 打印原始指令清单
    print(f"\n{'='*100}")
    print("原始汇编指令明细")
    print(f"{'='*100}")
    for r in results:
        if r is None:
            continue
        print(f"\n── {r['name']} ──")
        for pct, inst, operands, cat in r['details']:
            cat_short = cat[:20]
            print(f"  {pct:>6.1f}%  {inst:<12} {operands:<40}  ← {cat_short}")


if __name__ == '__main__':
    main()
