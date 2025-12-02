from __future__ import annotations
import os, re

def _strip_hash(v: str) -> str:
    if v is None: return ''
    s = str(v).strip()
    if s.startswith('#'): s = s[1:].strip()
    return s

def _parse_bool(s: str) -> bool:
    if s is None: return False
    return str(s).strip().upper() in {'Y','YES','TRUE','T','1'}

def parse_control(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    data = {}
    program_main_lines = []   # lines from "PROGRAM:"
    program_part_lines = []   # lines from "PROGRAM_1:", "PROGRAM_2:", etc.

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            if line.startswith('- '):
                line = line[2:].strip()
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip().upper()
            val = _strip_hash(val)

            if key == 'PROGRAM':
                program_main_lines.append(val)
            elif key.startswith('PROGRAM_'):
                program_part_lines.append(val)
            else:
                data[key] = val

    # booleans
    data['USE_CSV_BOOL'] = _parse_bool(data.get('USE_CSV')) if 'USE_CSV' in data else False
    data['DEPTH_LIMIT_ENABLED_BOOL'] = _parse_bool(data.get('DEPTH_LIMIT_ENABLED')) if 'DEPTH_LIMIT_ENABLED' in data else False
        # NEW: optional span sweep + one-way options
    if 'SPAN_SWEEP_FROM_MIN' in data:
        data['SPAN_SWEEP_FROM_MIN_BOOL'] = _parse_bool(data.get('SPAN_SWEEP_FROM_MIN'))
    else:
        data['SPAN_SWEEP_FROM_MIN_BOOL'] = False

    if 'ONE_WAY_IRREGULAR' in data:
        data['ONE_WAY_IRREGULAR_BOOL'] = _parse_bool(data.get('ONE_WAY_IRREGULAR'))
    else:
        data['ONE_WAY_IRREGULAR_BOOL'] = False


    # choose which program definitions to use:
    # - if any PROGRAM lines present -> use ONLY those
    # - else use PROGRAM_n lines
    chosen = program_main_lines if program_main_lines else program_part_lines

    def parse_ranges(entries):
        blocks = []
        seen = set()
        for entry in entries:
            parts = [p.strip() for p in entry.split(';') if p.strip()]
            for p in parts:
                if '=' in p:
                    # key=value form: start=,end=,use=
                    kvs = dict((kv.split('=')[0].strip().lower(), kv.split('=')[1].strip())
                               for kv in p.split(',') if '=' in kv)
                    try:
                        sf = int(float(kvs.get('start')))
                        ef = int(float(kvs.get('end')))
                        use = kvs.get('use')
                    except Exception:
                        continue
                    keyt = (sf, ef, use)
                    if use and keyt not in seen:
                        seen.add(keyt); blocks.append({'start_floor': sf, 'end_floor': ef, 'use': use})
                else:
                    # "1-5 Office_US"
                    m = re.match(r'\s*(\d+)\s*[-–]\s*(\d+)\s+(.+)$', p)
                    if m:
                        sf = int(m.group(1)); ef = int(m.group(2)); use = m.group(3).strip()
                        keyt = (sf, ef, use)
                        if keyt not in seen:
                            seen.add(keyt); blocks.append({'start_floor': sf, 'end_floor': ef, 'use': use})
        return blocks

    blocks = parse_ranges(chosen)
    if blocks:
        data['PROGRAM_BLOCKS'] = blocks

    return data
