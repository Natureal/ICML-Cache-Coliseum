import csv, glob, os

TARGETS = {
    'RPB-pb-0':     'RPB-OnlineMin[LRB]-msf-100-pb-0',
    'RPB-pb-1':     'RPB-OnlineMin[LRB]-msf-100-pb-1',
    'RPB-pb-2':     'RPB-OnlineMin[LRB]-msf-100-pb-2',
    'RPB-pb-4':     'RPB-OnlineMin[LRB]-msf-100-pb-4',
    'CombDet[LRU]': 'CombDet[Predict[LRB], LRU]',
}

rows = []
for path in sorted(glob.glob('stat/*_lrb_1.csv')):
    ds = os.path.basename(path).replace('_lrb_1.csv', '')
    hr = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            hr[r['Name']] = float(r['Hit Rate'])
    rec = {'dataset': ds}
    for label, name in TARGETS.items():
        rec[label] = hr.get(name)
    rows.append(rec)

labels = list(TARGETS.keys())
ref = 'CombDet[LRU]'
others = [l for l in labels if l != ref]

hdr = ['dataset'] + labels + [f'{o}-vs-CombDet' for o in others]
widths = [max(len(h), 14) for h in hdr]

def fmt(vals):
    out = []
    for v, w in zip(vals, widths):
        if isinstance(v, str):
            out.append(v.ljust(w))
        else:
            out.append(str(v).rjust(w))
    return ' | '.join(out)

print(fmt(hdr))
print('-+-'.join('-' * w for w in widths))
for r in rows:
    line = [r['dataset']]
    for l in labels:
        line.append(f'{r[l]:.4f}' if r[l] is not None else 'N/A')
    for o in others:
        if r[o] is not None and r[ref] is not None:
            line.append(f'{r[o] - r[ref]:+.4f}')
        else:
            line.append('N/A')
    print(fmt(line))
print('-+-'.join('-' * w for w in widths))

mean_line = ['MEAN']
for l in labels:
    vals = [r[l] for r in rows if r[l] is not None]
    mean_line.append(f'{sum(vals)/len(vals):.4f}')
for o in others:
    vals = [r[o] - r[ref] for r in rows if r[o] is not None and r[ref] is not None]
    mean_line.append(f'{sum(vals)/len(vals):+.4f}')
print(fmt(mean_line))
