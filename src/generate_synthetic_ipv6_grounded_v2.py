import os, json, math, random, zipfile, hashlib, shutil
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from PIL import Image
import ipaddress

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = "/mnt/data/synthetic_ipv6_grounded_v2"
IMG_DIR = os.path.join(OUT_DIR, "images")

# clean
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(IMG_DIR, exist_ok=True)

LABEL_COUNTS = {
    "Benign": 2237731,
    "Fuzzers": 33816,
    "Analysis": 2381,
    "Backdoor": 1226,
    "DoS": 5980,
    "Exploits": 42748,
    "Generic": 19651,
    "Reconnaissance": 17074,
    "Shellcode": 4659,
    "Worms": 158,
}
TOTAL = sum(LABEL_COUNTS.values())
LABELS = list(LABEL_COUNTS.keys())
P = np.array([LABEL_COUNTS[k] / TOTAL for k in LABELS], dtype=float)

N = 3000
MIN_PER_CLASS = 2  # ensures every class appears at least twice

NET_INTERNAL = ipaddress.IPv6Network("2001:db8:1::/48")
NET_EXTERNAL = ipaddress.IPv6Network("2001:db8:100::/40")
NET_LINKLOCAL = ipaddress.IPv6Network("fe80::/10")

COMMON_PORTS = [443,80,53,22,25,110,143,445,3389,8080,123]
APP_MAP = {80:"HTTP",443:"HTTPS",53:"DNS",22:"SSH",25:"SMTP",110:"POP3",143:"IMAP",445:"SMB",3389:"RDP",8080:"HTTP-ALT",123:"NTP"}
APP_LIST = ["HTTP","HTTPS","DNS","SSH","SMTP","OTHER"]
TRANSPORTS = ["TCP","UDP","ICMPv6","OTHER"]
DIRECTIONS = ["inbound","outbound","lateral"]

def rand_ipv6_in(net: ipaddress.IPv6Network) -> str:
    base = int(net.network_address)
    host_bits = net.max_prefixlen - net.prefixlen
    return str(ipaddress.IPv6Address(base + random.getrandbits(host_bits)))

def stable_hash_0_255(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(h, "big") % 256

def choose_transport(label: str) -> str:
    if label == "Benign":
        return random.choices(["TCP","UDP","ICMPv6"], [0.60,0.35,0.05])[0]
    if label == "Reconnaissance":
        return random.choices(["TCP","UDP","ICMPv6"], [0.40,0.20,0.40])[0]
    if label == "DoS":
        return random.choices(["UDP","TCP"], [0.65,0.35])[0]
    return random.choices(["TCP","UDP"], [0.75,0.25])[0]

def choose_direction(label: str) -> str:
    if label == "Benign":
        return random.choices(DIRECTIONS, [0.40,0.45,0.15])[0]
    if label == "Worms":
        return random.choices(["lateral","inbound"], [0.70,0.30])[0]
    return random.choices(DIRECTIONS, [0.70,0.20,0.10])[0]

def choose_dst_port(label: str) -> int:
    if label == "Reconnaissance":
        return random.choice(COMMON_PORTS) if random.random() < 0.55 else random.randint(1,65535)
    if label == "DoS":
        return random.choice([80,443,53,123,8080])
    if label in ["Backdoor","Shellcode"]:
        return random.choice([22,3389,445,4444,1337,6667])
    if label in ["Exploits","Fuzzers","Analysis","Generic","Worms"]:
        return random.choice(COMMON_PORTS + [21,23,69,161,1883,5900,27017])
    return random.choice(COMMON_PORTS)

def lognormal_clip(mu, sigma, lo, hi):
    x = np.random.lognormal(mean=mu, sigma=sigma)
    return float(np.clip(x, lo, hi))

def gen_flow_stats(label: str):
    if label == "Benign":
        dur = lognormal_clip(6.0, 0.9, 5, 120000)
        pk  = int(np.clip(np.random.lognormal(2.3,0.8), 1, 5000))
    elif label == "Reconnaissance":
        dur = lognormal_clip(5.0, 0.7, 2, 20000)
        pk  = int(np.clip(np.random.lognormal(1.0,0.6), 1, 150))
    elif label == "DoS":
        dur = lognormal_clip(4.0, 0.7, 1, 30000)
        pk  = int(np.clip(np.random.lognormal(4.3,0.7), 20, 50000))
    elif label == "Backdoor":
        dur = lognormal_clip(7.3, 0.8, 50, 600000)
        pk  = int(np.clip(np.random.lognormal(3.0,0.8), 5, 12000))
    else:
        dur = lognormal_clip(5.6, 0.8, 2, 120000)
        pk  = int(np.clip(np.random.lognormal(2.2,0.9), 2, 20000))

    bpp = float(np.clip(np.random.lognormal(4.7,0.7), 60, 2000))
    byt = int(np.clip(pk*bpp, 60, 2_000_000_000))
    pps = pk / max(dur/1000.0, 0.001)
    bps = byt / max(dur/1000.0, 0.001)
    return dur, pk, byt, pps, bps

def gen_ipv6_behavior(label: str):
    if label == "Fuzzers":
        ext, frag, nd = random.randint(3,10), random.randint(0,2), random.randint(0,1)
    elif label == "Exploits":
        ext, frag, nd = random.randint(1,6), random.randint(1,12), random.randint(0,1)
    elif label == "Reconnaissance":
        ext, frag, nd = random.randint(0,2), random.randint(0,1), random.randint(1,25)
    elif label == "DoS":
        ext, frag, nd = random.randint(0,2), random.randint(0,3), random.randint(0,2)
    else:
        ext, frag, nd = random.randint(0,2), random.randint(0,1), random.randint(0,2)

    hop_base = random.choices([64,128,255], [0.62,0.33,0.05])[0]
    hop = int(np.clip(np.random.normal(hop_base,3), 1, 255))
    flow_label = random.randint(0, 2**20 - 1)
    return ext, frag, nd, hop, flow_label

def gen_entropy(label: str):
    if label == "Benign":
        return float(np.clip(np.random.normal(5.2,1.0), 0.5, 8.0))
    if label == "Generic":
        return float(np.clip(np.random.normal(7.6,0.4), 2.0, 8.0))
    if label in ["Backdoor","Shellcode","Worms"]:
        return float(np.clip(np.random.normal(6.8,0.8), 1.0, 8.0))
    if label in ["DoS","Reconnaissance"]:
        return float(np.clip(np.random.normal(4.7,1.2), 0.5, 8.0))
    return float(np.clip(np.random.normal(6.0,1.0), 0.5, 8.0))

def vectorize_row(row: dict) -> np.ndarray:
    def scale_log(x, lo, hi):
        x = max(float(x), 0.0)
        v = (math.log1p(x) - math.log1p(lo)) / max((math.log1p(hi) - math.log1p(lo)), 1e-9)
        return int(np.clip(round(v*255), 0, 255))

    def scale_lin(x, lo, hi):
        v = (float(x) - lo) / max((hi - lo), 1e-9)
        return int(np.clip(round(v*255), 0, 255))

    feats = [
        scale_log(row["duration_ms"], 1, 600000),
        scale_log(row["packets"], 1, 50000),
        scale_log(row["bytes"], 60, 2_000_000_000),
        scale_log(row["pkts_per_s"], 0.01, 2_000_000),
        scale_log(row["bytes_per_s"], 0.01, 5_000_000_000),

        scale_lin(row["payload_entropy"], 0.5, 8.0),
        scale_lin(row["ipv6_ext_hdr_count"], 0, 10),
        scale_lin(row["ipv6_fragment_count"], 0, 12),
        scale_lin(row["icmpv6_nd_msgs"], 0, 25),
        scale_lin(row["ipv6_hop_limit"], 1, 255),

        stable_hash_0_255(row["src_ip"]),
        stable_hash_0_255(row["dst_ip"]),
        scale_lin(row["src_port"], 0, 65535),
        scale_lin(row["dst_port"], 0, 65535),

        scale_log(row["tcp_syn"], 0, 50000),
        scale_log(row["tcp_ack"], 0, 50000),
        scale_log(row["tcp_fin"], 0, 1000),
        scale_log(row["tcp_rst"], 0, 1000),
    ]
    fl = int(row["ipv6_flow_label"])
    feats += [(fl>>0)&0xFF, (fl>>8)&0xFF, (fl>>16)&0x0F, scale_lin(row["ipv6_next_header"],0,255)]

    t = row["transport"]; feats += [255 if t==x else 0 for x in TRANSPORTS]
    d = row["direction"]; feats += [255 if d==x else 0 for x in DIRECTIONS]
    a = row["app_proto"]; feats += [255 if a==x else 0 for x in APP_LIST]

    bpp = row["bytes"] / max(row["packets"], 1)
    feats += [scale_log(bpp, 40, 2000), scale_log(row["bytes"]/max(row["duration_ms"],1), 0.01, 1_000_000)]

    while len(feats) < 64:
        feats.append((stable_hash_0_255(row["record_id"]+str(len(feats))) + feats[len(feats)%len(feats)]) % 256)

    return np.array(feats[:64], dtype=np.uint8).reshape(8,8)

# Ensure all label folders exist
for lab in LABELS:
    os.makedirs(os.path.join(IMG_DIR, lab), exist_ok=True)

# Build label schedule with guaranteed minimum per class
rem = N - MIN_PER_CLASS * len(LABELS)
counts_extra = np.random.multinomial(rem, P)
counts = {lab: MIN_PER_CLASS + int(extra) for lab, extra in zip(LABELS, counts_extra)}

labels_schedule = []
for lab, c in counts.items():
    labels_schedule.extend([lab]*c)
random.shuffle(labels_schedule)

start = datetime(2026,1,1,tzinfo=timezone.utc)

records = []
index_rows = []

for i, label in enumerate(labels_schedule):
    transport = choose_transport(label)
    direction = choose_direction(label)

    if direction == "inbound":
        src_ip, dst_ip = rand_ipv6_in(NET_EXTERNAL), rand_ipv6_in(NET_INTERNAL)
    elif direction == "outbound":
        src_ip, dst_ip = rand_ipv6_in(NET_INTERNAL), rand_ipv6_in(NET_EXTERNAL)
    else:
        src_ip, dst_ip = rand_ipv6_in(NET_INTERNAL), rand_ipv6_in(NET_INTERNAL)

    if label == "Reconnaissance" and random.random() < 0.25:
        src_ip, dst_ip = rand_ipv6_in(NET_LINKLOCAL), rand_ipv6_in(NET_LINKLOCAL)

    dst_port = choose_dst_port(label)
    src_port = random.randint(1024,65535) if transport in ["TCP","UDP"] else 0
    app_proto = APP_MAP.get(dst_port, "OTHER") if transport in ["TCP","UDP"] else "OTHER"

    dur_ms, pk, byt, pps, bps = gen_flow_stats(label)
    ext, frag, nd, hop, flow_label = gen_ipv6_behavior(label)
    entropy = gen_entropy(label)

    tcp_syn=tcp_ack=tcp_fin=tcp_rst=0
    if transport == "TCP":
        if label == "DoS" and random.random() < 0.6:
            tcp_syn, tcp_ack = pk, 0
        else:
            tcp_syn = int(min(pk, random.randint(1,3)))
            tcp_ack = int(max(0, pk - tcp_syn))
        tcp_fin = int(np.random.binomial(n=min(pk,10), p=0.05))
        tcp_rst = int(np.random.binomial(n=min(pk,10), p=0.02))

    next_hdr = 6 if transport=="TCP" else 17 if transport=="UDP" else 58 if transport=="ICMPv6" else 59

    t0 = start + timedelta(seconds=int(random.random()*86400))
    t1 = t0 + timedelta(milliseconds=int(dur_ms))
    rid = f"R{i+1:07d}"

    row = {
        "record_id": rid,
        "label": label,
        "window_start_utc": t0.isoformat(),
        "window_end_utc": t1.isoformat(),
        "ip_version": 6,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": int(src_port),
        "dst_port": int(dst_port),
        "transport": transport,
        "app_proto": app_proto,
        "direction": direction,
        "duration_ms": round(dur_ms, 3),
        "packets": int(pk),
        "bytes": int(byt),
        "pkts_per_s": float(pps),
        "bytes_per_s": float(bps),
        "tcp_syn": int(tcp_syn),
        "tcp_ack": int(tcp_ack),
        "tcp_fin": int(tcp_fin),
        "tcp_rst": int(tcp_rst),
        "ipv6_flow_label": int(flow_label),
        "ipv6_next_header": int(next_hdr),
        "ipv6_hop_limit": int(hop),
        "ipv6_ext_hdr_count": int(ext),
        "ipv6_fragment_count": int(frag),
        "icmpv6_nd_msgs": int(nd),
        "payload_entropy": float(entropy),
    }
    records.append(row)

    img_arr = vectorize_row(row)
    img_rel = f"images/{label}/{rid}.png"
    img_abs = os.path.join(OUT_DIR, img_rel)
    Image.fromarray(img_arr).save(img_abs)
    index_rows.append({"record_id": rid, "label": label, "image_path": img_rel})

df = pd.DataFrame(records)
idx = pd.DataFrame(index_rows)

# Stratified split without sklearn
train_parts, val_parts, test_parts = [], [], []
for lab, g in idx.groupby("label"):
    g = g.sample(frac=1.0, random_state=SEED)
    n = len(g)
    n_train = int(round(n*0.70))
    n_val = int(round(n*0.15))
    train_parts.append(g.iloc[:n_train].assign(split="train"))
    val_parts.append(g.iloc[n_train:n_train+n_val].assign(split="val"))
    test_parts.append(g.iloc[n_train+n_val:].assign(split="test"))
splits = pd.concat(train_parts+val_parts+test_parts, ignore_index=True)

df.to_csv(os.path.join(OUT_DIR, "flows.csv"), index=False)
splits.to_csv(os.path.join(OUT_DIR, "images_index.csv"), index=False)
splits.query("split=='train'").to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
splits.query("split=='val'").to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
splits.query("split=='test'").to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

# Evidence table
dist_real = pd.DataFrame({"class": LABELS, "nf_unsw_nb15_v3_count": [LABEL_COUNTS[k] for k in LABELS]})
dist_real["nf_unsw_nb15_v3_pct"] = (dist_real["nf_unsw_nb15_v3_count"] / TOTAL * 100).round(2)

dist_synth = df["label"].value_counts().reindex(LABELS).reset_index()
dist_synth.columns = ["class", "synthetic_v2_count"]
dist_synth["synthetic_v2_pct"] = (dist_synth["synthetic_v2_count"] / N * 100).round(2)

dist_compare = dist_real.merge(dist_synth, on="class")
dist_compare["abs_pct_diff"] = (dist_compare["synthetic_v2_pct"] - dist_compare["nf_unsw_nb15_v3_pct"]).abs().round(2)

# Data card + pipeline proof
datacard = {
    "name": "Synthetic IPv6 Intrusion Dataset (Grounded) v2",
    "version": "2.0",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "size_records": N,
    "min_per_class": MIN_PER_CLASS,
    "labels": LABELS,
    "files": {
        "flows_csv": "flows.csv",
        "images_index_csv": "images_index.csv",
        "splits": ["train.csv", "val.csv", "test.csv"],
        "images_dir": "images/<label>/<record_id>.png",
    },
    "grounding_notes": [
        "Label priors come from NF-UNSW-NB15-v3 class counts (published).",
        "UNSW-NB15 lineage is described as a hybrid dataset: real modern normal activities + synthetic attack behaviours.",
        "Prior research shows how flow/tabular records (e.g., UNSW-NB15) can be transformed into images for CNN-based intrusion detection.",
    ],
    "important_disclaimer": "Synthetic data is intended for research prototyping and reproducible pipeline proof; it is not an identical replica of NF-UNSW-NB15 or UNSW-NB15.",
}
with open(os.path.join(OUT_DIR, "datacard.json"), "w") as f:
    json.dump(datacard, f, indent=2)

pipeline_md = """# Pipeline proof — Synthetic IPv6 Intrusion Dataset (Grounded) v2

## Evidence-based priors
1) Use published NF-UNSW-NB15-v3 class counts to define label probabilities.
2) Generate class-conditioned IPv6 flow behaviour (ports, protocols, rates, and IPv6 features such as extension headers, fragmentation, and ND activity).
3) Convert each record into an 8×8 grayscale image using a deterministic feature→pixel mapping (no label leakage).
4) Create stratified train/val/test splits and export CSV + image folders.

Outputs:
- flows.csv
- images/<label>/<record_id>.png
- images_index.csv, train.csv, val.csv, test.csv
- datacard.json
"""
with open(os.path.join(OUT_DIR, "PIPELINE_PROOF.md"), "w") as f:
    f.write(pipeline_md)

# Zip
zip_path = "/mnt/data/synthetic_ipv6_grounded_v2.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, _, files in os.walk(OUT_DIR):
        for fn in files:
            p = os.path.join(root, fn)
            arc = os.path.relpath(p, os.path.dirname(OUT_DIR))
            z.write(p, arcname=arc)

zip_path, df["label"].value_counts().reindex(LABELS)
