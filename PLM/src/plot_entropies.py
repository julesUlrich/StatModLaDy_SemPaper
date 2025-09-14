# scripts/plot_from_entropy_csv.py
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

# plt.style.use('seaborn-v0_8-whitegrid')  # Modern, clean style

SPECIAL = {"[PAD]", "[UNK]", "[BOS]", "[EOS]"}


def plot_sentence(group, out_path, show_mean=False):
    # group: DataFrame for one sentence, must contain columns:
    # phoneme, entropy, position, speaker, conversation, conversation_sentence_id, orthography (optional), global_id (optional)
    g = group.sort_values("position")
    g = g[~g["phoneme"].isin(SPECIAL)]
    if g.empty:
        return

    ent = g["entropy"].tolist()
    ph  = g["phoneme"].tolist()

    line_colour = "#0D47A1"
    mean_colour = "#FF9A6B"

    plt.figure(figsize=(12,6), dpi=600)
    plt.plot(range(len(ent)), ent, marker="o", linewidth=1.5, color=line_colour, zorder=3)
    if show_mean:
        m = sum(ent)/len(ent)
        plt.axhline(m, color=mean_colour, linestyle="--", alpha=1, linewidth=1.5, label=f"Mean: {m:.2f}", zorder=2)
        plt.legend()

    plt.xticks(range(len(ph)), ph, rotation=45, ha="right")
    spk = g["speaker"].iloc[0] if "speaker" in g else "NA"
    conv = g["conversation"].iloc[0] if "conversation" in g else "NA"
    sid = g["conversation_sentence_id"].iloc[0] if "conversation_sentence_id" in g else 0
    plt.title(f"Entropy Progression - Speaker: {spk}, Conv: {conv}, Sentence: {sid}")
    plt.xlabel("Phoneme")
    plt.ylabel("Entropy (bits)")
    plt.gca().set_facecolor('#F0F0F0')
    plt.grid(True, alpha=1, linestyle='-', linewidth=0.8, color='white', zorder=1)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--speaker")
    ap.add_argument("--conversation")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--show_mean", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    # Expect columns from evaluate script
    needed = {"speaker","conversation","conversation_sentence_id","position","phoneme","entropy"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")

    # Optional filter by speaker/conversation
    if args.speaker:
        df = df[df["speaker"] == args.speaker]
    if args.conversation:
        df = df[df["conversation"] == args.conversation]

    # Group by sentence to match evaluate plots
    groups = df.groupby(["speaker","conversation","conversation_sentence_id"], sort=True)
    count = 0
    for (spk, conv, sid), g in groups:
        out_name = f"entropy_sentence_{spk}_{conv}_{int(sid):04d}.png"
        out_path = os.path.join(args.out_dir, out_name)
        plot_sentence(g, out_path, show_mean=args.show_mean)
        count += 1
        if args.limit and count >= args.limit:
            break

if __name__ == "__main__":
    main()