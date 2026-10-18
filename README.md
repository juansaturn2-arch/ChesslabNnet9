# ChesslabNNet9

**ChesslabNNet9** is an original UCI chess engine by **K7**, part of the K7 ChessLab Collection.
It combines a modern alpha-beta search with its own NNUE network, trained from scratch on
the K7NN format. The net and an opening book are built into a single `.exe`, so there are no extra files to install.

> Early 4-CPU tests (300+3) put it at roughly **3350–3450 Elo**. More testing is welcome.

---

## Highlights

- **Own NNUE net:** 768 → 768×2 (SCReLU) → 1, trained on **533 million positions** for 35 epochs, with an AVX2 inference path.
- **Single exe:** the net and the Polyglot opening book are embedded. Run it from any folder.
- **Lazy SMP** multithreading (up to 64 threads).
- **Search:** iterative deepening with aspiration windows, PVS, a 4-way bucketed lockless transposition table,
  staged move generation, null move, reverse futility, futility and late-move pruning, LMR, SEE pruning,
  singular extensions, IIR, killer/counter/history/continuation/capture history, and correction history (pawn hash).
- **Time manager:** soft and hard limits, plus *Move Overhead* for GUI and network lag.
- **Opening book:** reads any Polyglot `.bin`. Choose *Best* (top-weight move) or *Weighted* (random by weight).
- **Search tunables** are exposed as UCI options, ready for SPSA tuning.

---

## Quick start

1. Download `ChesslabNNet9.exe`.
2. Add it to your GUI (Arena, Cute Chess, Banksia, …) as a **UCI** engine.
3. Set **Threads** and **Hash** to suit your machine and play.

The release exe is built for **x86-64 CPUs with AVX2** (Intel Haswell 2013+ / AMD Ryzen).
If it won't start on an older CPU, build it from source (see below).

---

## UCI options

| Option | Default | Description |
|---|---|---|
| Hash | 256 | Transposition table size in MB (1–4096) |
| Threads | 1 | Search threads (1–64), Lazy SMP |
| Move Overhead | 30 | Milliseconds reserved per move for GUI/network lag |
| NetFile | `<embedded>` | Built-in net, or a path to another K7NN `.bin` net |
| OwnBook | true | Use the engine's own opening book |
| BookFile | `<embedded>` | Built-in book, or a path to any Polyglot `.bin` |
| BookMaxPly | 24 | Stop using the book after this many half-moves (0–200) |
| BookPick | Best | `Best` = highest-weight move, `Weighted` = random by weight |
| Fifty | true | Scale the eval toward a draw as the 50-move counter rises |
| TimeManager | true | Soft/hard time limits (off = fixed share of the clock) |

**Search switches** (all `true` by default; for testing only):
RFP, NullMove+, Futility, LMP, LMR+, SEE, Delta, IIR, Ordering+, QSearchTT, TTEval, Singular,
ContHist, CorrHist, CapHist.

**Search tunables** (spin values used for SPSA tuning; leave at default for normal play):
RfpMargin, RfpImproving, RfpDepth, NmpBase, NmpDepthDiv, NmpEvalDiv, FutBase, FutMult, FutDepth,
LmpBase, LmrBase, LmrDiv, SeeMargin, SeeDepth, SingDepth, SingMargin, AspWindow, HistLmrDiv.

> **Testers:** for rating matches, either turn **OwnBook off** and use the GUI's opening set,
> or give both engines the same book. Otherwise the book's opening advantage skews the result.

---

## Extra commands

Type these in a console or send them through the GUI's engine log:

| Command | What it does |
|---|---|
| `bench [depth]` | Fixed-depth search of 14 positions (1 thread). Prints nodes and nps. |
| `perft n` | Move-generator test from the current position |
| `eval` | Prints the NNUE evaluation of the current position |
| `go nodes N` | Fixed-node search (compare nets without speed bias) |

Self-check: from the start position, `perft 5` = **4865609** and `bench 9` = **536905 nodes**.

---

## Building from source

ChesslabNNet9 is a single C file. Put these three files in one folder:

```
ChesslabNNet9.c
k7net.bin          (the 533M / 768 net)
k7Opennings.bin    (Polyglot book; rename any .bin to use it instead)
```

**Windows (MSYS2 MinGW64):** run `build_nnet9.bat`, or:

```
gcc   -O3 -march=native -static -Wa,-muse-unaligned-vector-move -o ChesslabNNet9.exe ChesslabNNet9.c -lm
clang -O3 -march=native -static -o ChesslabNNet9.exe ChesslabNNet9.c
```

The `-Wa,-muse-unaligned-vector-move` flag avoids a MinGW gcc AVX stack-alignment crash; clang doesn't need it.

**Linux:**

```
gcc -O3 -march=native -pthread -o ChesslabNNet9 ChesslabNNet9.c -lm
```

**Build options:**

- `-DK7_NET_FILE="path/net.bin"` / `-DK7_BOOK_FILE="path/book.bin"`: embed files from another path.
- `-DK7_NO_EMBED`: build without embedded files (load them with NetFile / BookFile).
- **PGO** (5–15% faster): build with `-fprofile-generate`, run `ChesslabNNet9.exe bench 10`,
  then rebuild with `-fprofile-use -fprofile-correction`.

---

## Lineage

ChesslabNNet9 is the final engine of the **K7CustomNet** line (V1 → V2 → V3 → V4). It uses the V4 search,
which is about 1.55× faster than V3 and scored about +152 Elo against it in head-to-head tests, together with the
new 533M-position net. The handcrafted eval and older "brain" nets from earlier ChesslabNnet versions have been removed.

---

## Author

**K7**, K7 ChessLab Collection.
Bug reports, test results and PGNs are welcome through GitHub Issues.
