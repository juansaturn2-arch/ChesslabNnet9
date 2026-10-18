/*
 * ChesslabNNet9 -- by K7
 * K7CustomNetV4 search + embedded K7 NNUE + embedded Polyglot opening book
 * 10x12 Mailbox + PVS/LMR/Null-move + Lazy SMP + UCI
 *
 * One exe, no extra files needed:
 *   - the K7 net (768 -> 768x2 -> 1, 533M positions, 35 epochs) is built in;
 *     NetFile = <embedded> uses it, or set NetFile to any K7NN .bin/.k7nn
 *   - the k7Opennings.bin Polyglot book is built in;
 *     OwnBook (on/off), BookFile (<embedded> or a path),
 *     BookMaxPly (how many plies from the start to use it, default 24),
 *     BookPick (Best = strongest book move, Weighted = random by weight)
 *     The book is never used in "go infinite" (analysis).
 *
 * Build: k7net.bin and k7Opennings.bin must sit next to this .c file
 * (the assembler copies them into the exe with .incbin):
 *   g++ -O3 -march=native -static -Wa,-muse-unaligned-vector-move -o ChesslabNNet9.exe ChesslabNNet9.c
 *   clang -O3 -march=native -static -o ChesslabNNet9.exe ChesslabNNet9.c
 * Files somewhere else: add -DK7_NET_FILE="\"D:/path/k7net.bin\"" (same for K7_BOOK_FILE)
 * No embedding at all: -DK7_NO_EMBED (then set NetFile/BookFile in the GUI)
 *
 * SEARCH (from K7CustomNetV4)
 *  Speed (about 1.5x the nodes per second of V3, same net):
 *   - sentinel board: off-board squares hold OFFBOARD, so sliding pieces stop
 *     on their own (V3 spent ~25% of its time in sq_valid())
 *   - piece lists: move generation visits only the pieces, not 120 squares
 *   - hash move tried before any move generation (staged), captures-only
 *     generation in quiescence
 *   - legality test after makeMove only when a move can expose the king;
 *     cheap "gives check" pre-test; SEE skipped when taking an equal or
 *     bigger piece
 *   - 4-way bucketed transposition table (64-byte clusters)
 *  Search (UCI check boxes, all on by default):
 *   - CorrHist    correction history: learns where the net's eval is off for
 *                 a pawn structure and corrects it on the fly
 *   - CapHist     capture history for ordering captures
 *   - TimeManager soft/hard time limits; thinks longer when the best move
 *                 keeps changing or the score drops, shorter when stable;
 *                 helper threads keep searching until the main thread stops
 *   - "Move Overhead" (ms reserved per move for GUI/OS lag, default 30)
 *  Tuning:
 *   - every hand-picked search number is a UCI spin option (RfpMargin,
 *     NmpBase, FutMult, LmrDiv, SeeMargin, SingMargin, AspWindow, ...);
 *     defaults = V3. Tune them with k7_spsa.py.
 *  Perft is exact. Switching CorrHist/CapHist/TimeManager off removes the
 *  new search features for A/B tests; the speed changes always stay on.
 *
 * FROM V3
 *  Speed (identical evals -- SIMD and scalar give the exact same numbers):
 *   - AVX2 SCReLU output layer with the mullo/madd trick (v*w fits int16,
 *     then madd with v): about 2x faster output layer, matters most at 1024
 *   - explicit AVX2 accumulator updates, 64-byte aligned weights/accumulators
 *   - accumulator stack sized to the search (256 plies), not the whole game;
 *     the root accumulator is built once per search so every node below is
 *     a cheap delta update (V2 could fall back to full refreshes)
 *   - measured (i9-class AVX2): output layer 1024 = 80 ns vs V2 1024 = 293 ns
 *     and V2 512 = 144 ns -> a 1024 net now evaluates faster than V2's 512
 *   - wall-clock timing (QueryPerformanceCounter), safe with threads
 *   - "ChesslabNNet9.exe bench" from the command line, for PGO builds
 *  Threads:
 *   - Lazy SMP: "Threads" option (1..64). Use 8 on an 8-core CPU.
 *     Shared lockless TT (XOR-verified), shared lockless eval cache.
 *  Search (each one is a UCI check box, all on by default):
 *   - QSearchTT  transposition table in quiescence
 *   - TTEval     TT score sharpens the static eval used for pruning
 *   - Singular   singular extensions + multi-cut
 *   - ContHist   continuation history (1 and 2 plies back)
 *   - Fifty      eval shrinks as the 50-move counter grows: a winning
 *                side keeps making progress instead of shuffling
 *   - TT aging: entries from old searches are replaced first
 *   V2 boxes (RFP, NullMove+, Futility, LMP, LMR+, SEE, Delta, IIR,
 *   Ordering+) are unchanged.
 *  Fixes:
 *   - PV output: about 1 in 4 PVs in V1/V2 ended in garbage moves (stale
 *     PV copied after TT cutoffs). Display only -- moves played were fine.
 *  Tools:
 *   - "go nodes N"     fixed-node search (compare nets without speed bias)
 *   - "bench [depth]"  fixed-depth search of 14 positions (1 thread)
 *   - "perft n", "eval", stop/quit/isready during search
 *
 * Compile (MSYS2 MinGW64, pick one):
 *   clang -O3 -march=native -static -o ChesslabNNet9.exe ChesslabNNet9.c
 *   gcc   -O3 -march=native -static -Wa,-muse-unaligned-vector-move -o ChesslabNNet9.exe ChesslabNNet9.c
 *   (the -Wa flag avoids a MinGW gcc AVX stack-alignment crash; clang does not need it)
 *   Linux: add -pthread
 *
 * PGO (extra 5-15% speed), gcc:
 *   gcc -O3 -march=native -static -Wa,-muse-unaligned-vector-move -fprofile-generate -o ChesslabNNet9.exe ChesslabNNet9.c
 *   ChesslabNNet9.exe bench 10
 *   gcc -O3 -march=native -static -Wa,-muse-unaligned-vector-move -fprofile-use -fprofile-correction -o ChesslabNNet9.exe ChesslabNNet9.c
 *
 * Verify : ChesslabNNet9.exe   then type:  perft 5   (expect 4865609)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <limits.h>
#include <stdint.h>

#if defined(__AVX2__) && !defined(K7_NO_SIMD)
#  include <immintrin.h>
#  define K7_AVX2 1
#else
#  define K7_AVX2 0
#endif

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/select.h>
#  include <sys/time.h>
#  include <unistd.h>
#  include <pthread.h>
#endif

#define MAX_PLY         64
#define MAX_MOVES       256
#define INF_SCORE       100000
#define MATE_SCORE      90000
#define TT_DEFAULT_MB   256
#define TT_MAX_MB       4096
#define SCORE_NONE      (-INT_MAX)
#define HIST_SIZE       2048
#define ACC_PLIES       256          /* accumulator stack depth per thread */
#define MAX_THREADS     64
#define THREAD_STACK    (16u * 1024u * 1024u)
#define START_FEN       "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

#define TT_EXACT 0
#define TT_ALPHA 1
#define TT_BETA  2

#define QUIET            0
#define DOUBLE_PUSH      1
#define KING_CASTLE      2
#define QUEEN_CASTLE     3
#define CAPTURE          4
#define EP_CAPTURE       5
#define KNIGHT_PROMO     8
#define BISHOP_PROMO     9
#define ROOK_PROMO      10
#define QUEEN_PROMO     11
#define KNIGHT_PROMO_CAP 12
#define BISHOP_PROMO_CAP 13
#define ROOK_PROMO_CAP   14
#define QUEEN_PROMO_CAP  15

#define IS_PROMO(t)   ((t) >= KNIGHT_PROMO)
#define IS_CAP(t)     ((t) == CAPTURE || (t) == EP_CAPTURE || (t) >= KNIGHT_PROMO_CAP)

/* 32-bit move: (from:7 | to:7 | type:4) */
typedef unsigned int MOVE;
typedef unsigned long long U64;

#define MV(f,t,ty)  (((MOVE)(f) << 11) | ((MOVE)(t) << 4) | (MOVE)(ty))
#define MV_FROM(m)  (((m) >> 11) & 0x7F)
#define MV_TO(m)    (((m) >> 4)  & 0x7F)
#define MV_TYPE(m)  ((m) & 0x0F)

enum {
    A1=21, B1, C1, D1, E1, F1, G1, H1,
    A2=31, B2, C2, D2, E2, F2, G2, H2,
    A3=41, B3, C3, D3, E3, F3, G3, H3,
    A4=51, B4, C4, D4, E4, F4, G4, H4,
    A5=61, B5, C5, D5, E5, F5, G5, H5,
    A6=71, B6, C6, D6, E6, F6, G6, H6,
    A7=81, B7, C7, D7, E7, F7, G7, H7,
    A8=91, B8, C8, D8, E8, F8, G8, H8
};

enum { UP = 10, DOWN = -10 };
enum { WKCA=8, WQCA=4, BKCA=2, BQCA=1 };
enum { WHITE, BLACK };
enum { EMPTY, wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK, OFFBOARD };
#define PLIST_MAX 10
enum { PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };   /* base types */

#define IS_BLACK(p) ((p) > 6)
#define COLOR_OF(p) (IS_BLACK(p) ? BLACK : WHITE)
#define BASE(p)     (IS_BLACK(p) ? (p) - 6 : (p))
#define MAKE_PC(side, base) ((base) + ((side) == BLACK ? 6 : 0))

typedef struct {
    MOVE move;
    int  enPas;
    int  castlePermission;
    int  capturedPiece;
    int  fiftyMove;
    U64  hash;
    U64  pawnHash;
    /* net feature delta of this move (kings included) */
    signed char   nRem, nAdd;
    unsigned char remP[2], remS[2], addP[2], addS[2];
} MOVE_UNDO;

typedef struct {
    int board[120];
    int hisPly;          /* total plies from game start -- indexes history[] */
    int stm;
    int enPas;
    int castlePermission;
    int fiftyMove;
    int kingSquare[2];
    int pcount[13];      /* number of each piece type on the board */
    unsigned char plist[13][PLIST_MAX]; /* squares of each piece type */
    unsigned char pidx[120];            /* slot of the piece on sq in plist */
    U64 hash;
    U64 pawnHash;        /* pawns only, for correction history */
    /* net accumulators (search boards only; NULL on the UCI board) */
    char  (*accComp)[2]; /* [ACC_PLIES][2] is this ply's accumulator valid? */
    short* accMem;       /* ACC_PLIES x 2 x H                              */
    int    accBase;      /* hisPly of accumulator slot 0 (search root)     */
    MOVE_UNDO history[HIST_SIZE];
} BOARD;

/* lockless TT entry: keyx = hash ^ data, so a torn write never matches */
typedef struct {
    U64 keyx;
    U64 data;
} TT_ENTRY;

typedef struct {
    int    depth;
    long   nodes;
    MOVE   killers[MAX_PLY][2];
    int    history[13][120];
    MOVE   pv[MAX_PLY][MAX_PLY];
    int    pvLen[MAX_PLY];
    double startTime;                /* wall clock, ms                    */
    int    timeLimit;                /* hard limit, ms (search aborts)    */
    int    softLimit;                /* soft limit, ms (no new iteration) */
    long   nodeLimit;                /* "go nodes" (0 = none)             */
    int    stopped;
    int    evalStack[MAX_PLY + 4];   /* static eval per ply ("improving") */
    MOVE   counter[13][120];         /* reply that refuted [piece][to]    */
    int    stPiece[MAX_PLY + 4];     /* piece moved at each ply (0 = null)*/
    int    stTo[MAX_PLY + 4];        /* its destination, 0..63            */
    int*   cont;                     /* continuation history (per thread) */
    int*   corr;                     /* correction history (per thread)   */
    int    capHist[13][64][7];       /* capture history [piece][to][victim] */
    int    rootDepth;
    int    stopSoft;
    int    id;                       /* 0 = main thread                   */
    int    silent;                   /* no info output                    */
    MOVE   bestMove;
} SEARCH_INFO;

static const char pieceChar[] = ".PNBRQKpnbrqk";
static const int  pieceValue[] = {0,100,320,330,500,900,20000,100,320,330,500,900,20000};

static const int knightDirs[8] = {-21,-19,-12,-8,8,12,19,21};
static const int bishopDirs[4] = {-11,-9,9,11};
static const int rookDirs[4]   = {-10,-1,1,10};
static const int kingDirs[8]   = {-11,-10,-9,-1,1,9,10,11};
static const int queenDirs[8]  = {-10,-1,1,10,-11,-9,9,11};

static BOARD*    gboard = NULL;
static TT_ENTRY* tt      = NULL;
static U64       ttMask  = 0;   /* entries-1, always a power of two */
static U64       ttCount = 0;
static unsigned  g_age   = 0;   /* search generation for TT aging   */

static U64 zobristPiece[13][120];
static U64 zobristCastle[16];
static U64 zobristEP[120];
static U64 zobristSide;
static U64 rand64_state = 0x9E3779B97F4A7C15ULL;

#define EVAL_CACHE_SIZE (1 << 20)
static U64* evalCache = NULL;

/* ------------------------------------------- K7 net accumulator flags */

static void accInvalidate(BOARD* b, int hisPly) {
    int i;
    if (!b->accComp) return;
    i = hisPly - b->accBase;
    if (i >= 0 && i < ACC_PLIES) b->accComp[i][0] = b->accComp[i][1] = 0;
}

/* ------------------------------------------------------------ wall clock */

static double nowMs(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

/* ------------------------------------------------------------------ utils */

static U64 rand64(void) {
    rand64_state ^= rand64_state << 13;
    rand64_state ^= rand64_state >> 7;
    rand64_state ^= rand64_state << 17;
    return rand64_state;
}

static void initZobrist(void) {
    int i, sq;
    for (i = 0; i < 13; i++) for (sq = 0; sq < 120; sq++) zobristPiece[i][sq] = rand64();
    for (i = 0; i < 16; i++) zobristCastle[i] = rand64();
    for (sq = 0; sq < 120; sq++) zobristEP[sq] = rand64();
    zobristSide = rand64();
}

static int sq_valid(int sq) {
    int file;
    if (sq < 21 || sq > 98) return 0;
    file = sq % 10;
    return file >= 1 && file <= 8;
}

static int sq_rank(int sq) { return (sq / 10) - 2; }
static int sq_file(int sq) { return (sq % 10) - 1; }
static signed char sq64tab[120];
static int sq_to64(int sq) { return sq64tab[sq]; }
static void initSq64(void) {
    int sq;
    for (sq = 0; sq < 120; sq++) sq64tab[sq] = (signed char)(sq_valid(sq) ? sq_rank(sq) * 8 + sq_file(sq) : 0);
}

static void sq_to_str(int sq, char* str) {
    if (!sq_valid(sq)) { strcpy(str, "??"); return; }
    str[0] = (char)('a' + sq_file(sq));
    str[1] = (char)('1' + sq_rank(sq));
    str[2] = '\0';
}

static int str_to_sq(const char* str) {
    int file, rank;
    if (!str[0] || !str[1]) return -1;
    file = str[0] - 'a';
    rank = str[1] - '1';
    if (file < 0 || file > 7 || rank < 0 || rank > 7) return -1;
    return 21 + rank * 10 + file;
}

/* ------------------------------------------------------------------ board
 * V4: off-board squares hold OFFBOARD, so sliding loops stop on their own
 * (no sq_valid() call per step). Every piece is also kept in a piece list
 * (plist[piece][0..pcount-1], pidx[sq] = its slot) so move generation
 * visits only the pieces instead of all 120 squares.
 * --------------------------------------------------------------------- */

static unsigned char pcColor[14];      /* WHITE / BLACK / 2 empty / 3 off-board */
static signed char   lineDir[120][120];/* queen direction a->b if on one line, else 0 */
static unsigned char knightRel[240];   /* 1 if delta+120 is a knight jump */
static signed char   rankOf[120];

static void initBoardTables(void) {
    int p, a, i;
    for (p = 0; p < 14; p++) pcColor[p] = (p == EMPTY) ? 2 : (p == OFFBOARD ? 3 : (IS_BLACK(p) ? BLACK : WHITE));
    memset(lineDir, 0, sizeof(lineDir));
    memset(knightRel, 0, sizeof(knightRel));
    for (a = 0; a < 120; a++) rankOf[a] = (signed char)(sq_valid(a) ? sq_rank(a) : -1);
    for (a = 0; a < 120; a++) {
        if (!sq_valid(a)) continue;
        for (i = 0; i < 8; i++) {
            int d = queenDirs[i], t;
            for (t = a + d; sq_valid(t); t += d) lineDir[a][t] = (signed char)d;
        }
    }
    for (i = 0; i < 8; i++) knightRel[knightDirs[i] + 120] = 1;
}

static void addPiece(BOARD* b, int p, int sq) {
    b->board[sq] = p;
    b->pidx[sq] = (unsigned char)b->pcount[p];
    b->plist[p][b->pcount[p]++] = (unsigned char)sq;
}

static void remPiece(BOARD* b, int p, int sq) {
    int i = b->pidx[sq], last = b->plist[p][--b->pcount[p]];
    b->plist[p][i] = (unsigned char)last;
    b->pidx[last] = (unsigned char)i;
    b->board[sq] = EMPTY;
}

static void movePiece(BOARD* b, int p, int from, int to) {
    int i = b->pidx[from];
    b->plist[p][i] = (unsigned char)to;
    b->pidx[to] = (unsigned char)i;
    b->board[from] = EMPTY;
    b->board[to] = p;
}

static void boardClear(BOARD* b) {
    int i;
    for (i = 0; i < 120; i++) b->board[i] = sq_valid(i) ? EMPTY : OFFBOARD;
    b->stm = WHITE;
    b->hisPly = 0;
    b->enPas = -1;
    b->castlePermission = 0;
    b->fiftyMove = 0;
    b->kingSquare[WHITE] = E1;
    b->kingSquare[BLACK] = E8;
    b->hash = 0;
    b->pawnHash = 0;
    memset(b->pcount, 0, sizeof(b->pcount));
}

static U64 computeHash(BOARD* b) {
    U64 h = 0;
    int sq;
    for (sq = 21; sq <= 98; sq++) {
        if (!sq_valid(sq)) continue;
        if (b->board[sq] != EMPTY) h ^= zobristPiece[b->board[sq]][sq];
    }
    if (b->stm == BLACK) h ^= zobristSide;
    if (b->enPas >= 0)   h ^= zobristEP[b->enPas];
    h ^= zobristCastle[b->castlePermission];
    return h;
}

static U64 computePawnHash(BOARD* b) {
    U64 h = 0;
    int sq;
    for (sq = 21; sq <= 98; sq++)
        if (b->board[sq] == wP || b->board[sq] == bP) h ^= zobristPiece[b->board[sq]][sq];
    return h;
}

static void parseFEN(BOARD* b, const char* fen) {
    int sq = A8;
    const char* p = fen;

    boardClear(b);

    while (*p && *p != ' ') {
        if (*p >= '1' && *p <= '8') {
            sq += (*p - '0');
        } else if (*p == '/') {
            sq -= 18;
        } else {
            const char* pc = strchr(pieceChar, *p);
            if (pc && *p != '.') {
                int piece = (int)(pc - pieceChar);
                if (sq_valid(sq) && b->board[sq] == EMPTY && b->pcount[piece] < PLIST_MAX) {
                    addPiece(b, piece, sq);
                    if (piece == wK) b->kingSquare[WHITE] = sq;
                    if (piece == bK) b->kingSquare[BLACK] = sq;
                }
                sq++;
            }
        }
        p++;
    }

    while (*p == ' ') p++;
    b->stm = (*p == 'b') ? BLACK : WHITE;
    while (*p && *p != ' ') p++;
    while (*p == ' ') p++;

    b->castlePermission = 0;
    while (*p && *p != ' ') {
        if (*p == 'K') b->castlePermission |= WKCA;
        if (*p == 'Q') b->castlePermission |= WQCA;
        if (*p == 'k') b->castlePermission |= BKCA;
        if (*p == 'q') b->castlePermission |= BQCA;
        p++;
    }
    while (*p == ' ') p++;

    b->enPas = -1;
    if (*p && *p != '-') {
        int e = str_to_sq(p);
        if (e > 0) b->enPas = e;
    }
    while (*p && *p != ' ') p++;
    while (*p == ' ') p++;
    if (*p >= '0' && *p <= '9') b->fiftyMove = atoi(p);

    b->hash = computeHash(b);
    b->pawnHash = computePawnHash(b);
}

/* ------------------------------------------------------- attack detection */

/* Is `sq` attacked by any piece of colour `bySide`? (sentinel board) */
static int isSquareAttacked(BOARD* b, int sq, int bySide) {
    const int* bd = b->board;
    int i, t, p;
    int kn = MAKE_PC(bySide, KNIGHT), kg = MAKE_PC(bySide, KING);
    int bi = MAKE_PC(bySide, BISHOP), rk = MAKE_PC(bySide, ROOK), qu = MAKE_PC(bySide, QUEEN);

    if (bySide == WHITE) { if (bd[sq - 11] == wP || bd[sq - 9] == wP) return 1; }
    else                 { if (bd[sq + 11] == bP || bd[sq + 9] == bP) return 1; }

    if (b->pcount[kn])
        for (i = 0; i < 8; i++) if (bd[sq + knightDirs[i]] == kn) return 1;
    for (i = 0; i < 8; i++) if (bd[sq + kingDirs[i]] == kg) return 1;

    if (b->pcount[rk] | b->pcount[qu])
        for (i = 0; i < 4; i++) {
            int d = rookDirs[i];
            for (t = sq + d; (p = bd[t]) == EMPTY; t += d) ;
            if (p == rk || p == qu) return 1;
        }
    if (b->pcount[bi] | b->pcount[qu])
        for (i = 0; i < 4; i++) {
            int d = bishopDirs[i];
            for (t = sq + d; (p = bd[t]) == EMPTY; t += d) ;
            if (p == bi || p == qu) return 1;
        }
    return 0;
}

static int inCheck(BOARD* b, int side) {
    return isSquareAttacked(b, b->kingSquare[side], 1 - side);
}

/* ------------------------------------------------------- move generation */

/* Pseudo-legal moves of the piece on `sq`. capsOnly: captures + all
   promotions only (quiescence). */
static int genPieceMoves(BOARD* b, int sq, int piece, MOVE* moves, int count, int capsOnly) {
    const int* bd = b->board;
    int stm = b->stm, enemy = 1 - stm, i, baseType = BASE(piece);

    if (baseType == PAWN) {
        int dir        = (stm == WHITE) ? UP : DOWN;
        int start_rank = (stm == WHITE) ? 1 : 6;
        int promo_rank = (stm == WHITE) ? 7 : 0;
        int to = sq + dir, dx;

        if (bd[to] == EMPTY) {
            if (rankOf[to] == promo_rank) {
                moves[count++] = MV(sq, to, QUEEN_PROMO);
                moves[count++] = MV(sq, to, ROOK_PROMO);
                moves[count++] = MV(sq, to, BISHOP_PROMO);
                moves[count++] = MV(sq, to, KNIGHT_PROMO);
            } else if (!capsOnly) {
                moves[count++] = MV(sq, to, QUIET);
                if (rankOf[sq] == start_rank && bd[to + dir] == EMPTY)
                    moves[count++] = MV(sq, to + dir, DOUBLE_PUSH);
            }
        }
        for (dx = -1; dx <= 1; dx += 2) {
            int capto = to + dx, target = bd[capto];
            if (pcColor[target] == enemy) {
                if (rankOf[capto] == promo_rank) {
                    moves[count++] = MV(sq, capto, QUEEN_PROMO_CAP);
                    moves[count++] = MV(sq, capto, ROOK_PROMO_CAP);
                    moves[count++] = MV(sq, capto, BISHOP_PROMO_CAP);
                    moves[count++] = MV(sq, capto, KNIGHT_PROMO_CAP);
                } else {
                    moves[count++] = MV(sq, capto, CAPTURE);
                }
            } else if (target == EMPTY && capto == b->enPas) {
                moves[count++] = MV(sq, capto, EP_CAPTURE);
            }
        }
    }
    else if (baseType == KNIGHT || baseType == KING) {
        const int* dirs = (baseType == KNIGHT) ? knightDirs : kingDirs;
        for (i = 0; i < 8; i++) {
            int to = sq + dirs[i], target = bd[to];
            if (target == EMPTY) { if (!capsOnly) moves[count++] = MV(sq, to, QUIET); }
            else if (pcColor[target] == enemy) moves[count++] = MV(sq, to, CAPTURE);
        }
        if (baseType == KING && !capsOnly) {
            if (stm == WHITE && sq == E1) {
                if ((b->castlePermission & WKCA) && bd[F1] == EMPTY && bd[G1] == EMPTY && bd[H1] == wR &&
                    !isSquareAttacked(b, E1, BLACK) && !isSquareAttacked(b, F1, BLACK) && !isSquareAttacked(b, G1, BLACK))
                    moves[count++] = MV(E1, G1, KING_CASTLE);
                if ((b->castlePermission & WQCA) && bd[D1] == EMPTY && bd[C1] == EMPTY && bd[B1] == EMPTY && bd[A1] == wR &&
                    !isSquareAttacked(b, E1, BLACK) && !isSquareAttacked(b, D1, BLACK) && !isSquareAttacked(b, C1, BLACK))
                    moves[count++] = MV(E1, C1, QUEEN_CASTLE);
            } else if (stm == BLACK && sq == E8) {
                if ((b->castlePermission & BKCA) && bd[F8] == EMPTY && bd[G8] == EMPTY && bd[H8] == bR &&
                    !isSquareAttacked(b, E8, WHITE) && !isSquareAttacked(b, F8, WHITE) && !isSquareAttacked(b, G8, WHITE))
                    moves[count++] = MV(E8, G8, KING_CASTLE);
                if ((b->castlePermission & BQCA) && bd[D8] == EMPTY && bd[C8] == EMPTY && bd[B8] == EMPTY && bd[A8] == bR &&
                    !isSquareAttacked(b, E8, WHITE) && !isSquareAttacked(b, D8, WHITE) && !isSquareAttacked(b, C8, WHITE))
                    moves[count++] = MV(E8, C8, QUEEN_CASTLE);
            }
        }
    }
    else {                                          /* bishop, rook, queen */
        int i0 = (baseType == ROOK) ? 0 : (baseType == BISHOP ? 4 : 0);
        int i1 = (baseType == ROOK) ? 4 : 8;
        for (i = i0; i < i1; i++) {
            int dir = queenDirs[i], to = sq + dir, target;
            while ((target = bd[to]) == EMPTY) {
                if (!capsOnly) moves[count++] = MV(sq, to, QUIET);
                to += dir;
            }
            if (pcColor[target] == enemy) moves[count++] = MV(sq, to, CAPTURE);
        }
    }
    return count;
}

static int genMovesEx(BOARD* b, MOVE* moves, int capsOnly) {
    int count = 0, base = (b->stm == BLACK) ? 6 : 0, t, i;
    for (t = PAWN; t <= KING; t++) {
        int p = base + t, n = b->pcount[p];
        for (i = 0; i < n; i++) count = genPieceMoves(b, b->plist[p][i], p, moves, count, capsOnly);
    }
    return count;
}

static int genMoves(BOARD* b, MOVE* moves) { return genMovesEx(b, moves, 0); }

/* Is `m` a pseudo-legal move in this position? (used to try the hash move
   before generating anything; guards against TT key collisions) */
static int isPseudoLegal(BOARD* b, MOVE m) {
    MOVE list[32];
    int from = MV_FROM(m), piece, n, i;
    if (!m || from < 21 || from > 98) return 0;
    piece = b->board[from];
    if (pcColor[piece] != b->stm) return 0;
    n = genPieceMoves(b, from, piece, list, 0, 0);
    for (i = 0; i < n; i++) if (list[i] == m) return 1;
    return 0;
}

/* --------------------------------------------------------- make / unmake */

static void deltaRem(MOVE_UNDO* u, int piece, int sq) {
    if (u->nRem >= 2) return;
    u->remP[u->nRem] = (unsigned char)piece;
    u->remS[u->nRem] = (unsigned char)sq_to64(sq);
    u->nRem++;
}

static void deltaAdd(MOVE_UNDO* u, int piece, int sq) {
    if (u->nAdd >= 2) return;
    u->addP[u->nAdd] = (unsigned char)piece;
    u->addS[u->nAdd] = (unsigned char)sq_to64(sq);
    u->nAdd++;
}

static void makeMove(BOARD* b, MOVE move) {
    int from = MV_FROM(move), to = MV_TO(move), type = MV_TYPE(move);
    int piece = b->board[from];
    int captured = b->board[to];
    int oldCastle;
    MOVE_UNDO* undo = &b->history[b->hisPly];

    undo->move             = move;
    undo->enPas            = b->enPas;
    undo->castlePermission = b->castlePermission;
    undo->capturedPiece    = captured;
    undo->fiftyMove        = b->fiftyMove;
    undo->hash             = b->hash;
    undo->pawnHash         = b->pawnHash;
    undo->nRem = undo->nAdd = 0;

    if (b->enPas >= 0) b->hash ^= zobristEP[b->enPas];
    b->enPas = -1;

    deltaRem(undo, piece, from);
    b->hash ^= zobristPiece[piece][from];

    if (captured != EMPTY) {
        deltaRem(undo, captured, to);
        b->hash ^= zobristPiece[captured][to];
        if (BASE(captured) == PAWN) b->pawnHash ^= zobristPiece[captured][to];
        remPiece(b, captured, to);
    }
    movePiece(b, piece, from, to);
    b->hash ^= zobristPiece[piece][to];

    if (BASE(piece) == PAWN) {
        b->fiftyMove = 0;
        b->pawnHash ^= zobristPiece[piece][from];
        if (!IS_PROMO(type)) b->pawnHash ^= zobristPiece[piece][to];
    } else if (captured != EMPTY) b->fiftyMove = 0;
    else b->fiftyMove++;

    if (type == DOUBLE_PUSH) {
        b->enPas = from + ((b->stm == WHITE) ? UP : DOWN);
        b->hash ^= zobristEP[b->enPas];
    }

    if (type == EP_CAPTURE) {
        int capSq  = to + ((b->stm == WHITE) ? DOWN : UP);
        int capPce = (b->stm == WHITE) ? bP : wP;
        deltaRem(undo, capPce, capSq);
        b->hash ^= zobristPiece[capPce][capSq];
        b->pawnHash ^= zobristPiece[capPce][capSq];
        remPiece(b, capPce, capSq);
    }

    if (type == KING_CASTLE || type == QUEEN_CASTLE) {
        int rook, rf, rt;
        if (b->stm == WHITE) { rook = wR; if (type == KING_CASTLE) { rf = H1; rt = F1; } else { rf = A1; rt = D1; } }
        else                 { rook = bR; if (type == KING_CASTLE) { rf = H8; rt = F8; } else { rf = A8; rt = D8; } }
        deltaRem(undo, rook, rf);
        deltaAdd(undo, rook, rt);
        b->hash ^= zobristPiece[rook][rf];
        movePiece(b, rook, rf, rt);
        b->hash ^= zobristPiece[rook][rt];
    }

    if (IS_PROMO(type)) {
        static const int promo[5] = {0, wN, wB, wR, wQ};
        int promoPiece = promo[(type & 3) + 1];
        if (b->stm == BLACK) promoPiece += 6;
        b->hash ^= zobristPiece[piece][to];
        remPiece(b, piece, to);
        addPiece(b, promoPiece, to);
        b->hash ^= zobristPiece[promoPiece][to];
    }
    deltaAdd(undo, b->board[to], to);

    if (BASE(piece) == KING) b->kingSquare[b->stm] = to;

    oldCastle = b->castlePermission;
    if (oldCastle) {
        if (from == E1 || to == E1) b->castlePermission &= ~(WKCA | WQCA);
        if (from == E8 || to == E8) b->castlePermission &= ~(BKCA | BQCA);
        if (from == H1 || to == H1) b->castlePermission &= ~WKCA;
        if (from == A1 || to == A1) b->castlePermission &= ~WQCA;
        if (from == H8 || to == H8) b->castlePermission &= ~BKCA;
        if (from == A8 || to == A8) b->castlePermission &= ~BQCA;
        if (oldCastle != b->castlePermission) {
            b->hash ^= zobristCastle[oldCastle];
            b->hash ^= zobristCastle[b->castlePermission];
        }
    }

    b->hash ^= zobristSide;
    b->stm = 1 - b->stm;
    b->hisPly++;
    accInvalidate(b, b->hisPly);
}

static void undoMove(BOARD* b) {
    MOVE_UNDO* undo;
    MOVE move;
    int from, to, type, piece;

    b->hisPly--;
    undo = &b->history[b->hisPly];
    move = undo->move;
    from = MV_FROM(move);
    to   = MV_TO(move);
    type = MV_TYPE(move);

    b->stm  = 1 - b->stm;
    b->hash = undo->hash;
    b->pawnHash         = undo->pawnHash;
    b->enPas            = undo->enPas;
    b->castlePermission = undo->castlePermission;
    b->fiftyMove        = undo->fiftyMove;

    piece = b->board[to];
    if (IS_PROMO(type)) {
        remPiece(b, piece, to);
        addPiece(b, (b->stm == WHITE) ? wP : bP, from);
    } else {
        movePiece(b, piece, to, from);
    }
    if (undo->capturedPiece != EMPTY) addPiece(b, undo->capturedPiece, to);

    if (type == KING_CASTLE || type == QUEEN_CASTLE) {
        int rook, rf, rt;
        if (b->stm == WHITE) { rook = wR; if (type == KING_CASTLE) { rf = H1; rt = F1; } else { rf = A1; rt = D1; } }
        else                 { rook = bR; if (type == KING_CASTLE) { rf = H8; rt = F8; } else { rf = A8; rt = D8; } }
        movePiece(b, rook, rt, rf);
    }

    if (type == EP_CAPTURE) {
        int capSq = to + ((b->stm == WHITE) ? DOWN : UP);
        addPiece(b, (b->stm == WHITE) ? bP : wP, capSq);
    }

    if (BASE(b->board[from]) == KING) b->kingSquare[b->stm] = from;
}

/* ---------------------------------------------------- draw / null move */

static int isRepetition(BOARD* b) {
    int i, start = b->hisPly - b->fiftyMove;
    if (start < 0) start = 0;
    for (i = b->hisPly - 2; i >= start; i -= 2)
        if (b->history[i].hash == b->hash) return 1;
    return 0;
}

static int hasNonPawnMaterial(BOARD* b, int side) {
    const int* c = b->pcount + (side == BLACK ? 6 : 0);
    return (c[KNIGHT] + c[BISHOP] + c[ROOK] + c[QUEEN]) > 0;
}

#ifndef K7_V1_EXACT
/* K v K, K+minor v K: nobody can ever mate -> dead draw */
static int insufficientMaterial(BOARD* b) {
    const int* c = b->pcount;
    if (c[wP] + c[bP] + c[wR] + c[bR] + c[wQ] + c[bQ]) return 0;
    return (c[wN] + c[wB] + c[bN] + c[bB]) <= 1;
}
#endif

static void makeNullMove(BOARD* b) {
    MOVE_UNDO* undo = &b->history[b->hisPly];
    undo->move             = 0;
    undo->enPas            = b->enPas;
    undo->castlePermission = b->castlePermission;
    undo->capturedPiece    = EMPTY;
    undo->fiftyMove        = b->fiftyMove;
    undo->hash             = b->hash;
    undo->pawnHash         = b->pawnHash;
    undo->nRem = undo->nAdd = 0;

    if (b->enPas >= 0) b->hash ^= zobristEP[b->enPas];
    b->enPas = -1;
    b->hash ^= zobristSide;
    b->stm = 1 - b->stm;
    b->hisPly++;
    accInvalidate(b, b->hisPly);
}

static void undoNullMove(BOARD* b) {
    MOVE_UNDO* undo;
    b->hisPly--;
    undo = &b->history[b->hisPly];
    b->stm              = 1 - b->stm;
    b->hash             = undo->hash;
    b->enPas            = undo->enPas;
    b->castlePermission = undo->castlePermission;
    b->fiftyMove        = undo->fiftyMove;
}

static int genLegal(BOARD* b, MOVE* moves) {
    MOVE pseudo[MAX_MOVES];
    int n = genMoves(b, pseudo);
    int count = 0, i, us = b->stm;
    for (i = 0; i < n; i++) {
        makeMove(b, pseudo[i]);
        if (!inCheck(b, us)) moves[count++] = pseudo[i];
        undoMove(b);
    }
    return count;
}

/* Can this move possibly leave our own king in check? If not, the legality
   test after makeMove can be skipped. (checked = we are in check now) */
static int mayExposeKing(BOARD* b, MOVE m, int checked) {
    int from = MV_FROM(m);
    if (checked) return 1;
    if (BASE(b->board[from]) == KING || MV_TYPE(m) == EP_CAPTURE) return 1;
    return lineDir[b->kingSquare[b->stm]][from] != 0;
}

/* Call right after makeMove(m): does the move check the side to move now?
   Cheap pre-test first, full attack test only when a check is possible. */
static int givesCheckAfter(BOARD* b, MOVE m) {
    int from = MV_FROM(m), to = MV_TO(m), type = MV_TYPE(m);
    int ek = b->kingSquare[b->stm], p = b->board[to], bt = BASE(p), d;
    if (type == KING_CASTLE || type == QUEEN_CASTLE || type == EP_CAPTURE || IS_PROMO(type)
        || lineDir[ek][from] != 0)
        return inCheck(b, b->stm);
    switch (bt) {
    case PAWN:
        return (b->stm == BLACK) ? (ek == to + 9 || ek == to + 11) : (ek == to - 9 || ek == to - 11);
    case KNIGHT:
        return knightRel[ek - to + 120];
    case KING:
        return 0;
    default:
        d = lineDir[to][ek];
        if (!d) return 0;
        if (bt == BISHOP && (d == 10 || d == -10 || d == 1 || d == -1)) return 0;
        if (bt == ROOK && !(d == 10 || d == -10 || d == 1 || d == -1)) return 0;
        {
            int t = to + d;
            while (b->board[t] == EMPTY) t += d;
            return t == ek;
        }
    }
}

/* ------------------------------------------------------------------ perft */

static long perft(BOARD* b, int depth) {
    MOVE moves[MAX_MOVES];
    int n, i, us;
    long total = 0;

    if (depth == 0) return 1;

    n  = genMoves(b, moves);
    us = b->stm;
    for (i = 0; i < n; i++) {
        makeMove(b, moves[i]);
        if (!inCheck(b, us)) total += perft(b, depth - 1);
        undoMove(b);
    }
    return total;
}


/* ================================================================
 *  EVALUATION: K7 CUSTOM NET (K7NN format from k7_train)
 *
 *  768 inputs -> H x2 (SCReLU) -> 1
 *  feature = colour*384 + type*64 + square      (White's view, a1 = 0)
 *            (colour^1)*384 + type*64 + (sq^56)  (Black's view, board flipped)
 *  Kings are ordinary features here, so a king move is just two row
 *  updates -- the accumulator never needs a full refresh during search.
 * ================================================================ */

static int      nnH       = 0;       /* hidden size read from the file   */
static int      nnQA      = 255;
static int      nnQB      = 64;
static int      nnScale   = 400;
static short*   nnW1      = NULL;    /* [768][H]                         */
static short*   nnB1      = NULL;    /* [H]                              */
static short*   nnW2      = NULL;    /* [2H]  side to move, then other   */
static int      nnB2      = 0;
static int      nnLoaded  = 0;
static int      nnSimdAcc = 0;       /* AVX2 accumulator updates usable  */
static int      nnSimdOut = 0;       /* AVX2 output layer usable         */
static char     nnPath[1024] = "<embedded>";

/* 64-byte aligned allocation (cache line / AVX friendly) */
static void* amalloc(size_t n) {
    unsigned char* raw = (unsigned char*)malloc(n + 64 + sizeof(void*));
    uintptr_t p;
    if (!raw) return NULL;
    p = ((uintptr_t)(raw + sizeof(void*)) + 63) & ~(uintptr_t)63;
    ((void**)p)[-1] = raw;
    return (void*)p;
}

static void afree(void* p) {
    if (p) free(((void**)p)[-1]);
}

static short* accPtr(BOARD* b, int idx, int p) {
    return b->accMem + ((size_t)idx * 2 + (size_t)p) * (size_t)nnH;
}

static unsigned int rd32(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

static void nnFree(void) {
    afree(nnW1); afree(nnB1); afree(nnW2);
    nnW1 = nnB1 = nnW2 = NULL;
    nnH = 0;
    nnLoaded = 0;
    nnSimdAcc = nnSimdOut = 0;
}

/* parse a K7NN net from memory (file contents or the embedded net) */
static int nnParse(const unsigned char* d, long fsz, const char* path) {
    size_t off, i, nW1, need;
    int H, maxW2 = 0;
    const char* how;

    nnFree();
    if (fsz < 24) { printf("info string NetFile: %s is too small to be a K7 net\n", path); return 0; }
    if (memcmp(d, "K7NN", 4) != 0 || rd32(d + 4) != 1u) {
        printf("info string NetFile: %s is not a K7NN v1 net (make it with k7_train)\n", path);
        return 0;
    }
    H       = (int)rd32(d + 8);
    nnQA    = (int)rd32(d + 12);
    nnQB    = (int)rd32(d + 16);
    nnScale = (int)rd32(d + 20);
    nW1     = (size_t)768 * (size_t)H;
    need    = 24 + 2 * (nW1 + (size_t)H + 2 * (size_t)H) + 4;
    if (H <= 0 || H > 4096 || nnQA <= 0 || nnQA > 32767 || nnQB <= 0 || (size_t)fsz != need) {
        printf("info string NetFile: %s has a bad header or size\n", path);
        return 0;
    }

    nnW1 = (short*)amalloc(sizeof(short) * nW1);
    nnB1 = (short*)amalloc(sizeof(short) * (size_t)H);
    nnW2 = (short*)amalloc(sizeof(short) * 2 * (size_t)H);
    if (!nnW1 || !nnB1 || !nnW2) {
        nnFree();
        printf("info string NetFile: out of memory\n");
        return 0;
    }

    off = 24;
    for (i = 0; i < nW1; i++, off += 2)            nnW1[i] = (short)(d[off] | (d[off + 1] << 8));
    for (i = 0; i < (size_t)H; i++, off += 2)      nnB1[i] = (short)(d[off] | (d[off + 1] << 8));
    for (i = 0; i < 2 * (size_t)H; i++, off += 2) {
        int w;
        nnW2[i] = (short)(d[off] | (d[off + 1] << 8));
        w = nnW2[i] < 0 ? -nnW2[i] : nnW2[i];
        if (w > maxW2) maxW2 = w;
    }
    nnB2 = (int)rd32(d + off);

    nnH = H;
    /* SIMD paths give bit-identical results; they need H % 16 == 0, and the
       output trick needs clamp(v) * w to fit in int16 (QA * |w2| <= 32767) */
    nnSimdAcc = K7_AVX2 && (H % 16 == 0);
    nnSimdOut = nnSimdAcc && (long)maxW2 * (long)nnQA <= 32767L;
    nnLoaded = 1;
    how = nnSimdOut ? "AVX2" : (nnSimdAcc ? "AVX2 acc, scalar output" : "scalar");
    printf("info string NetFile: loaded %s (768 -> %dx2 -> 1, %ld bytes, %s)\n", path, H, fsz, how);
    return 1;
}

static int nnLoad(const char* path) {
    FILE* f;
    long fsz;
    unsigned char* d;
    int ok;

    f = fopen(path, "rb");
    if (!f) { printf("info string NetFile: cannot open %s\n", path); return 0; }
    fseek(f, 0, SEEK_END);
    fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsz < 24) { fclose(f); printf("info string NetFile: %s is too small to be a K7 net\n", path); return 0; }
    d = (unsigned char*)malloc((size_t)fsz);
    if (!d || fread(d, 1, (size_t)fsz, f) != (size_t)fsz) {
        fclose(f); free(d);
        printf("info string NetFile: read error on %s\n", path);
        return 0;
    }
    fclose(f);
    ok = nnParse(d, fsz, path);
    free(d);
    return ok;
}

/* ================================================================
 *  EMBEDDED NET + BOOK (incbin)
 *  The files are read by the assembler at compile time, so put
 *  k7net.bin and k7Opennings.bin next to this .c file (or pass full
 *  paths with -DK7_NET_FILE="\"D:/path/k7net.bin\"" and
 *  -DK7_BOOK_FILE="\"D:/path/k7Opennings.bin\""). Build with
 *  -DK7_NO_EMBED to make an exe that only uses NetFile/BookFile.
 * ================================================================ */
#ifndef K7_NET_FILE
#  define K7_NET_FILE "k7net.bin"
#endif
#ifndef K7_BOOK_FILE
#  define K7_BOOK_FILE "k7Opennings.bin"
#endif

#ifndef K7_NO_EMBED
#  if defined(_WIN32) || defined(__CYGWIN__)
#    define K7_RODATA ".section .rdata,\"dr\"\n"
#  elif defined(__APPLE__)
#    define K7_RODATA ".const_data\n"
#  else
#    define K7_RODATA ".section .rodata\n"
#  endif
#  if defined(__APPLE__) || (defined(_WIN32) && !defined(_WIN64))
#    define K7_SYM(x) "_" #x
#  else
#    define K7_SYM(x) #x
#  endif
#  define K7_INCBIN(name, file)                                              \
     __asm__(K7_RODATA ".balign 64\n.globl " K7_SYM(name##_data) "\n"        \
             K7_SYM(name##_data) ":\n.incbin \"" file "\"\n"                 \
             ".globl " K7_SYM(name##_end) "\n" K7_SYM(name##_end) ":\n"      \
             ".byte 0\n.text\n");
#  ifdef __cplusplus
extern "C" {
#  endif
extern const unsigned char k7EmbNet_data[], k7EmbNet_end[];
extern const unsigned char k7EmbBook_data[], k7EmbBook_end[];
#  ifdef __cplusplus
}
#  endif
K7_INCBIN(k7EmbNet, K7_NET_FILE)
K7_INCBIN(k7EmbBook, K7_BOOK_FILE)
#endif

static int isEmbeddedName(const char* s) {
    return s[0] == 0 || !strcmp(s, "<embedded>") || !strcmp(s, "embedded");
}

/* NetFile = <embedded> -> the built-in net, anything else -> that file */
static int nnLoadAny(const char* path) {
#ifndef K7_NO_EMBED
    if (isEmbeddedName(path))
        return nnParse(k7EmbNet_data, (long)(k7EmbNet_end - k7EmbNet_data), "<embedded> (" K7_NET_FILE ")");
#else
    if (isEmbeddedName(path)) { printf("info string NetFile: this build has no embedded net\n"); return 0; }
#endif
    return nnLoad(path);
}

static int nnFeature(int persp, int piece, int s64) {
    int color = IS_BLACK(piece) ? BLACK : WHITE, type = BASE(piece) - 1;
    if (persp == WHITE) return color * 384 + type * 64 + s64;
    return (color ^ 1) * 384 + type * 64 + (s64 ^ 56);
}

static void nnAddRow(short* acc, int f) {
    const short* w = nnW1 + (size_t)f * (size_t)nnH;
    int i;
    for (i = 0; i < nnH; i++) acc[i] = (short)(acc[i] + w[i]);
}

static void nnSubRow(short* acc, int f) {
    const short* w = nnW1 + (size_t)f * (size_t)nnH;
    int i;
    for (i = 0; i < nnH; i++) acc[i] = (short)(acc[i] - w[i]);
}

static void nnRefresh(BOARD* b, int idx, int p) {
    short* acc = accPtr(b, idx, p);
    int sq;
    memcpy(acc, nnB1, sizeof(short) * (size_t)nnH);
    for (sq = 21; sq <= 98; sq++) {
        int pc;
        if (!sq_valid(sq)) continue;
        pc = b->board[sq];
        if (pc != EMPTY) nnAddRow(acc, nnFeature(p, pc, sq_to64(sq)));
    }
    b->accComp[idx][p] = 1;
}

#if defined(__GNUC__) || defined(__clang__)
#  define K7R __restrict
#else
#  define K7R
#endif

static const short* nnRow(int p, int piece, int s64) {
    return nnW1 + (size_t)nnFeature(p, piece, s64) * (size_t)nnH;
}

/* scalar: dst = src - removed rows + added rows in one pass */
static void nnApplyDeltaScalar(short* K7R dst, const short* K7R src, const MOVE_UNDO* u, int p) {
    int i, H = nnH;
    if (u->nRem == 1 && u->nAdd == 1) {                 /* quiet move */
        const short* K7R r0 = nnRow(p, u->remP[0], u->remS[0]);
        const short* K7R a0 = nnRow(p, u->addP[0], u->addS[0]);
        for (i = 0; i < H; i++) dst[i] = (short)(src[i] - r0[i] + a0[i]);
    } else if (u->nRem == 2 && u->nAdd == 1) {          /* capture / promotion-capture / ep */
        const short* K7R r0 = nnRow(p, u->remP[0], u->remS[0]);
        const short* K7R r1 = nnRow(p, u->remP[1], u->remS[1]);
        const short* K7R a0 = nnRow(p, u->addP[0], u->addS[0]);
        for (i = 0; i < H; i++) dst[i] = (short)(src[i] - r0[i] - r1[i] + a0[i]);
    } else if (u->nRem == 2 && u->nAdd == 2) {          /* castling */
        const short* K7R r0 = nnRow(p, u->remP[0], u->remS[0]);
        const short* K7R r1 = nnRow(p, u->remP[1], u->remS[1]);
        const short* K7R a0 = nnRow(p, u->addP[0], u->addS[0]);
        const short* K7R a1 = nnRow(p, u->addP[1], u->addS[1]);
        for (i = 0; i < H; i++) dst[i] = (short)(src[i] - r0[i] - r1[i] + a0[i] + a1[i]);
    } else {                                            /* null move or anything else */
        int r;
        memcpy(dst, src, sizeof(short) * (size_t)H);
        for (r = 0; r < u->nRem; r++) nnSubRow(dst, nnFeature(p, u->remP[r], u->remS[r]));
        for (r = 0; r < u->nAdd; r++) nnAddRow(dst, nnFeature(p, u->addP[r], u->addS[r]));
    }
}

#if K7_AVX2
#define LD(x)    _mm256_loadu_si256((const __m256i*)(x))
#define ST(x, v) _mm256_storeu_si256((__m256i*)(x), (v))

static void nnApplyDeltaAVX2(short* dst, const short* src, const MOVE_UNDO* u, int p) {
    int i, H = nnH;
    if (u->nRem == 1 && u->nAdd == 1) {
        const short* r0 = nnRow(p, u->remP[0], u->remS[0]);
        const short* a0 = nnRow(p, u->addP[0], u->addS[0]);
        for (i = 0; i < H; i += 16)
            ST(dst + i, _mm256_add_epi16(_mm256_sub_epi16(LD(src + i), LD(r0 + i)), LD(a0 + i)));
    } else if (u->nRem == 2 && u->nAdd == 1) {
        const short* r0 = nnRow(p, u->remP[0], u->remS[0]);
        const short* r1 = nnRow(p, u->remP[1], u->remS[1]);
        const short* a0 = nnRow(p, u->addP[0], u->addS[0]);
        for (i = 0; i < H; i += 16) {
            __m256i v = _mm256_sub_epi16(LD(src + i), LD(r0 + i));
            v = _mm256_sub_epi16(v, LD(r1 + i));
            ST(dst + i, _mm256_add_epi16(v, LD(a0 + i)));
        }
    } else if (u->nRem == 2 && u->nAdd == 2) {
        const short* r0 = nnRow(p, u->remP[0], u->remS[0]);
        const short* r1 = nnRow(p, u->remP[1], u->remS[1]);
        const short* a0 = nnRow(p, u->addP[0], u->addS[0]);
        const short* a1 = nnRow(p, u->addP[1], u->addS[1]);
        for (i = 0; i < H; i += 16) {
            __m256i v = _mm256_sub_epi16(LD(src + i), LD(r0 + i));
            v = _mm256_sub_epi16(v, LD(r1 + i));
            v = _mm256_add_epi16(v, LD(a0 + i));
            ST(dst + i, _mm256_add_epi16(v, LD(a1 + i)));
        }
    } else {
        nnApplyDeltaScalar(dst, src, u, p);
    }
}

static long long hsum32(__m256i v) {
    int t[8], k;
    long long s = 0;
    _mm256_storeu_si256((__m256i*)t, v);
    for (k = 0; k < 8; k++) s += t[k];
    return s;
}

/* sum of clamp(a)^2 * w over both perspectives.
   clamp(a) * w fits int16 (checked at load), madd with clamp(a) gives
   int32 pairs; flushed to int64 every 32 steps so nothing can overflow.
   Integer math throughout -> exactly the scalar result. */
static long long nnOutAVX2(const short* as, const short* ao) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16((short)nnQA);
    const short* w2s = nnW2;
    const short* w2o = nnW2 + nnH;
    __m256i acc = zero;
    long long sum = 0;
    int i, n = 0;
    for (i = 0; i < nnH; i += 16) {
        __m256i vs = _mm256_min_epi16(_mm256_max_epi16(LD(as + i), zero), qa);
        __m256i vo = _mm256_min_epi16(_mm256_max_epi16(LD(ao + i), zero), qa);
        __m256i ps = _mm256_mullo_epi16(vs, LD(w2s + i));
        __m256i po = _mm256_mullo_epi16(vo, LD(w2o + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ps, vs));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(po, vo));
        if (++n == 32) { sum += hsum32(acc); acc = zero; n = 0; }
    }
    return sum + hsum32(acc);
}
#endif

static void nnApplyDelta(short* dst, const short* src, const MOVE_UNDO* u, int p) {
#if K7_AVX2
    if (nnSimdAcc) { nnApplyDeltaAVX2(dst, src, u, p); return; }
#endif
    nnApplyDeltaScalar(dst, src, u, p);
}

/* V2 output layer (blocks of 64 keep partial sums inside int32) */
static long long nnOutScalar(const short* as, const short* ao) {
    const short* K7R w2s = nnW2;
    const short* K7R w2o = nnW2 + nnH;
    long long sum = 0;
    int i, blk, H = nnH, QA = nnQA;
    for (blk = 0; blk < H; blk += 64) {
        int end = blk + 64 < H ? blk + 64 : H, s32 = 0;
        for (i = blk; i < end; i++) {
            int cs = as[i] < 0 ? 0 : (as[i] > QA ? QA : as[i]);
            int co = ao[i] < 0 ? 0 : (ao[i] > QA ? QA : ao[i]);
            s32 += cs * cs * w2s[i] + co * co * w2o[i];
        }
        sum += s32;
    }
    return sum;
}

/* Bring this ply's accumulator up to date by replaying move deltas from the
   nearest computed ancestor (a full refresh only if none is close). */
static void nnUpdate(BOARD* b, int p) {
    int h = b->hisPly - b->accBase, j = h, steps = 0, k;
    if (b->accComp[h][p]) return;
    while (!b->accComp[j][p]) {
        if (j == 0 || ++steps > 32) { nnRefresh(b, h, p); return; }
        j--;
    }
    for (k = j; k < h; k++) {
        nnApplyDelta(accPtr(b, k + 1, p), accPtr(b, k, p), &b->history[b->accBase + k], p);
        b->accComp[k + 1][p] = 1;
    }
}

/* Network output in centipawns, side to move. */
static int evalNet(BOARD* b) {
    const short *as, *ao;
    long long sum, out;
    int idx = b->hisPly - b->accBase;

    nnUpdate(b, WHITE);
    nnUpdate(b, BLACK);
    as = accPtr(b, idx, b->stm);
    ao = accPtr(b, idx, 1 - b->stm);

#if K7_AVX2
    if (nnSimdOut) sum = nnOutAVX2(as, ao);
    else
#endif
    sum = nnOutScalar(as, ao);

    out = sum / nnQA + nnB2;
    out = out * nnScale / ((long long)nnQA * nnQB);
    if (out >  20000) out =  20000;
    if (out < -20000) out = -20000;
    return (int)out;
}

/* Used only when no net is loaded, so the engine still plays legal chess. */
static int evalMaterial(BOARD* b) {
    int sq, s = 0;
    for (sq = 21; sq <= 98; sq++) {
        int p;
        if (!sq_valid(sq)) continue;
        p = b->board[sq];
        if (p == EMPTY || BASE(p) == KING) continue;
        s += IS_BLACK(p) ? -pieceValue[p] : pieceValue[p];
    }
    return (b->stm == WHITE) ? s : -s;
}

/* ================================================================
 *  EVAL + CACHE  (shared by all threads, lockless: one 64-bit word
 *  per entry = upper 48 hash bits | score + 32768)
 * ================================================================ */

static void evalCacheClear(void) {
    if (evalCache) memset(evalCache, 0, sizeof(U64) * EVAL_CACHE_SIZE);
}

static int optFifty = 1;   /* scale eval down as the 50-move counter grows */

/* 50-move scaling: an advantage that makes no progress (no pawn move or
   capture) slowly shrinks toward 0, so the engine pushes pawns while it
   is winning instead of shuffling until the draw is close. Applied after
   the cache (the hash does not include the 50-move counter). */
static int fiftyScale(BOARD* b, int s) {
    if (!optFifty) return s;
    return s * (200 - b->fiftyMove) / 200;
}

static int eval(BOARD* b) {
    volatile U64* e;
    U64 v;
    int s;
    if (!nnLoaded) return fiftyScale(b, evalMaterial(b));
    e = &evalCache[b->hash & (EVAL_CACHE_SIZE - 1)];
    v = *e;
    if (v && (v >> 16) == (b->hash >> 16)) return fiftyScale(b, (int)(v & 0xFFFF) - 32768);
    s = evalNet(b);
    *e = ((b->hash >> 16) << 16) | (U64)(unsigned)(s + 32768);
    return fiftyScale(b, s);
}

/* ------------------------------------------------------------------- TT
 * data = move:18 | score+2^19:20 | depth:8 | flag:2 | age:8
 * --------------------------------------------------------------------- */

typedef struct { MOVE move; int score, depth, flag, age; } TTDATA;

static void ttClear(void) {
    if (tt) memset(tt, 0, sizeof(TT_ENTRY) * (size_t)ttCount);
    g_age = 0;
}

/* V4: the table is split into clusters of 4 entries (64 bytes = one cache
   line). A position may live in any of its cluster's 4 slots, so deep or
   current entries are no longer thrown out by every collision. */
#define TT_WAYS 4

static void ttResize(int mb) {
    U64 bytes, entries = TT_WAYS;
    if (mb < 1) mb = 1;
    if (mb > TT_MAX_MB) mb = TT_MAX_MB;
    bytes = (U64)mb * 1024ULL * 1024ULL;
    while ((entries * 2ULL) * sizeof(TT_ENTRY) <= bytes) entries *= 2ULL;
    if (tt) afree(tt);
    tt = (TT_ENTRY*)amalloc(sizeof(TT_ENTRY) * (size_t)entries);
    while (!tt && entries > 1024) {
        entries /= 2ULL;
        tt = (TT_ENTRY*)amalloc(sizeof(TT_ENTRY) * (size_t)entries);
    }
    ttCount = entries;
    ttMask  = entries / TT_WAYS - 1ULL;          /* cluster index mask */
    ttClear();
}

static int ttHashfull(void) {
    int used = 0, i, n = (ttCount < 1000) ? (int)ttCount : 1000;
    for (i = 0; i < n; i++)
        if (tt[i].data && (int)((tt[i].data >> 48) & 0xFF) == (int)(g_age & 0xFF)) used++;
    return n ? (used * 1000 / n) : 0;
}

static int ttRead(U64 hash, TTDATA* out) {
    volatile TT_ENTRY* c = &tt[(hash & ttMask) * TT_WAYS];
    int w;
    for (w = 0; w < TT_WAYS; w++) {
        U64 d = c[w].data, k = c[w].keyx;
        if (!d || (k ^ d) != hash) continue;
        out->move  = (MOVE)(d & 0x3FFFF);
        out->score = (int)((d >> 18) & 0xFFFFF) - (1 << 19);
        out->depth = (int)((d >> 38) & 0xFF);
        out->flag  = (int)((d >> 46) & 3);
        out->age   = (int)((d >> 48) & 0xFF);
        return 1;
    }
    return 0;
}

static int ttScoreFrom(int sc, int ply) {
    if (sc >  MATE_SCORE - MAX_PLY) return sc - ply;
    if (sc < -MATE_SCORE + MAX_PLY) return sc + ply;
    return sc;
}

static void ttStore(U64 hash, int depth, int ply, int score, int flag, MOVE move) {
    volatile TT_ENTRY* c = &tt[(hash & ttMask) * TT_WAYS];
    volatile TT_ENTRY* e = NULL;
    U64 nd;
    int age = (int)(g_age & 0xFF), w, worst = INT_MAX;

    if (score >  MATE_SCORE - MAX_PLY) score += ply;
    else if (score < -MATE_SCORE + MAX_PLY) score -= ply;
    if (depth < 0) depth = 0;
    if (depth > 255) depth = 255;

    for (w = 0; w < TT_WAYS; w++) {
        U64 od = c[w].data, ok = c[w].keyx;
        int odepth, oage, value;
        if (!od) { if (worst > -1000) { worst = -1000; e = &c[w]; } continue; }
        odepth = (int)((od >> 38) & 0xFF);
        oage   = (int)((od >> 48) & 0xFF);
        if ((ok ^ od) == hash) {                      /* same position: update it */
            if (!move) move = (MOVE)(od & 0x3FFFF);
            if (flag != TT_EXACT && oage == age && depth + 2 < odepth) return;
            e = &c[w];
            break;
        }
        value = odepth - (oage != age ? 256 : 0);     /* old searches go first, then shallow */
        if (value < worst) { worst = value; e = &c[w]; }
    }

    nd = ((U64)move & 0x3FFFFULL)
       | (((U64)(score + (1 << 19)) & 0xFFFFFULL) << 18)
       | ((U64)depth << 38)
       | ((U64)(flag & 3) << 46)
       | ((U64)age << 48);
    e->data = nd;
    e->keyx = hash ^ nd;
}

/* =====================================================================
 *  SEARCH OPTIONS  (UCI check boxes)
 * ===================================================================== */
static int optRFP      = 1;   /* reverse futility pruning                    */
static int optNullPlus = 1;   /* adaptive null move, only when eval >= beta  */
static int optFutility = 1;   /* futility pruning of quiet moves             */
static int optLMP      = 1;   /* late move pruning                           */
static int optLMRPlus  = 1;   /* log-table late move reductions              */
static int optSEE      = 1;   /* static exchange eval: ordering + pruning    */
static int optDelta    = 0;   /* delta pruning in quiescence (off: tested -40 Elo) */
static int optIIR      = 1;   /* internal iterative reduction                */
static int optOrdering = 1;   /* counter-moves + history gravity/malus       */
/* V3 */
static int optQsTT     = 1;   /* transposition table in quiescence           */
static int optTTEval   = 1;   /* TT score refines the pruning eval           */
static int optSingular = 1;   /* singular extensions + multi-cut             */
static int optCont     = 1;   /* continuation history                        */
/* V4 */
static int optCorr     = 1;   /* correction history (pawn-structure eval fix) */
static int optCapHist  = 1;   /* capture history for ordering captures        */
static int optTimeMgr  = 1;   /* smarter time use (stability / score drops)   */

/* V4 tunables: every hand-picked search number is a UCI spin option, so
   they can be tuned (SPSA). Defaults = V3 values. */
static int tRfpMargin  = 80,  tRfpImp     = 40,  tRfpDepth  = 7;
static int tNmpBase    = 3,   tNmpDiv     = 4,   tNmpEvalDiv = 200;
static int tFutBase    = 110, tFutMult    = 100, tFutDepth  = 6;
static int tLmpBase    = 3;
static int tLmrBase    = 75,  tLmrDiv     = 225;        /* /100 */
static int tSeeMargin  = 100, tSeeDepth   = 6;
static int tSingDepth  = 8,   tSingMargin = 2;
static int tAspWindow  = 35;
static int tHistLmrDiv = 6000;
static int tMoveOverhead = 30;

typedef struct { const char* name; int* v; int lo, hi; } TUNE;
static TUNE tunes[] = {
    {"RfpMargin", &tRfpMargin, 20, 200}, {"RfpImproving", &tRfpImp, 0, 120}, {"RfpDepth", &tRfpDepth, 2, 12},
    {"NmpBase", &tNmpBase, 1, 6}, {"NmpDepthDiv", &tNmpDiv, 2, 8}, {"NmpEvalDiv", &tNmpEvalDiv, 50, 600},
    {"FutBase", &tFutBase, 0, 300}, {"FutMult", &tFutMult, 30, 250}, {"FutDepth", &tFutDepth, 2, 12},
    {"LmpBase", &tLmpBase, 1, 10}, {"LmrBase", &tLmrBase, 0, 200}, {"LmrDiv", &tLmrDiv, 100, 400},
    {"SeeMargin", &tSeeMargin, 20, 300}, {"SeeDepth", &tSeeDepth, 2, 12},
    {"SingDepth", &tSingDepth, 4, 14}, {"SingMargin", &tSingMargin, 1, 6},
    {"AspWindow", &tAspWindow, 5, 150}, {"HistLmrDiv", &tHistLmrDiv, 1000, 30000},
};
#define NTUNES ((int)(sizeof(tunes) / sizeof(tunes[0])))

#define MATE_BOUND (MATE_SCORE - MAX_PLY)
#define EVAL_NONE  (-INF_SCORE - 1)
#define CONT_DIM   (13 * 64)
#define CONT_SIZE  ((size_t)CONT_DIM * CONT_DIM)

static int lmrTable[64][64];

/* natural log without libm (only used once, to fill the LMR table) */
static double k7log(double x) {
    double y, y2, sum = 0.0, term;
    int k = 0, n;
    while (x > 2.0) { x *= 0.5; k++; }
    y = (x - 1.0) / (x + 1.0);
    y2 = y * y;
    term = y;
    for (n = 1; n < 40; n += 2) { sum += term / n; term *= y2; }
    return 2.0 * sum + k * 0.69314718055994531;
}

static void initLMR(void) {
    int d, m;
    for (d = 0; d < 64; d++)
        for (m = 0; m < 64; m++)
            lmrTable[d][m] = (d < 1 || m < 1) ? 0
                           : (int)(tLmrBase / 100.0 + k7log((double)d) * k7log((double)m) / (tLmrDiv / 100.0));
}

/* ---------------------------------------------------------------- SEE */
static const int seeVal[13] = { 0, 100, 320, 330, 500, 900, 20000,
                                   100, 320, 330, 500, 900, 20000 };

static int lvaSquare(const int* bd, int sq, int side) {
    int i, t, p, d;
    int pn = MAKE_PC(side, PAWN), kn = MAKE_PC(side, KNIGHT), bi = MAKE_PC(side, BISHOP);
    int rk = MAKE_PC(side, ROOK), qu = MAKE_PC(side, QUEEN), kg = MAKE_PC(side, KING);
    int qsq = 0;

    if (side == WHITE) { if (bd[sq - 11] == pn) return sq - 11; if (bd[sq - 9] == pn) return sq - 9; }
    else               { if (bd[sq + 11] == pn) return sq + 11; if (bd[sq + 9] == pn) return sq + 9; }

    for (i = 0; i < 8; i++) {
        t = sq + knightDirs[i];
        if (bd[t] == kn) return t;
    }
    for (i = 0; i < 4; i++) {                     /* diagonals: bishop first */
        d = bishopDirs[i];
        for (t = sq + d; ; t += d) {
            p = bd[t];
            if (p == EMPTY) continue;
            if (p == bi) return t;
            if (p == qu && !qsq) qsq = t;
            break;
        }
    }
    for (i = 0; i < 4; i++) {                     /* lines: rook before queen */
        d = rookDirs[i];
        for (t = sq + d; ; t += d) {
            p = bd[t];
            if (p == EMPTY) continue;
            if (p == rk) return t;
            if (p == qu && !qsq) qsq = t;
            break;
        }
    }
    if (qsq) return qsq;
    for (i = 0; i < 8; i++) {
        t = sq + kingDirs[i];
        if (bd[t] == kg) return t;
    }
    return 0;
}

static int see(BOARD* b, MOVE m) {
    int bd[120], gain[40], d = 0;
    int from = MV_FROM(m), to = MV_TO(m), type = MV_TYPE(m);
    int piece = b->board[from], side = COLOR_OF(piece), captured, fromSq;

    memcpy(bd, b->board, sizeof(bd));
    if (type == EP_CAPTURE) {
        captured = (side == WHITE) ? bP : wP;
        bd[to + ((side == WHITE) ? DOWN : UP)] = EMPTY;
    } else {
        captured = bd[to];
    }
    gain[0] = (captured != EMPTY) ? seeVal[captured] : 0;
    if (IS_PROMO(type)) {
        static const int promoBase[4] = { KNIGHT, BISHOP, ROOK, QUEEN };
        piece = MAKE_PC(side, promoBase[type & 3]);
        gain[0] += seeVal[piece] - 100;
    }
    fromSq = from;

    for (;;) {
        d++;
        gain[d] = seeVal[piece] - gain[d - 1];        /* if this piece gets taken */
        bd[fromSq] = EMPTY;
        side ^= 1;
        fromSq = lvaSquare(bd, to, side);
        if (!fromSq) break;
        piece = bd[fromSq];
        if (BASE(piece) == KING) {                    /* king may not take a defended piece */
            bd[fromSq] = EMPTY;
            if (lvaSquare(bd, to, side ^ 1)) { bd[fromSq] = piece; break; }
            bd[fromSq] = piece;
        }
        if (d >= 38) break;
    }
    while (--d)
        gain[d - 1] = -(-gain[d - 1] > gain[d] ? -gain[d - 1] : gain[d]);
    return gain[0];
}

/* -------------------------------------------------------------- ordering */

#define SCORE_TT        3000000
#define SCORE_GOODCAP   2000000
#define SCORE_QPROMO    1900000
#define SCORE_KILLER1    800000
#define SCORE_KILLER2    700000
#define SCORE_COUNTER    600000
#define SCORE_BADCAP   (-1000000)
#define SCORE_UNDERPRO (-1500000)
#define HIST_MAX          16384

/* ---------------------------------------------------- V4 correction history
 * Remembers, per pawn structure, how far the net's static eval has been off
 * from what the search later found, and corrects the static eval by that
 * amount. Learns during the game; one table per thread.
 * ------------------------------------------------------------------------ */
#define CORR_SIZE  16384
#define CORR_GRAIN 256
#define CORR_MAX   (CORR_GRAIN * 64)

static int corrected(SEARCH_INFO* info, BOARD* b, int raw) {
    int v;
    if (!optCorr || !info->corr) return raw;
    v = raw + info->corr[b->stm * CORR_SIZE + (int)(b->pawnHash & (CORR_SIZE - 1))] / CORR_GRAIN;
    if (v >  MATE_BOUND - 1) v =  MATE_BOUND - 1;
    if (v < -MATE_BOUND + 1) v = -MATE_BOUND + 1;
    return v;
}

static void corrUpdate(SEARCH_INFO* info, BOARD* b, int depth, int diff) {
    int* e;
    int w = depth + 1 < 16 ? depth + 1 : 16;
    if (!optCorr || !info->corr) return;
    e = &info->corr[b->stm * CORR_SIZE + (int)(b->pawnHash & (CORR_SIZE - 1))];
    *e = (*e * (256 - w) + diff * CORR_GRAIN * w) / 256;
    if (*e >  CORR_MAX) *e =  CORR_MAX;
    if (*e < -CORR_MAX) *e = -CORR_MAX;
}

/* ------------------------------------------------------- capture history */
static int* capHistPtr(SEARCH_INFO* info, BOARD* b, MOVE m) {
    int type = MV_TYPE(m), to = MV_TO(m);
    int victim = (type == EP_CAPTURE) ? PAWN : BASE(b->board[to]);
    return &info->capHist[b->board[MV_FROM(m)]][sq_to64(to)][victim];
}


static int contIdx(int piece, int sq120) { return piece * 64 + sq_to64(sq120); }

static int scoreMove(BOARD* b, MOVE m, MOVE ttMove, SEARCH_INFO* info, int ply, MOVE counterMove,
                     const int* c1, const int* c2) {
    int type, victim, attacker, mvvlva;
    if (m == ttMove) return optSEE || optOrdering ? SCORE_TT : 2000000;
    type = MV_TYPE(m);
    if (IS_CAP(type)) {
        victim   = (type == EP_CAPTURE) ? wP : b->board[MV_TO(m)];
        attacker = b->board[MV_FROM(m)];
        mvvlva   = pieceValue[victim] * 10 - pieceValue[attacker];
        if (!optSEE) return 1000000 + mvvlva;                      /* V1 */
        if (optCapHist) mvvlva += *capHistPtr(info, b, m) / 16;
        if (IS_PROMO(type) && (type & 3) == 3) return SCORE_GOODCAP + 9000 + mvvlva;
        /* taking an equal or bigger piece can never lose material: no SEE needed */
        if (seeVal[victim] >= seeVal[attacker]) return SCORE_GOODCAP + mvvlva;
        return (see(b, m) >= 0 ? SCORE_GOODCAP : SCORE_BADCAP) + mvvlva;
    }
    if (IS_PROMO(type)) {
        if (!optSEE && !optOrdering) return 900000;                  /* V1 */
        return (type & 3) == 3 ? SCORE_QPROMO : SCORE_UNDERPRO;
    }
    if (m == info->killers[ply][0]) return SCORE_KILLER1;
    if (m == info->killers[ply][1]) return SCORE_KILLER2;
    if (optOrdering && m && m == counterMove) return SCORE_COUNTER;
    {
        int pc = b->board[MV_FROM(m)], to = MV_TO(m), h = info->history[pc][to];
        if (c1 || c2) {
            int k = contIdx(pc, to);
            if (c1) h += c1[k];
            if (c2) h += c2[k];
        }
        return h;
    }
}

static void sortMoves(BOARD* b, MOVE* moves, int n, MOVE ttMove, SEARCH_INFO* info, int ply) {
    int scores[MAX_MOVES], i, j;
    for (i = 0; i < n; i++) scores[i] = scoreMove(b, moves[i], ttMove, info, ply, 0, NULL, NULL);
    for (i = 1; i < n; i++) {
        MOVE m = moves[i];
        int  s = scores[i];
        j = i - 1;
        while (j >= 0 && scores[j] < s) {
            scores[j + 1] = scores[j];
            moves[j + 1]  = moves[j];
            j--;
        }
        scores[j + 1] = s;
        moves[j + 1]  = m;
    }
}

/* history with "gravity": stays inside +-HIST_MAX, recent results count most */
static void histAdd(int* h, int bonus) {
    int ab = bonus < 0 ? -bonus : bonus;
    *h += bonus - *h * ab / HIST_MAX;
}

static void scoreMoves(BOARD* b, const MOVE* moves, int* scores, int n, MOVE ttMove,
                       SEARCH_INFO* info, int ply, MOVE counterMove, const int* c1, const int* c2) {
    int i;
    for (i = 0; i < n; i++) scores[i] = scoreMove(b, moves[i], ttMove, info, ply, counterMove, c1, c2);
}

static void pickNext(MOVE* moves, int* scores, int n, int i) {
    int j, bi = i, bs = scores[i];
    MOVE m;
    for (j = i + 1; j < n; j++) if (scores[j] > bs) { bs = scores[j]; bi = j; }
    if (bi == i) return;
    m = moves[bi];
    memmove(moves + i + 1, moves + i, sizeof(MOVE) * (size_t)(bi - i));
    memmove(scores + i + 1, scores + i, sizeof(int) * (size_t)(bi - i));
    moves[i] = m;
    scores[i] = bs;
}

/* ============================================================ threads */

typedef struct {
    BOARD*      board;
    SEARCH_INFO info;
    int*        cont;
    int*        corr;
    char        accComp[ACC_PLIES][2];
    short*      accMem;
    int         accH;
} THREAD;

static THREAD       g_th[MAX_THREADS];
static int          g_numThreads = 1;
static int          g_active = 1;          /* threads in the current search */
static volatile int g_stop = 0;

static long totalNodes(void) {
    long s = 0;
    int i;
    for (i = 0; i < g_active; i++) s += *(volatile long*)&g_th[i].info.nodes;
    return s;
}

/* make sure threads 0..n-1 have their memory (accumulators sized for the net) */
static int threadsPrepare(int n) {
    int i;
    for (i = 0; i < n; i++) {
        THREAD* t = &g_th[i];
        if (!t->board) {
            t->board = (BOARD*)calloc(1, sizeof(BOARD));
            if (!t->board) return i;
        }
        if (!t->cont) {
            t->cont = (int*)calloc(CONT_SIZE, sizeof(int));
            if (!t->cont) return i;
        }
        if (!t->corr) {
            t->corr = (int*)calloc(2 * CORR_SIZE, sizeof(int));
            if (!t->corr) return i;
        }
        if (nnLoaded && t->accH != nnH) {
            afree(t->accMem);
            t->accMem = (short*)amalloc(sizeof(short) * (size_t)ACC_PLIES * 2 * (size_t)nnH);
            t->accH = t->accMem ? nnH : 0;
            if (!t->accMem) return i;
        }
    }
    return n;
}

/* copy the UCI position into a thread's own board */
static void threadSetup(THREAD* t, int id, const BOARD* src, const SEARCH_INFO* proto) {
    memcpy(t->board, src, sizeof(BOARD));
    t->board->accComp = t->accComp;
    t->board->accMem  = t->accMem;
    t->board->accBase = src->hisPly;
    memset(t->accComp, 0, sizeof(t->accComp));
    /* compute the root accumulators once: every node below can then be
       reached by replaying move deltas instead of a full refresh */
    if (nnLoaded && t->accMem && t->accH == nnH) {
        nnRefresh(t->board, 0, WHITE);
        nnRefresh(t->board, 0, BLACK);
    }
    memcpy(&t->info, proto, sizeof(SEARCH_INFO));
    t->info.id     = id;
    t->info.cont   = t->cont;
    t->info.corr   = t->corr;
    t->info.silent = proto->silent || id > 0;
    t->info.nodes  = 0;
    t->info.stopped = 0;
    t->info.bestMove = 0;
}

/* --------------------------------------------------------------- search */

static int timeUp(SEARCH_INFO* info) {
    if (info->timeLimit <= 0) return 0;
    return nowMs() - info->startTime >= info->timeLimit;
}

/* ----------------------------------------------- GUI commands during search */
static int  g_quit = 0;          /* "quit" arrived during a search            */
static char g_pending[8192];     /* other command that arrived mid-search:     */
                                 /* run by uciLoop once the search is over     */

static int inputWaiting(void) {
#ifdef _WIN32
    static int init = 0, pipe = 0;
    static HANDLE inh;
    DWORD dw;
    if (!init) {
        init = 1;
        inh = GetStdHandle(STD_INPUT_HANDLE);
        pipe = !GetConsoleMode(inh, &dw);
        if (!pipe) {
            SetConsoleMode(inh, dw & ~(ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT));
            FlushConsoleInputBuffer(inh);
        }
    }
    if (pipe) {
        if (!PeekNamedPipe(inh, NULL, 0, NULL, &dw, NULL)) return 1;   /* GUI closed the pipe */
        return dw > 0;
    }
    GetNumberOfConsoleInputEvents(inh, &dw);
    return dw > 1;
#else
    fd_set readfds;
    struct timeval tv;
    FD_ZERO(&readfds);
    FD_SET(fileno(stdin), &readfds);
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    select(fileno(stdin) + 1, &readfds, NULL, NULL, &tv);
    return FD_ISSET(fileno(stdin), &readfds);
#endif
}

/* Handle one command that arrived while thinking. Returns 1 if it means "stop". */
static int handleSearchCommand(void) {
    char cmd[8192];
    if (!fgets(cmd, sizeof(cmd), stdin)) { g_quit = 1; return 1; }    /* stdin closed */
    cmd[strcspn(cmd, "\r\n")] = 0;
    if (!strncmp(cmd, "stop", 4))  return 1;
    if (!strncmp(cmd, "quit", 4))  { g_quit = 1; return 1; }
    if (!strncmp(cmd, "isready", 7)) { printf("readyok\n"); fflush(stdout); return 0; }
    if (cmd[0] && !g_pending[0]) {                 /* keep it for later */
        strncpy(g_pending, cmd, sizeof(g_pending) - 1);
        g_pending[sizeof(g_pending) - 1] = 0;
    }
    return 0;
}

/* Called every 2048 nodes. Helpers only watch the shared stop flag; the main
   thread also checks time, node limit and the GUI. */
static int checkUp(SEARCH_INFO* info) {
    if (info->stopped) return 1;
    if (g_stop) { info->stopped = 1; return 1; }
    if (info->id != 0) return 0;
    if (timeUp(info)) info->stopped = 1;
    else if (info->nodeLimit > 0 && totalNodes() >= info->nodeLimit) info->stopped = 1;
    else if (!info->silent && !g_pending[0] && inputWaiting() && handleSearchCommand()) info->stopped = 1;
    if (info->stopped) g_stop = 1;
    return info->stopped;
}

/* --------------------------------------------------------------- quiesce */

static int quiesce(BOARD* b, int alpha, int beta, int ply, SEARCH_INFO* info) {
    MOVE moves[MAX_MOVES], bestMove = 0;
    int scores[MAX_MOVES];
    int n, i, us, stand, alphaOrig = alpha, qChecked = -1;

    if (ply < MAX_PLY) info->pvLen[ply] = ply;     /* no stale PV from a sibling */
    if ((info->nodes & 2047) == 0 && checkUp(info)) return 0;
    info->nodes++;

    if (optQsTT && beta - alpha == 1) {
        TTDATA te;
        if (ttRead(b->hash, &te)) {
            int sc = ttScoreFrom(te.score, ply);
            if (te.flag == TT_EXACT) return sc;
            if (te.flag == TT_ALPHA && sc <= alpha) return alpha;
            if (te.flag == TT_BETA  && sc >= beta)  return beta;
        }
    }

    stand = corrected(info, b, eval(b));
    if (stand >= beta) return beta;
    if (stand > alpha) alpha = stand;
    if (b->hisPly >= HIST_SIZE - 2 || b->hisPly - b->accBase >= ACC_PLIES - 2) return alpha;

    n  = genMovesEx(b, moves, 1);              /* captures and promotions only */
    us = b->stm;
    scoreMoves(b, moves, scores, n, 0, info, 0, 0, NULL, NULL);

    for (i = 0; i < n; i++) {
        int score, type;
        pickNext(moves, scores, n, i);
        type = MV_TYPE(moves[i]);

        /* SEE: the rest are losing captures / under-promotions -> skip them */
        if (optSEE && scores[i] < 0) break;

        /* delta pruning: even winning this piece cannot lift us to alpha */
        if (optDelta && !IS_PROMO(type)) {
            int victim = (type == EP_CAPTURE) ? wP : b->board[MV_TO(moves[i])];
            if (stand + seeVal[victim] + 200 <= alpha) continue;
        }

        {
            int from = MV_FROM(moves[i]);
            int need = BASE(b->board[from]) == KING || type == EP_CAPTURE
                       || lineDir[b->kingSquare[us]][from];
            if (!need) {                           /* in check? (asked once per node) */
                if (qChecked < 0) qChecked = inCheck(b, us);
                need = qChecked;
            }
            makeMove(b, moves[i]);
            if (need && inCheck(b, us)) { undoMove(b); continue; }
        }
        score = -quiesce(b, -beta, -alpha, ply + 1, info);
        undoMove(b);
        if (info->stopped) return 0;
        if (score >= beta) {
            if (optQsTT) ttStore(b->hash, 0, ply, beta, TT_BETA, moves[i]);
            return beta;
        }
        if (score > alpha) { alpha = score; bestMove = moves[i]; }
    }
    if (optQsTT) ttStore(b->hash, 0, ply, alpha, alpha > alphaOrig ? TT_EXACT : TT_ALPHA, bestMove);
    return alpha;
}

/* ---------------------------------------------------------- main search */

static int alphabeta(BOARD* b, int depth, int ply, int alpha, int beta, SEARCH_INFO* info,
                     int allowNull, MOVE excluded) {
    MOVE moves[MAX_MOVES], ttMove = 0, bestMove = 0, counterMove = 0;
    MOVE quietsTried[MAX_MOVES], capsTried[MAX_MOVES];
    int scores[MAX_MOVES];
    int n, i, us, legal = 0, searched = 0, nQuiets = 0, nCaps = 0, best = -INF_SCORE;
    int checked, rawEval = EVAL_NONE, staticEval = EVAL_NONE, pruneEval, improving = 0;
    int alphaOrig, ttHit = 0, ttSc = 0, stage;
    int *c1 = NULL, *c2 = NULL;
    TTDATA te;
    int isPV = (beta - alpha) > 1;

    memset(&te, 0, sizeof(te));
    if (ply < MAX_PLY) info->pvLen[ply] = ply;     /* no stale PV from a sibling */
    if ((info->nodes & 2047) == 0 && checkUp(info)) return 0;

    if (ply > 0) {
        if (isRepetition(b) || b->fiftyMove >= 100) return 0;
        if (insufficientMaterial(b)) return 0;

        /* mate distance pruning: a shorter mate was already found */
        if (alpha < -MATE_SCORE + ply)    alpha = -MATE_SCORE + ply;
        if (beta  >  MATE_SCORE - ply - 1) beta =  MATE_SCORE - ply - 1;
        if (alpha >= beta) return alpha;

        if (!excluded && ttRead(b->hash, &te)) {
            ttHit  = 1;
            ttMove = te.move;
            ttSc   = ttScoreFrom(te.score, ply);
            if (!isPV && te.depth > 0 && te.depth >= depth) {
                if (te.flag == TT_EXACT) return ttSc;
                if (te.flag == TT_ALPHA && ttSc <= alpha) return alpha;
                if (te.flag == TT_BETA  && ttSc >= beta)  return beta;
            }
        }
    }
    alphaOrig = alpha;

    checked = inCheck(b, b->stm);
    if (checked) depth++;

    if (depth <= 0) return quiesce(b, alpha, beta, ply, info);

    info->nodes++;

    if (ply > MAX_PLY - 2 || b->hisPly >= HIST_SIZE - 2) return eval(b);

    /* static eval of this node (net + correction history);
       "improving" = better than 2 plies ago */
    if (!checked) {
        rawEval    = eval(b);
        staticEval = corrected(info, b, rawEval);
        improving  = ply >= 2 && info->evalStack[ply - 2] != EVAL_NONE
                     && staticEval > info->evalStack[ply - 2];
    }
    info->evalStack[ply] = staticEval;

    /* TT score is a better guess than the raw eval when its bound agrees */
    pruneEval = staticEval;
    if (optTTEval && ttHit && staticEval != EVAL_NONE && ttSc > -MATE_BOUND && ttSc < MATE_BOUND) {
        if (te.flag == TT_EXACT
            || (te.flag == TT_BETA  && ttSc > staticEval)
            || (te.flag == TT_ALPHA && ttSc < staticEval))
            pruneEval = ttSc;
    }

    /* reverse futility pruning: far above beta, a quiet position stays there */
    if (optRFP && !isPV && !checked && !excluded && ply > 0 && depth <= tRfpDepth
        && beta > -MATE_BOUND && beta < MATE_BOUND
        && pruneEval - (tRfpMargin * depth - (improving ? tRfpImp : 0)) >= beta)
        return pruneEval;

    /* null move: give the opponent a free move; still >= beta -> prune */
    if (!isPV && !checked && !excluded && depth >= 3 && ply > 0 && hasNonPawnMaterial(b, b->stm)
        && (!optNullPlus || (allowNull && pruneEval >= beta))) {
        int R;
        int nullScore;
        if (optNullPlus) {
            int e = (pruneEval - beta) / tNmpEvalDiv;
            R = tNmpBase + depth / tNmpDiv + (e < 3 ? e : 3);
        } else R = 2 + (depth > 6 ? 1 : 0);
        info->stPiece[ply] = 0;
        makeNullMove(b);
        nullScore = -alphabeta(b, depth - 1 - R, ply + 1, -beta, -beta + 1, info, 0, 0);
        undoNullMove(b);
        if (info->stopped) return 0;
        if (nullScore >= beta && nullScore < MATE_BOUND) return beta;
    }

    /* internal iterative reduction: no hash move -> search a bit shallower */
    if (optIIR && depth >= 4 && !ttMove && !excluded) depth--;

    /* counter-move: our known best reply to the opponent's last move */
    if (optOrdering && b->hisPly > 0) {
        MOVE prev = b->history[b->hisPly - 1].move;
        if (prev) counterMove = info->counter[b->board[MV_TO(prev)]][MV_TO(prev)];
    }

    /* continuation history rows for the last two moves */
    if (optCont && info->cont) {
        if (ply >= 1 && info->stPiece[ply - 1])
            c1 = info->cont + (size_t)(info->stPiece[ply - 1] * 64 + info->stTo[ply - 1]) * CONT_DIM;
        if (ply >= 2 && info->stPiece[ply - 2])
            c2 = info->cont + (size_t)(info->stPiece[ply - 2] * 64 + info->stTo[ply - 2]) * CONT_DIM;
    }

    us = b->stm;

    /* stage 0: try the hash move before generating anything. If it cuts
       off (the usual case at cut nodes), no move list is ever built. */
    if (ttMove && ttMove != excluded && isPseudoLegal(b, ttMove)) {
        moves[0] = ttMove;
        scores[0] = SCORE_TT;
        n = 1;
        stage = 0;
    } else {
        ttMove = 0;
        n = genMoves(b, moves);
        scoreMoves(b, moves, scores, n, 0, info, ply, counterMove, c1, c2);
        stage = 1;
    }

    for (i = 0; ; i++) {
        int score, type, quiet, reduction = 0, givesCheck, isKiller, ext = 0, newDepth, mpiece;
        MOVE m;

        if (i >= n) {
            int k, j = n;
            if (stage == 1) break;
            stage = 1;                               /* stage 1: everything else */
            n += genMoves(b, moves + n);
            for (k = j; k < n; k++) {
                if (moves[k] == ttMove) { moves[k] = moves[--n]; k--; continue; }
            }
            scoreMoves(b, moves + j, scores + j, n - j, 0, info, ply, counterMove, c1, c2);
            if (i >= n) break;
        }

        pickNext(moves, scores, n, i);
        m = moves[i];
        if (m == excluded) continue;
        type  = MV_TYPE(m);
        quiet = !IS_CAP(type) && !IS_PROMO(type);
        isKiller = (m == info->killers[ply][0] || m == info->killers[ply][1]);
        mpiece = b->board[MV_FROM(m)];

        /* SEE pruning of clearly losing captures near the leaves */
        if (optSEE && !isPV && !checked && searched > 0 && best > -MATE_BOUND
            && depth <= tSeeDepth && IS_CAP(type) && scores[i] < 0 && see(b, m) < -tSeeMargin * depth)
            continue;

        /* singular extension: if every other move is clearly worse than the
           hash move, the hash move gets one extra ply. If even without it
           we beat beta, several moves refute -> cut (multi-cut). */
        if (optSingular && m == ttMove && ttHit && !excluded && ply > 0 && depth >= tSingDepth
            && ply < 2 * info->rootDepth && te.depth >= depth - 3
            && (te.flag == TT_BETA || te.flag == TT_EXACT)
            && ttSc > -MATE_BOUND && ttSc < MATE_BOUND) {
            int sBeta = ttSc - tSingMargin * depth;
            int s = alphabeta(b, (depth - 1) / 2, ply, sBeta - 1, sBeta, info, 0, m);
            if (info->stopped) return 0;
            if (s < sBeta) ext = 1;
            else if (sBeta >= beta) return sBeta;
        }

        {
            int need = mayExposeKing(b, m, checked);
            makeMove(b, m);
            if (need && inCheck(b, us)) { undoMove(b); continue; }
        }
        legal++;
        givesCheck = givesCheckAfter(b, m);

        /* prune hopeless quiet moves once one move has been searched */
        if (ply > 0 && quiet && !checked && !givesCheck && searched > 0 && best > -MATE_BOUND) {
            if (optLMP && depth <= 7
                && nQuiets >= (tLmpBase + depth * depth) / (improving ? 1 : 2)) {
                undoMove(b);
                continue;
            }
            if (optFutility && depth <= tFutDepth && !isKiller && staticEval != EVAL_NONE
                && pruneEval + tFutBase + tFutMult * depth <= alpha) {
                undoMove(b);
                continue;
            }
        }

        searched++;
        if (quiet) { if (nQuiets < MAX_MOVES) quietsTried[nQuiets++] = m; }
        else if (IS_CAP(type) && nCaps < MAX_MOVES) capsTried[nCaps++] = m;
        info->stPiece[ply] = mpiece;
        info->stTo[ply]    = sq_to64(MV_TO(m));
        newDepth = depth - 1 + ext;

        if (searched == 1) {
            score = -alphabeta(b, newDepth, ply + 1, -beta, -alpha, info, 1, 0);
        } else {
            if (optLMRPlus) {
                if (depth >= 3 && !checked && (quiet || scores[i] < 0)) {
                    int hist = 0;
                    if (quiet) {
                        int k = contIdx(mpiece, MV_TO(m));
                        hist = info->history[mpiece][MV_TO(m)];
                        if (c1) hist += c1[k];
                        if (c2) hist += c2[k];
                    }
                    reduction = lmrTable[depth < 64 ? depth : 63][searched < 64 ? searched : 63];
                    if (isPV)          reduction--;
                    if (isKiller || (counterMove && m == counterMove)) reduction--;
                    if (givesCheck)    reduction--;
                    if (!improving)    reduction++;
                    reduction -= hist / ((c1 || c2) ? 2 * tHistLmrDiv : tHistLmrDiv);
                    if (reduction < 0) reduction = 0;
                    if (reduction > newDepth - 1) reduction = newDepth - 1;
                }
            } else if (depth >= 3 && searched > 3 && quiet && !checked && !givesCheck) {  /* V1 */
                reduction = 1;
                if (searched > 6) reduction++;
                if (depth > 6 && searched > 10) reduction++;
                if (reduction > newDepth - 1) reduction = newDepth - 1;
            }

            score = -alphabeta(b, newDepth - reduction, ply + 1, -alpha - 1, -alpha, info, 1, 0);

            if (score > alpha && reduction > 0)
                score = -alphabeta(b, newDepth, ply + 1, -alpha - 1, -alpha, info, 1, 0);

            if (score > alpha && score < beta)
                score = -alphabeta(b, newDepth, ply + 1, -beta, -alpha, info, 1, 0);
        }

        undoMove(b);
        if (info->stopped) return 0;

        if (score > best) { best = score; bestMove = m; }
        if (score > alpha) {
            alpha = score;
            if (ply < MAX_PLY - 1) {
                int j;
                info->pv[ply][ply] = m;
                for (j = ply + 1; j < info->pvLen[ply + 1]; j++)
                    info->pv[ply][j] = info->pv[ply + 1][j];
                info->pvLen[ply] = info->pvLen[ply + 1];
            }
            if (alpha >= beta) {
                int bonus = depth * depth < 1200 ? depth * depth : 1200, q;
                if (quiet && ply < MAX_PLY) {
                    info->killers[ply][1] = info->killers[ply][0];
                    info->killers[ply][0] = m;
                    if (optOrdering) {
                        histAdd(&info->history[mpiece][MV_TO(m)], bonus);
                        if (c1) histAdd(&c1[contIdx(mpiece, MV_TO(m))], bonus);
                        if (c2) histAdd(&c2[contIdx(mpiece, MV_TO(m))], bonus);
                        for (q = 0; q < nQuiets - 1; q++) {        /* tried before, failed */
                            int qp = b->board[MV_FROM(quietsTried[q])], qt = MV_TO(quietsTried[q]);
                            histAdd(&info->history[qp][qt], -bonus);
                            if (c1) histAdd(&c1[contIdx(qp, qt)], -bonus);
                            if (c2) histAdd(&c2[contIdx(qp, qt)], -bonus);
                        }
                        if (b->hisPly > 0 && b->history[b->hisPly - 1].move) {
                            MOVE prev = b->history[b->hisPly - 1].move;
                            info->counter[b->board[MV_TO(prev)]][MV_TO(prev)] = m;
                        }
                    } else {
                        info->history[mpiece][MV_TO(m)] += depth * depth;   /* V1 */
                    }
                }
                if (optCapHist) {                    /* captures that failed get a malus */
                    if (IS_CAP(type)) histAdd(capHistPtr(info, b, m), bonus);
                    for (q = 0; q < nCaps; q++)
                        if (capsTried[q] != m) histAdd(capHistPtr(info, b, capsTried[q]), -bonus);
                }
                /* correction history: the true score was at least beta */
                if (!checked && !excluded && quiet && rawEval != EVAL_NONE && beta > staticEval
                    && beta < MATE_BOUND && beta > -MATE_BOUND)
                    corrUpdate(info, b, depth, beta - rawEval);
                if (!excluded) ttStore(b->hash, depth, ply, beta, TT_BETA, m);
                return beta;
            }
        }
    }

    if (legal == 0) return excluded ? alpha : (checked ? (-MATE_SCORE + ply) : 0);
    if (searched == 0) return alpha;

    /* correction history: exact score, or an upper bound below the eval */
    if (!checked && !excluded && rawEval != EVAL_NONE && best > -MATE_BOUND && best < MATE_BOUND
        && (!bestMove || !IS_CAP(MV_TYPE(bestMove)))
        && (alpha > alphaOrig || best < staticEval))
        corrUpdate(info, b, depth, best - rawEval);

    if (!excluded)
        ttStore(b->hash, depth, ply, best, (alpha > alphaOrig) ? TT_EXACT : TT_ALPHA, bestMove);
    return best;
}

static void moveToStr(MOVE move, char* str) {
    int type = MV_TYPE(move);
    sq_to_str(MV_FROM(move), str);
    sq_to_str(MV_TO(move), str + 2);
    if (IS_PROMO(type)) { str[4] = "nbrq"[type & 3]; str[5] = '\0'; }
    else str[4] = '\0';
}

/* Iterative deepening for one thread. The main thread (id 0) prints info
   and owns the result; helper threads search the same tree with their own
   history tables, sharing only the TT -- that is Lazy SMP. */
static void searchPosition(THREAD* t) {
    BOARD* b = t->board;
    SEARCH_INFO* info = &t->info;
    MOVE rootMoves[MAX_MOVES], bestMove = 0;
    MOVE bestPv[MAX_PLY], lastPv[MAX_PLY];
    int nRoot, d, prevScore = 0, bestPvLen = 0, lastPvLen = 0, startDepth;
    int stableIters = 0;
    double instability = 0.0;

    info->nodes = 0;
    info->stopped = 0;
    memset(info->killers, 0, sizeof(info->killers));
    memset(info->history, 0, sizeof(info->history));
    memset(info->counter, 0, sizeof(info->counter));
    memset(info->stPiece, 0, sizeof(info->stPiece));
    if (info->cont) memset(info->cont, 0, sizeof(int) * CONT_SIZE);
    info->evalStack[0] = EVAL_NONE;

    nRoot = genLegal(b, rootMoves);
    info->bestMove = 0;
    if (nRoot == 0) return;
    bestMove = rootMoves[0];
    info->bestMove = bestMove;

    /* helpers: odd ones start one ply deeper so threads spread out */
    startDepth = (info->id > 0 && (info->id & 1)) ? 2 : 1;
    if (startDepth > info->depth) startDepth = info->depth;

    for (d = startDepth; d <= info->depth; d++) {
        MOVE iterBest = 0;
        int bestScore = -INF_SCORE;
        int window = tAspWindow, alpha, beta;
        double ms;

        if (g_stop) break;
        info->rootDepth = d;

        if (d >= 4 && prevScore > -MATE_SCORE + MAX_PLY
                   && prevScore <  MATE_SCORE - MAX_PLY) {
            alpha = prevScore - window;
            beta  = prevScore + window;
        } else {
            alpha = -INF_SCORE;
            beta  =  INF_SCORE;
        }

        for (;;) {
            int i, curAlpha = alpha;
            iterBest = 0;
            bestScore = -INF_SCORE;
            bestPvLen = 0;

            sortMoves(b, rootMoves, nRoot, bestMove, info, 0);

            for (i = 0; i < nRoot; i++) {
                int score;

                if (!info->silent && nowMs() - info->startTime > 400.0) {
                    char cm[6];
                    moveToStr(rootMoves[i], cm);
                    printf("info depth %d currmove %s currmovenumber %d\n", d, cm, i + 1);
                    fflush(stdout);
                }

                info->stPiece[0] = b->board[MV_FROM(rootMoves[i])];
                info->stTo[0]    = sq_to64(MV_TO(rootMoves[i]));
                makeMove(b, rootMoves[i]);
                score = -alphabeta(b, d - 1, 1, -beta, -curAlpha, info, 1, 0);
                undoMove(b);
                if (info->stopped) break;
                if (score > bestScore) {
                    int j;
                    bestScore = score;
                    iterBest = rootMoves[i];
                    bestPvLen = 1;
                    bestPv[0] = rootMoves[i];
                    for (j = 1; j < info->pvLen[1] && j < MAX_PLY - 1; j++)
                        bestPv[bestPvLen++] = info->pv[1][j];
                }
                if (score > curAlpha) curAlpha = score;
            }

            if (info->stopped) break;

            if (bestScore <= alpha) {
                window *= 3;
                alpha = bestScore - window;
                if (alpha < -INF_SCORE) alpha = -INF_SCORE;
                continue;
            }
            if (bestScore >= beta) {
                window *= 3;
                beta = bestScore + window;
                if (beta > INF_SCORE) beta = INF_SCORE;
                continue;
            }
            break;
        }

        if (info->stopped && iterBest == 0) break;
        if (iterBest) {
            if (iterBest != bestMove && d > startDepth) { instability += 1.0; stableIters = 0; }
            else stableIters++;
            bestMove = iterBest;
        }
        info->bestMove = bestMove;
        {
            int drop = (d > startDepth) ? prevScore - bestScore : 0;
            prevScore = bestScore;
            ms = nowMs() - info->startTime;
            /* only the main thread decides when to stop; helpers keep
               searching until it does (V3 let them quit early) */
            if (info->id == 0 && info->softLimit > 0 && !info->stopped) {
                double f = 1.0;
                if (optTimeMgr) {
                    f = 0.75 + 0.45 * instability;          /* best move keeps changing: think longer */
                    if (stableIters >= 4) f *= 0.8;         /* same move many times: move sooner     */
                    if (drop > 25) f *= (drop > 75 ? 1.6 : 1.3);  /* score falling: look harder  */
                    if (f < 0.5) f = 0.5;
                    if (f > 2.5) f = 2.5;
                }
                instability *= 0.5;
                if (ms > info->softLimit * f) info->stopSoft = 1;
            }
        }

        ms = nowMs() - info->startTime;
        if (!info->silent) {
            long nodes = totalNodes();
            long nps = (long)(nodes / ((ms / 1000.0) + 0.001));
            char buf[8];
            int k;

            if (bestPvLen > 0) {
                lastPvLen = bestPvLen;
                for (k = 0; k < bestPvLen; k++) lastPv[k] = bestPv[k];
            }

            printf("info depth %d ", d);
            if (bestScore > MATE_SCORE - MAX_PLY)
                printf("score mate %d ", (MATE_SCORE - bestScore + 1) / 2);
            else if (bestScore < -MATE_SCORE + MAX_PLY)
                printf("score mate %d ", -((MATE_SCORE + bestScore + 1) / 2));
            else
                printf("score cp %d ", bestScore);

            printf("nodes %ld nps %ld time %d hashfull %d pv",
                   nodes, nps, (int)ms, ttHashfull());
            for (k = 0; k < lastPvLen; k++) {
                moveToStr(lastPv[k], buf);
                printf(" %s", buf);
            }
            printf("\n");
            fflush(stdout);
        }
        if (bestScore > MATE_SCORE - MAX_PLY) break;

        if (info->stopped || info->stopSoft) break;
    }
}

/* ------------------------------------------------------- thread launch */

#ifdef _WIN32
static DWORD WINAPI threadMain(LPVOID arg) { searchPosition((THREAD*)arg); return 0; }
typedef HANDLE K7THREAD;
static int threadStart(K7THREAD* h, THREAD* t) {
    *h = CreateThread(NULL, THREAD_STACK, threadMain, t, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    return *h != NULL;
}
static void threadJoin(K7THREAD h) { WaitForSingleObject(h, INFINITE); CloseHandle(h); }
#else
static void* threadMain(void* arg) { searchPosition((THREAD*)arg); return NULL; }
typedef pthread_t K7THREAD;
static int threadStart(K7THREAD* h, THREAD* t) {
    pthread_attr_t a;
    int ok;
    pthread_attr_init(&a);
    pthread_attr_setstacksize(&a, THREAD_STACK);
    ok = pthread_create(h, &a, threadMain, t) == 0;
    pthread_attr_destroy(&a);
    return ok;
}
static void threadJoin(K7THREAD h) { pthread_join(h, NULL); }
#endif

/* Run a search on `src` with `nThreads` threads; returns the main thread's move.
   Every search thread (main included) gets a 16 MB stack. */
static MOVE runSearch(BOARD* src, SEARCH_INFO* proto, int nThreads) {
    K7THREAD h[MAX_THREADS];
    int started[MAX_THREADS], i;

    if (nThreads < 1) nThreads = 1;
    if (nThreads > MAX_THREADS) nThreads = MAX_THREADS;
    nThreads = threadsPrepare(nThreads);
    if (nThreads < 1) return 0;

    g_stop = 0;
    g_active = nThreads;
    proto->startTime = nowMs();
    for (i = 0; i < nThreads; i++) threadSetup(&g_th[i], i, src, proto);

    for (i = 1; i < nThreads; i++) started[i] = threadStart(&h[i], &g_th[i]);
    if (threadStart(&h[0], &g_th[0])) threadJoin(h[0]);
    else searchPosition(&g_th[0]);              /* could not spawn: run here */

    g_stop = 1;                                 /* main is done: stop helpers */
    for (i = 1; i < nThreads; i++) if (started[i]) threadJoin(h[i]);

    proto->stopped = g_th[0].info.stopped;
    proto->nodes   = totalNodes();
    return g_th[0].info.bestMove;
}

/* ------------------------------------------------------------------ UCI */

static MOVE strToMove(BOARD* b, const char* str) {
    MOVE moves[MAX_MOVES];
    int n, i, from, to;

    if (strlen(str) < 4) return 0;
    from = str_to_sq(str);
    to   = str_to_sq(str + 2);
    if (from < 0 || to < 0) return 0;

    n = genLegal(b, moves);
    for (i = 0; i < n; i++) {
        if ((int)MV_FROM(moves[i]) != from || (int)MV_TO(moves[i]) != to) continue;
        if (IS_PROMO(MV_TYPE(moves[i]))) {
            char want = (strlen(str) > 4) ? (char)tolower((unsigned char)str[4]) : 'q';
            if ("nbrq"[MV_TYPE(moves[i]) & 3] != want) continue;
        }
        return moves[i];
    }
    return 0;
}

static int strieq(const char* a, const char* b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == *b;
}

static void trim(char* s) {
    size_t n = strlen(s), i = 0;
    while (n && isspace((unsigned char)s[n - 1])) s[--n] = 0;
    while (s[i] && isspace((unsigned char)s[i])) i++;
    if (i) memmove(s, s + i, n - i + 1);
}

/* The net is loaded lazily: at isready/go, or right away when NetFile is set. */
static int netTried = 0;

static void ensureNetReady(void) {
    if (nnLoaded || netTried) return;
    netTried = 1;
    if (!nnLoadAny(nnPath))
        printf("info string NO NET LOADED -- playing on material only. Set NetFile to a k7_train net (.bin/.k7nn)\n");
}

/* ================================================================
 *  POLYGLOT OPENING BOOK
 *  Standard .bin format (16-byte entries, sorted by key, big-endian).
 *  Options: OwnBook, BookFile (<embedded> or a path), BookMaxPly,
 *  BookPick (Best = highest weight, Weighted = random by weight).
 * ================================================================ */
static const U64 polyRandom[781] = {
    0x9D39247E33776D41ULL, 0x2AF7398005AAA5C7ULL, 0x44DB015024623547ULL, 0x9C15F73E62A76AE2ULL,
    0x75834465489C0C89ULL, 0x3290AC3A203001BFULL, 0x0FBBAD1F61042279ULL, 0xE83A908FF2FB60CAULL,
    0x0D7E765D58755C10ULL, 0x1A083822CEAFE02DULL, 0x9605D5F0E25EC3B0ULL, 0xD021FF5CD13A2ED5ULL,
    0x40BDF15D4A672E32ULL, 0x011355146FD56395ULL, 0x5DB4832046F3D9E5ULL, 0x239F8B2D7FF719CCULL,
    0x05D1A1AE85B49AA1ULL, 0x679F848F6E8FC971ULL, 0x7449BBFF801FED0BULL, 0x7D11CDB1C3B7ADF0ULL,
    0x82C7709E781EB7CCULL, 0xF3218F1C9510786CULL, 0x331478F3AF51BBE6ULL, 0x4BB38DE5E7219443ULL,
    0xAA649C6EBCFD50FCULL, 0x8DBD98A352AFD40BULL, 0x87D2074B81D79217ULL, 0x19F3C751D3E92AE1ULL,
    0xB4AB30F062B19ABFULL, 0x7B0500AC42047AC4ULL, 0xC9452CA81A09D85DULL, 0x24AA6C514DA27500ULL,
    0x4C9F34427501B447ULL, 0x14A68FD73C910841ULL, 0xA71B9B83461CBD93ULL, 0x03488B95B0F1850FULL,
    0x637B2B34FF93C040ULL, 0x09D1BC9A3DD90A94ULL, 0x3575668334A1DD3BULL, 0x735E2B97A4C45A23ULL,
    0x18727070F1BD400BULL, 0x1FCBACD259BF02E7ULL, 0xD310A7C2CE9B6555ULL, 0xBF983FE0FE5D8244ULL,
    0x9F74D14F7454A824ULL, 0x51EBDC4AB9BA3035ULL, 0x5C82C505DB9AB0FAULL, 0xFCF7FE8A3430B241ULL,
    0x3253A729B9BA3DDEULL, 0x8C74C368081B3075ULL, 0xB9BC6C87167C33E7ULL, 0x7EF48F2B83024E20ULL,
    0x11D505D4C351BD7FULL, 0x6568FCA92C76A243ULL, 0x4DE0B0F40F32A7B8ULL, 0x96D693460CC37E5DULL,
    0x42E240CB63689F2FULL, 0x6D2BDCDAE2919661ULL, 0x42880B0236E4D951ULL, 0x5F0F4A5898171BB6ULL,
    0x39F890F579F92F88ULL, 0x93C5B5F47356388BULL, 0x63DC359D8D231B78ULL, 0xEC16CA8AEA98AD76ULL,
    0x5355F900C2A82DC7ULL, 0x07FB9F855A997142ULL, 0x5093417AA8A7ED5EULL, 0x7BCBC38DA25A7F3CULL,
    0x19FC8A768CF4B6D4ULL, 0x637A7780DECFC0D9ULL, 0x8249A47AEE0E41F7ULL, 0x79AD695501E7D1E8ULL,
    0x14ACBAF4777D5776ULL, 0xF145B6BECCDEA195ULL, 0xDABF2AC8201752FCULL, 0x24C3C94DF9C8D3F6ULL,
    0xBB6E2924F03912EAULL, 0x0CE26C0B95C980D9ULL, 0xA49CD132BFBF7CC4ULL, 0xE99D662AF4243939ULL,
    0x27E6AD7891165C3FULL, 0x8535F040B9744FF1ULL, 0x54B3F4FA5F40D873ULL, 0x72B12C32127FED2BULL,
    0xEE954D3C7B411F47ULL, 0x9A85AC909A24EAA1ULL, 0x70AC4CD9F04F21F5ULL, 0xF9B89D3E99A075C2ULL,
    0x87B3E2B2B5C907B1ULL, 0xA366E5B8C54F48B8ULL, 0xAE4A9346CC3F7CF2ULL, 0x1920C04D47267BBDULL,
    0x87BF02C6B49E2AE9ULL, 0x092237AC237F3859ULL, 0xFF07F64EF8ED14D0ULL, 0x8DE8DCA9F03CC54EULL,
    0x9C1633264DB49C89ULL, 0xB3F22C3D0B0B38EDULL, 0x390E5FB44D01144BULL, 0x5BFEA5B4712768E9ULL,
    0x1E1032911FA78984ULL, 0x9A74ACB964E78CB3ULL, 0x4F80F7A035DAFB04ULL, 0x6304D09A0B3738C4ULL,
    0x2171E64683023A08ULL, 0x5B9B63EB9CEFF80CULL, 0x506AACF489889342ULL, 0x1881AFC9A3A701D6ULL,
    0x6503080440750644ULL, 0xDFD395339CDBF4A7ULL, 0xEF927DBCF00C20F2ULL, 0x7B32F7D1E03680ECULL,
    0xB9FD7620E7316243ULL, 0x05A7E8A57DB91B77ULL, 0xB5889C6E15630A75ULL, 0x4A750A09CE9573F7ULL,
    0xCF464CEC899A2F8AULL, 0xF538639CE705B824ULL, 0x3C79A0FF5580EF7FULL, 0xEDE6C87F8477609DULL,
    0x799E81F05BC93F31ULL, 0x86536B8CF3428A8CULL, 0x97D7374C60087B73ULL, 0xA246637CFF328532ULL,
    0x043FCAE60CC0EBA0ULL, 0x920E449535DD359EULL, 0x70EB093B15B290CCULL, 0x73A1921916591CBDULL,
    0x56436C9FE1A1AA8DULL, 0xEFAC4B70633B8F81ULL, 0xBB215798D45DF7AFULL, 0x45F20042F24F1768ULL,
    0x930F80F4E8EB7462ULL, 0xFF6712FFCFD75EA1ULL, 0xAE623FD67468AA70ULL, 0xDD2C5BC84BC8D8FCULL,
    0x7EED120D54CF2DD9ULL, 0x22FE545401165F1CULL, 0xC91800E98FB99929ULL, 0x808BD68E6AC10365ULL,
    0xDEC468145B7605F6ULL, 0x1BEDE3A3AEF53302ULL, 0x43539603D6C55602ULL, 0xAA969B5C691CCB7AULL,
    0xA87832D392EFEE56ULL, 0x65942C7B3C7E11AEULL, 0xDED2D633CAD004F6ULL, 0x21F08570F420E565ULL,
    0xB415938D7DA94E3CULL, 0x91B859E59ECB6350ULL, 0x10CFF333E0ED804AULL, 0x28AED140BE0BB7DDULL,
    0xC5CC1D89724FA456ULL, 0x5648F680F11A2741ULL, 0x2D255069F0B7DAB3ULL, 0x9BC5A38EF729ABD4ULL,
    0xEF2F054308F6A2BCULL, 0xAF2042F5CC5C2858ULL, 0x480412BAB7F5BE2AULL, 0xAEF3AF4A563DFE43ULL,
    0x19AFE59AE451497FULL, 0x52593803DFF1E840ULL, 0xF4F076E65F2CE6F0ULL, 0x11379625747D5AF3ULL,
    0xBCE5D2248682C115ULL, 0x9DA4243DE836994FULL, 0x066F70B33FE09017ULL, 0x4DC4DE189B671A1CULL,
    0x51039AB7712457C3ULL, 0xC07A3F80C31FB4B4ULL, 0xB46EE9C5E64A6E7CULL, 0xB3819A42ABE61C87ULL,
    0x21A007933A522A20ULL, 0x2DF16F761598AA4FULL, 0x763C4A1371B368FDULL, 0xF793C46702E086A0ULL,
    0xD7288E012AEB8D31ULL, 0xDE336A2A4BC1C44BULL, 0x0BF692B38D079F23ULL, 0x2C604A7A177326B3ULL,
    0x4850E73E03EB6064ULL, 0xCFC447F1E53C8E1BULL, 0xB05CA3F564268D99ULL, 0x9AE182C8BC9474E8ULL,
    0xA4FC4BD4FC5558CAULL, 0xE755178D58FC4E76ULL, 0x69B97DB1A4C03DFEULL, 0xF9B5B7C4ACC67C96ULL,
    0xFC6A82D64B8655FBULL, 0x9C684CB6C4D24417ULL, 0x8EC97D2917456ED0ULL, 0x6703DF9D2924E97EULL,
    0xC547F57E42A7444EULL, 0x78E37644E7CAD29EULL, 0xFE9A44E9362F05FAULL, 0x08BD35CC38336615ULL,
    0x9315E5EB3A129ACEULL, 0x94061B871E04DF75ULL, 0xDF1D9F9D784BA010ULL, 0x3BBA57B68871B59DULL,
    0xD2B7ADEEDED1F73FULL, 0xF7A255D83BC373F8ULL, 0xD7F4F2448C0CEB81ULL, 0xD95BE88CD210FFA7ULL,
    0x336F52F8FF4728E7ULL, 0xA74049DAC312AC71ULL, 0xA2F61BB6E437FDB5ULL, 0x4F2A5CB07F6A35B3ULL,
    0x87D380BDA5BF7859ULL, 0x16B9F7E06C453A21ULL, 0x7BA2484C8A0FD54EULL, 0xF3A678CAD9A2E38CULL,
    0x39B0BF7DDE437BA2ULL, 0xFCAF55C1BF8A4424ULL, 0x18FCF680573FA594ULL, 0x4C0563B89F495AC3ULL,
    0x40E087931A00930DULL, 0x8CFFA9412EB642C1ULL, 0x68CA39053261169FULL, 0x7A1EE967D27579E2ULL,
    0x9D1D60E5076F5B6FULL, 0x3810E399B6F65BA2ULL, 0x32095B6D4AB5F9B1ULL, 0x35CAB62109DD038AULL,
    0xA90B24499FCFAFB1ULL, 0x77A225A07CC2C6BDULL, 0x513E5E634C70E331ULL, 0x4361C0CA3F692F12ULL,
    0xD941ACA44B20A45BULL, 0x528F7C8602C5807BULL, 0x52AB92BEB9613989ULL, 0x9D1DFA2EFC557F73ULL,
    0x722FF175F572C348ULL, 0x1D1260A51107FE97ULL, 0x7A249A57EC0C9BA2ULL, 0x04208FE9E8F7F2D6ULL,
    0x5A110C6058B920A0ULL, 0x0CD9A497658A5698ULL, 0x56FD23C8F9715A4CULL, 0x284C847B9D887AAEULL,
    0x04FEABFBBDB619CBULL, 0x742E1E651C60BA83ULL, 0x9A9632E65904AD3CULL, 0x881B82A13B51B9E2ULL,
    0x506E6744CD974924ULL, 0xB0183DB56FFC6A79ULL, 0x0ED9B915C66ED37EULL, 0x5E11E86D5873D484ULL,
    0xF678647E3519AC6EULL, 0x1B85D488D0F20CC5ULL, 0xDAB9FE6525D89021ULL, 0x0D151D86ADB73615ULL,
    0xA865A54EDCC0F019ULL, 0x93C42566AEF98FFBULL, 0x99E7AFEABE000731ULL, 0x48CBFF086DDF285AULL,
    0x7F9B6AF1EBF78BAFULL, 0x58627E1A149BBA21ULL, 0x2CD16E2ABD791E33ULL, 0xD363EFF5F0977996ULL,
    0x0CE2A38C344A6EEDULL, 0x1A804AADB9CFA741ULL, 0x907F30421D78C5DEULL, 0x501F65EDB3034D07ULL,
    0x37624AE5A48FA6E9ULL, 0x957BAF61700CFF4EULL, 0x3A6C27934E31188AULL, 0xD49503536ABCA345ULL,
    0x088E049589C432E0ULL, 0xF943AEE7FEBF21B8ULL, 0x6C3B8E3E336139D3ULL, 0x364F6FFA464EE52EULL,
    0xD60F6DCEDC314222ULL, 0x56963B0DCA418FC0ULL, 0x16F50EDF91E513AFULL, 0xEF1955914B609F93ULL,
    0x565601C0364E3228ULL, 0xECB53939887E8175ULL, 0xBAC7A9A18531294BULL, 0xB344C470397BBA52ULL,
    0x65D34954DAF3CEBDULL, 0xB4B81B3FA97511E2ULL, 0xB422061193D6F6A7ULL, 0x071582401C38434DULL,
    0x7A13F18BBEDC4FF5ULL, 0xBC4097B116C524D2ULL, 0x59B97885E2F2EA28ULL, 0x99170A5DC3115544ULL,
    0x6F423357E7C6A9F9ULL, 0x325928EE6E6F8794ULL, 0xD0E4366228B03343ULL, 0x565C31F7DE89EA27ULL,
    0x30F5611484119414ULL, 0xD873DB391292ED4FULL, 0x7BD94E1D8E17DEBCULL, 0xC7D9F16864A76E94ULL,
    0x947AE053EE56E63CULL, 0xC8C93882F9475F5FULL, 0x3A9BF55BA91F81CAULL, 0xD9A11FBB3D9808E4ULL,
    0x0FD22063EDC29FCAULL, 0xB3F256D8ACA0B0B9ULL, 0xB03031A8B4516E84ULL, 0x35DD37D5871448AFULL,
    0xE9F6082B05542E4EULL, 0xEBFAFA33D7254B59ULL, 0x9255ABB50D532280ULL, 0xB9AB4CE57F2D34F3ULL,
    0x693501D628297551ULL, 0xC62C58F97DD949BFULL, 0xCD454F8F19C5126AULL, 0xBBE83F4ECC2BDECBULL,
    0xDC842B7E2819E230ULL, 0xBA89142E007503B8ULL, 0xA3BC941D0A5061CBULL, 0xE9F6760E32CD8021ULL,
    0x09C7E552BC76492FULL, 0x852F54934DA55CC9ULL, 0x8107FCCF064FCF56ULL, 0x098954D51FFF6580ULL,
    0x23B70EDB1955C4BFULL, 0xC330DE426430F69DULL, 0x4715ED43E8A45C0AULL, 0xA8D7E4DAB780A08DULL,
    0x0572B974F03CE0BBULL, 0xB57D2E985E1419C7ULL, 0xE8D9ECBE2CF3D73FULL, 0x2FE4B17170E59750ULL,
    0x11317BA87905E790ULL, 0x7FBF21EC8A1F45ECULL, 0x1725CABFCB045B00ULL, 0x964E915CD5E2B207ULL,
    0x3E2B8BCBF016D66DULL, 0xBE7444E39328A0ACULL, 0xF85B2B4FBCDE44B7ULL, 0x49353FEA39BA63B1ULL,
    0x1DD01AAFCD53486AULL, 0x1FCA8A92FD719F85ULL, 0xFC7C95D827357AFAULL, 0x18A6A990C8B35EBDULL,
    0xCCCB7005C6B9C28DULL, 0x3BDBB92C43B17F26ULL, 0xAA70B5B4F89695A2ULL, 0xE94C39A54A98307FULL,
    0xB7A0B174CFF6F36EULL, 0xD4DBA84729AF48ADULL, 0x2E18BC1AD9704A68ULL, 0x2DE0966DAF2F8B1CULL,
    0xB9C11D5B1E43A07EULL, 0x64972D68DEE33360ULL, 0x94628D38D0C20584ULL, 0xDBC0D2B6AB90A559ULL,
    0xD2733C4335C6A72FULL, 0x7E75D99D94A70F4DULL, 0x6CED1983376FA72BULL, 0x97FCAACBF030BC24ULL,
    0x7B77497B32503B12ULL, 0x8547EDDFB81CCB94ULL, 0x79999CDFF70902CBULL, 0xCFFE1939438E9B24ULL,
    0x829626E3892D95D7ULL, 0x92FAE24291F2B3F1ULL, 0x63E22C147B9C3403ULL, 0xC678B6D860284A1CULL,
    0x5873888850659AE7ULL, 0x0981DCD296A8736DULL, 0x9F65789A6509A440ULL, 0x9FF38FED72E9052FULL,
    0xE479EE5B9930578CULL, 0xE7F28ECD2D49EECDULL, 0x56C074A581EA17FEULL, 0x5544F7D774B14AEFULL,
    0x7B3F0195FC6F290FULL, 0x12153635B2C0CF57ULL, 0x7F5126DBBA5E0CA7ULL, 0x7A76956C3EAFB413ULL,
    0x3D5774A11D31AB39ULL, 0x8A1B083821F40CB4ULL, 0x7B4A38E32537DF62ULL, 0x950113646D1D6E03ULL,
    0x4DA8979A0041E8A9ULL, 0x3BC36E078F7515D7ULL, 0x5D0A12F27AD310D1ULL, 0x7F9D1A2E1EBE1327ULL,
    0xDA3A361B1C5157B1ULL, 0xDCDD7D20903D0C25ULL, 0x36833336D068F707ULL, 0xCE68341F79893389ULL,
    0xAB9090168DD05F34ULL, 0x43954B3252DC25E5ULL, 0xB438C2B67F98E5E9ULL, 0x10DCD78E3851A492ULL,
    0xDBC27AB5447822BFULL, 0x9B3CDB65F82CA382ULL, 0xB67B7896167B4C84ULL, 0xBFCED1B0048EAC50ULL,
    0xA9119B60369FFEBDULL, 0x1FFF7AC80904BF45ULL, 0xAC12FB171817EEE7ULL, 0xAF08DA9177DDA93DULL,
    0x1B0CAB936E65C744ULL, 0xB559EB1D04E5E932ULL, 0xC37B45B3F8D6F2BAULL, 0xC3A9DC228CAAC9E9ULL,
    0xF3B8B6675A6507FFULL, 0x9FC477DE4ED681DAULL, 0x67378D8ECCEF96CBULL, 0x6DD856D94D259236ULL,
    0xA319CE15B0B4DB31ULL, 0x073973751F12DD5EULL, 0x8A8E849EB32781A5ULL, 0xE1925C71285279F5ULL,
    0x74C04BF1790C0EFEULL, 0x4DDA48153C94938AULL, 0x9D266D6A1CC0542CULL, 0x7440FB816508C4FEULL,
    0x13328503DF48229FULL, 0xD6BF7BAEE43CAC40ULL, 0x4838D65F6EF6748FULL, 0x1E152328F3318DEAULL,
    0x8F8419A348F296BFULL, 0x72C8834A5957B511ULL, 0xD7A023A73260B45CULL, 0x94EBC8ABCFB56DAEULL,
    0x9FC10D0F989993E0ULL, 0xDE68A2355B93CAE6ULL, 0xA44CFE79AE538BBEULL, 0x9D1D84FCCE371425ULL,
    0x51D2B1AB2DDFB636ULL, 0x2FD7E4B9E72CD38CULL, 0x65CA5B96B7552210ULL, 0xDD69A0D8AB3B546DULL,
    0x604D51B25FBF70E2ULL, 0x73AA8A564FB7AC9EULL, 0x1A8C1E992B941148ULL, 0xAAC40A2703D9BEA0ULL,
    0x764DBEAE7FA4F3A6ULL, 0x1E99B96E70A9BE8BULL, 0x2C5E9DEB57EF4743ULL, 0x3A938FEE32D29981ULL,
    0x26E6DB8FFDF5ADFEULL, 0x469356C504EC9F9DULL, 0xC8763C5B08D1908CULL, 0x3F6C6AF859D80055ULL,
    0x7F7CC39420A3A545ULL, 0x9BFB227EBDF4C5CEULL, 0x89039D79D6FC5C5CULL, 0x8FE88B57305E2AB6ULL,
    0xA09E8C8C35AB96DEULL, 0xFA7E393983325753ULL, 0xD6B6D0ECC617C699ULL, 0xDFEA21EA9E7557E3ULL,
    0xB67C1FA481680AF8ULL, 0xCA1E3785A9E724E5ULL, 0x1CFC8BED0D681639ULL, 0xD18D8549D140CAEAULL,
    0x4ED0FE7E9DC91335ULL, 0xE4DBF0634473F5D2ULL, 0x1761F93A44D5AEFEULL, 0x53898E4C3910DA55ULL,
    0x734DE8181F6EC39AULL, 0x2680B122BAA28D97ULL, 0x298AF231C85BAFABULL, 0x7983EED3740847D5ULL,
    0x66C1A2A1A60CD889ULL, 0x9E17E49642A3E4C1ULL, 0xEDB454E7BADC0805ULL, 0x50B704CAB602C329ULL,
    0x4CC317FB9CDDD023ULL, 0x66B4835D9EAFEA22ULL, 0x219B97E26FFC81BDULL, 0x261E4E4C0A333A9DULL,
    0x1FE2CCA76517DB90ULL, 0xD7504DFA8816EDBBULL, 0xB9571FA04DC089C8ULL, 0x1DDC0325259B27DEULL,
    0xCF3F4688801EB9AAULL, 0xF4F5D05C10CAB243ULL, 0x38B6525C21A42B0EULL, 0x36F60E2BA4FA6800ULL,
    0xEB3593803173E0CEULL, 0x9C4CD6257C5A3603ULL, 0xAF0C317D32ADAA8AULL, 0x258E5A80C7204C4BULL,
    0x8B889D624D44885DULL, 0xF4D14597E660F855ULL, 0xD4347F66EC8941C3ULL, 0xE699ED85B0DFB40DULL,
    0x2472F6207C2D0484ULL, 0xC2A1E7B5B459AEB5ULL, 0xAB4F6451CC1D45ECULL, 0x63767572AE3D6174ULL,
    0xA59E0BD101731A28ULL, 0x116D0016CB948F09ULL, 0x2CF9C8CA052F6E9FULL, 0x0B090A7560A968E3ULL,
    0xABEEDDB2DDE06FF1ULL, 0x58EFC10B06A2068DULL, 0xC6E57A78FBD986E0ULL, 0x2EAB8CA63CE802D7ULL,
    0x14A195640116F336ULL, 0x7C0828DD624EC390ULL, 0xD74BBE77E6116AC7ULL, 0x804456AF10F5FB53ULL,
    0xEBE9EA2ADF4321C7ULL, 0x03219A39EE587A30ULL, 0x49787FEF17AF9924ULL, 0xA1E9300CD8520548ULL,
    0x5B45E522E4B1B4EFULL, 0xB49C3B3995091A36ULL, 0xD4490AD526F14431ULL, 0x12A8F216AF9418C2ULL,
    0x001F837CC7350524ULL, 0x1877B51E57A764D5ULL, 0xA2853B80F17F58EEULL, 0x993E1DE72D36D310ULL,
    0xB3598080CE64A656ULL, 0x252F59CF0D9F04BBULL, 0xD23C8E176D113600ULL, 0x1BDA0492E7E4586EULL,
    0x21E0BD5026C619BFULL, 0x3B097ADAF088F94EULL, 0x8D14DEDB30BE846EULL, 0xF95CFFA23AF5F6F4ULL,
    0x3871700761B3F743ULL, 0xCA672B91E9E4FA16ULL, 0x64C8E531BFF53B55ULL, 0x241260ED4AD1E87DULL,
    0x106C09B972D2E822ULL, 0x7FBA195410E5CA30ULL, 0x7884D9BC6CB569D8ULL, 0x0647DFEDCD894A29ULL,
    0x63573FF03E224774ULL, 0x4FC8E9560F91B123ULL, 0x1DB956E450275779ULL, 0xB8D91274B9E9D4FBULL,
    0xA2EBEE47E2FBFCE1ULL, 0xD9F1F30CCD97FB09ULL, 0xEFED53D75FD64E6BULL, 0x2E6D02C36017F67FULL,
    0xA9AA4D20DB084E9BULL, 0xB64BE8D8B25396C1ULL, 0x70CB6AF7C2D5BCF0ULL, 0x98F076A4F7A2322EULL,
    0xBF84470805E69B5FULL, 0x94C3251F06F90CF3ULL, 0x3E003E616A6591E9ULL, 0xB925A6CD0421AFF3ULL,
    0x61BDD1307C66E300ULL, 0xBF8D5108E27E0D48ULL, 0x240AB57A8B888B20ULL, 0xFC87614BAF287E07ULL,
    0xEF02CDD06FFDB432ULL, 0xA1082C0466DF6C0AULL, 0x8215E577001332C8ULL, 0xD39BB9C3A48DB6CFULL,
    0x2738259634305C14ULL, 0x61CF4F94C97DF93DULL, 0x1B6BACA2AE4E125BULL, 0x758F450C88572E0BULL,
    0x959F587D507A8359ULL, 0xB063E962E045F54DULL, 0x60E8ED72C0DFF5D1ULL, 0x7B64978555326F9FULL,
    0xFD080D236DA814BAULL, 0x8C90FD9B083F4558ULL, 0x106F72FE81E2C590ULL, 0x7976033A39F7D952ULL,
    0xA4EC0132764CA04BULL, 0x733EA705FAE4FA77ULL, 0xB4D8F77BC3E56167ULL, 0x9E21F4F903B33FD9ULL,
    0x9D765E419FB69F6DULL, 0xD30C088BA61EA5EFULL, 0x5D94337FBFAF7F5BULL, 0x1A4E4822EB4D7A59ULL,
    0x6FFE73E81B637FB3ULL, 0xDDF957BC36D8B9CAULL, 0x64D0E29EEA8838B3ULL, 0x08DD9BDFD96B9F63ULL,
    0x087E79E5A57D1D13ULL, 0xE328E230E3E2B3FBULL, 0x1C2559E30F0946BEULL, 0x720BF5F26F4D2EAAULL,
    0xB0774D261CC609DBULL, 0x443F64EC5A371195ULL, 0x4112CF68649A260EULL, 0xD813F2FAB7F5C5CAULL,
    0x660D3257380841EEULL, 0x59AC2C7873F910A3ULL, 0xE846963877671A17ULL, 0x93B633ABFA3469F8ULL,
    0xC0C0F5A60EF4CDCFULL, 0xCAF21ECD4377B28CULL, 0x57277707199B8175ULL, 0x506C11B9D90E8B1DULL,
    0xD83CC2687A19255FULL, 0x4A29C6465A314CD1ULL, 0xED2DF21216235097ULL, 0xB5635C95FF7296E2ULL,
    0x22AF003AB672E811ULL, 0x52E762596BF68235ULL, 0x9AEBA33AC6ECC6B0ULL, 0x944F6DE09134DFB6ULL,
    0x6C47BEC883A7DE39ULL, 0x6AD047C430A12104ULL, 0xA5B1CFDBA0AB4067ULL, 0x7C45D833AFF07862ULL,
    0x5092EF950A16DA0BULL, 0x9338E69C052B8E7BULL, 0x455A4B4CFE30E3F5ULL, 0x6B02E63195AD0CF8ULL,
    0x6B17B224BAD6BF27ULL, 0xD1E0CCD25BB9C169ULL, 0xDE0C89A556B9AE70ULL, 0x50065E535A213CF6ULL,
    0x9C1169FA2777B874ULL, 0x78EDEFD694AF1EEDULL, 0x6DC93D9526A50E68ULL, 0xEE97F453F06791EDULL,
    0x32AB0EDB696703D3ULL, 0x3A6853C7E70757A7ULL, 0x31865CED6120F37DULL, 0x67FEF95D92607890ULL,
    0x1F2B1D1F15F6DC9CULL, 0xB69E38A8965C6B65ULL, 0xAA9119FF184CCCF4ULL, 0xF43C732873F24C13ULL,
    0xFB4A3D794A9A80D2ULL, 0x3550C2321FD6109CULL, 0x371F77E76BB8417EULL, 0x6BFA9AAE5EC05779ULL,
    0xCD04F3FF001A4778ULL, 0xE3273522064480CAULL, 0x9F91508BFFCFC14AULL, 0x049A7F41061A9E60ULL,
    0xFCB6BE43A9F2FE9BULL, 0x08DE8A1C7797DA9BULL, 0x8F9887E6078735A1ULL, 0xB5B4071DBFC73A66ULL,
    0x230E343DFBA08D33ULL, 0x43ED7F5A0FAE657DULL, 0x3A88A0FBBCB05C63ULL, 0x21874B8B4D2DBC4FULL,
    0x1BDEA12E35F6A8C9ULL, 0x53C065C6C8E63528ULL, 0xE34A1D250E7A8D6BULL, 0xD6B04D3B7651DD7EULL,
    0x5E90277E7CB39E2DULL, 0x2C046F22062DC67DULL, 0xB10BB459132D0A26ULL, 0x3FA9DDFB67E2F199ULL,
    0x0E09B88E1914F7AFULL, 0x10E8B35AF3EEAB37ULL, 0x9EEDECA8E272B933ULL, 0xD4C718BC4AE8AE5FULL,
    0x81536D601170FC20ULL, 0x91B534F885818A06ULL, 0xEC8177F83F900978ULL, 0x190E714FADA5156EULL,
    0xB592BF39B0364963ULL, 0x89C350C893AE7DC1ULL, 0xAC042E70F8B383F2ULL, 0xB49B52E587A1EE60ULL,
    0xFB152FE3FF26DA89ULL, 0x3E666E6F69AE2C15ULL, 0x3B544EBE544C19F9ULL, 0xE805A1E290CF2456ULL,
    0x24B33C9D7ED25117ULL, 0xE74733427B72F0C1ULL, 0x0A804D18B7097475ULL, 0x57E3306D881EDB4FULL,
    0x4AE7D6A36EB5DBCBULL, 0x2D8D5432157064C8ULL, 0xD1E649DE1E7F268BULL, 0x8A328A1CEDFE552CULL,
    0x07A3AEC79624C7DAULL, 0x84547DDC3E203C94ULL, 0x990A98FD5071D263ULL, 0x1A4FF12616EEFC89ULL,
    0xF6F7FD1431714200ULL, 0x30C05B1BA332F41CULL, 0x8D2636B81555A786ULL, 0x46C9FEB55D120902ULL,
    0xCCEC0A73B49C9921ULL, 0x4E9D2827355FC492ULL, 0x19EBB029435DCB0FULL, 0x4659D2B743848A2CULL,
    0x963EF2C96B33BE31ULL, 0x74F85198B05A2E7DULL, 0x5A0F544DD2B1FB18ULL, 0x03727073C2E134B1ULL,
    0xC7F6AA2DE59AEA61ULL, 0x352787BAA0D7C22FULL, 0x9853EAB63B5E0B35ULL, 0xABBDCDD7ED5C0860ULL,
    0xCF05DAF5AC8D77B0ULL, 0x49CAD48CEBF4A71EULL, 0x7A4C10EC2158C4A6ULL, 0xD9E92AA246BF719EULL,
    0x13AE978D09FE5557ULL, 0x730499AF921549FFULL, 0x4E4B705B92903BA4ULL, 0xFF577222C14F0A3AULL,
    0x55B6344CF97AAFAEULL, 0xB862225B055B6960ULL, 0xCAC09AFBDDD2CDB4ULL, 0xDAF8E9829FE96B5FULL,
    0xB5FDFC5D3132C498ULL, 0x310CB380DB6F7503ULL, 0xE87FBB46217A360EULL, 0x2102AE466EBB1148ULL,
    0xF8549E1A3AA5E00DULL, 0x07A69AFDCC42261AULL, 0xC4C118BFE78FEAAEULL, 0xF9F4892ED96BD438ULL,
    0x1AF3DBE25D8F45DAULL, 0xF5B4B0B0D2DEEEB4ULL, 0x962ACEEFA82E1C84ULL, 0x046E3ECAAF453CE9ULL,
    0xF05D129681949A4CULL, 0x964781CE734B3C84ULL, 0x9C2ED44081CE5FBDULL, 0x522E23F3925E319EULL,
    0x177E00F9FC32F791ULL, 0x2BC60A63A6F3B3F2ULL, 0x222BBFAE61725606ULL, 0x486289DDCC3D6780ULL,
    0x7DC7785B8EFDFC80ULL, 0x8AF38731C02BA980ULL, 0x1FAB64EA29A2DDF7ULL, 0xE4D9429322CD065AULL,
    0x9DA058C67844F20CULL, 0x24C0E332B70019B0ULL, 0x233003B5A6CFE6ADULL, 0xD586BD01C5C217F6ULL,
    0x5E5637885F29BC2BULL, 0x7EBA726D8C94094BULL, 0x0A56A5F0BFE39272ULL, 0xD79476A84EE20D06ULL,
    0x9E4C1269BAA4BF37ULL, 0x17EFEE45B0DEE640ULL, 0x1D95B0A5FCF90BC6ULL, 0x93CBE0B699C2585DULL,
    0x65FA4F227A2B6D79ULL, 0xD5F9E858292504D5ULL, 0xC2B5A03F71471A6FULL, 0x59300222B4561E00ULL,
    0xCE2F8642CA0712DCULL, 0x7CA9723FBB2E8988ULL, 0x2785338347F2BA08ULL, 0xC61BB3A141E50E8CULL,
    0x150F361DAB9DEC26ULL, 0x9F6A419D382595F4ULL, 0x64A53DC924FE7AC9ULL, 0x142DE49FFF7A7C3DULL,
    0x0C335248857FA9E7ULL, 0x0A9C32D5EAE45305ULL, 0xE6C42178C4BBB92EULL, 0x71F1CE2490D20B07ULL,
    0xF1BCC3D275AFE51AULL, 0xE728E8C83C334074ULL, 0x96FBF83A12884624ULL, 0x81A1549FD6573DA5ULL,
    0x5FA7867CAF35E149ULL, 0x56986E2EF3ED091BULL, 0x917F1DD5F8886C61ULL, 0xD20D8C88C8FFE65FULL,
    0x31D71DCE64B2C310ULL, 0xF165B587DF898190ULL, 0xA57E6339DD2CF3A0ULL, 0x1EF6E6DBB1961EC9ULL,
    0x70CC73D90BC26E24ULL, 0xE21A6B35DF0C3AD7ULL, 0x003A93D8B2806962ULL, 0x1C99DED33CB890A1ULL,
    0xCF3145DE0ADD4289ULL, 0xD0E4427A5514FB72ULL, 0x77C621CC9FB3A483ULL, 0x67A34DAC4356550BULL,
    0xF8D626AAAF278509ULL,
};

static int   optOwnBook  = 1;
static int   tBookMaxPly = 24;
static int   optBookBest = 1;              /* 1 = Best (default), 0 = Weighted */
static char  bookPath[1024] = "<embedded>";
static const unsigned char* bookData = NULL;
static size_t bookN = 0;                   /* number of 16-byte entries */
static unsigned char* bookOwned = NULL;
static int   bookTried = 0;
static U64   bookRng = 0x1234567ULL;

static void bookFree(void) {
    free(bookOwned);
    bookOwned = NULL; bookData = NULL; bookN = 0;
}

static void bookLoad(void) {
    bookFree();
    bookTried = 1;
    if (isEmbeddedName(bookPath)) {
#ifndef K7_NO_EMBED
        bookData = k7EmbBook_data;
        bookN = (size_t)(k7EmbBook_end - k7EmbBook_data) / 16;
        printf("info string Book: <embedded> (" K7_BOOK_FILE ", %lu entries)\n", (unsigned long)bookN);
#else
        printf("info string Book: this build has no embedded book\n");
#endif
    } else {
        FILE* f = fopen(bookPath, "rb");
        long sz;
        if (!f) { printf("info string Book: cannot open %s\n", bookPath); return; }
        fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
        bookOwned = (unsigned char*)malloc(sz > 0 ? (size_t)sz : 1);
        if (!bookOwned || sz < 16 || fread(bookOwned, 1, (size_t)sz, f) != (size_t)sz) {
            fclose(f); bookFree();
            printf("info string Book: read error on %s\n", bookPath);
            return;
        }
        fclose(f);
        bookData = bookOwned;
        bookN = (size_t)sz / 16;
        printf("info string Book: %s (%lu entries)\n", bookPath, (unsigned long)bookN);
    }
}

static U64 rdBE(const unsigned char* p, int n) {
    U64 v = 0;
    int i;
    for (i = 0; i < n; i++) v = (v << 8) | p[i];
    return v;
}

/* Polyglot key of a position (independent of the engine's own Zobrist keys) */
static U64 polyKey(BOARD* b) {
    U64 k = 0;
    int sq;
    for (sq = 21; sq <= 98; sq++) {
        int p = b->board[sq], kind;
        if (!sq_valid(sq) || p == EMPTY) continue;
        kind = 2 * (BASE(p) - 1) + (IS_BLACK(p) ? 0 : 1);
        k ^= polyRandom[64 * kind + 8 * sq_rank(sq) + sq_file(sq)];
    }
    if (b->castlePermission & WKCA) k ^= polyRandom[768];
    if (b->castlePermission & WQCA) k ^= polyRandom[769];
    if (b->castlePermission & BKCA) k ^= polyRandom[770];
    if (b->castlePermission & BQCA) k ^= polyRandom[771];
    if (b->enPas >= 0) {                   /* only if a pawn can really take en passant */
        int ep = b->enPas, can;
        if (b->stm == WHITE) can = b->board[ep - 9] == wP || b->board[ep - 11] == wP;
        else                 can = b->board[ep + 9] == bP || b->board[ep + 11] == bP;
        if (can) k ^= polyRandom[772 + sq_file(ep)];
    }
    if (b->stm == WHITE) k ^= polyRandom[780];
    return k;
}

/* A legal book move for this position, or 0 */
static MOVE bookProbe(BOARD* b) {
    MOVE legal[MAX_MOVES], cand[64];
    unsigned wts[64];
    size_t lo = 0, hi, i;
    int nl, nc = 0, j;
    U64 key, total = 0;

    if (!bookData || !bookN) return 0;
    key = polyKey(b);
    hi = bookN;
    while (lo < hi) {                      /* first entry with this key */
        size_t mid = (lo + hi) / 2;
        if (rdBE(bookData + mid * 16, 8) < key) lo = mid + 1; else hi = mid;
    }
    nl = genLegal(b, legal);
    for (i = lo; i < bookN && nc < 64; i++) {
        const unsigned char* e = bookData + i * 16;
        unsigned m16, w;
        int tf, tr, ff, fr, pr, from, to;
        if (rdBE(e, 8) != key) break;
        m16 = (unsigned)rdBE(e + 8, 2);
        w   = (unsigned)rdBE(e + 10, 2);
        tf = m16 & 7; tr = (m16 >> 3) & 7; ff = (m16 >> 6) & 7; fr = (m16 >> 9) & 7; pr = (m16 >> 12) & 7;
        from = 21 + fr * 10 + ff;
        to   = 21 + tr * 10 + tf;
        if (BASE(b->board[from]) == KING) {        /* Polyglot castling = king takes own rook */
            if (from == E1 && to == H1) to = G1;
            else if (from == E1 && to == A1) to = C1;
            else if (from == E8 && to == H8) to = G8;
            else if (from == E8 && to == A8) to = C8;
        }
        for (j = 0; j < nl; j++) {
            MOVE m = legal[j];
            int t = MV_TYPE(m);
            if ((int)MV_FROM(m) != from || (int)MV_TO(m) != to) continue;
            if (IS_PROMO(t) ? ((t & 3) + 1 != pr) : (pr != 0)) continue;
            cand[nc] = m; wts[nc] = w; total += w; nc++;
            break;
        }
    }
    if (!nc) return 0;
    if (optBookBest || total == 0) {
        int bi = 0;
        for (j = 1; j < nc; j++) if (wts[j] > wts[bi]) bi = j;
        if (total == 0) {                          /* no weights: pick any */
            bookRng ^= bookRng << 13; bookRng ^= bookRng >> 7; bookRng ^= bookRng << 17;
            bi = (int)(bookRng % (U64)nc);
        }
        return cand[bi];
    }
    bookRng ^= bookRng << 13; bookRng ^= bookRng >> 7; bookRng ^= bookRng << 17;
    {
        U64 r = bookRng % total, acc = 0;
        for (j = 0; j < nc; j++) { acc += wts[j]; if (r < acc) return cand[j]; }
    }
    return cand[nc - 1];
}

static void handleSetOption(char* line) {
    char name[256] = {0}, value[1024] = {0};
    char* n = strstr(line, "name ");
    char* v = strstr(line, " value ");
    if (!n) return;
    n += 5;
    if (v) {
        size_t len = (size_t)(v - n);
        if (len >= sizeof(name)) len = sizeof(name) - 1;
        memcpy(name, n, len);
        strncpy(value, v + 7, sizeof(value) - 1);
    } else {
        strncpy(name, n, sizeof(name) - 1);
    }
    trim(name);
    trim(value);

    if (strieq(name, "Hash")) {
        int mb = atoi(value);
        if (mb > 0) ttResize(mb);
    }
    else if (strieq(name, "Threads")) {
        int t = atoi(value);
        if (t < 1) t = 1;
        if (t > MAX_THREADS) t = MAX_THREADS;
        g_numThreads = t;
    }
    else if (strieq(name, "RFP"))       optRFP      = strieq(value, "true");
    else if (strieq(name, "NullMove+")) optNullPlus = strieq(value, "true");
    else if (strieq(name, "Futility"))  optFutility = strieq(value, "true");
    else if (strieq(name, "LMP"))       optLMP      = strieq(value, "true");
    else if (strieq(name, "LMR+"))      optLMRPlus  = strieq(value, "true");
    else if (strieq(name, "SEE"))       optSEE      = strieq(value, "true");
    else if (strieq(name, "Delta"))     optDelta    = strieq(value, "true");
    else if (strieq(name, "IIR"))       optIIR      = strieq(value, "true");
    else if (strieq(name, "Ordering+")) optOrdering = strieq(value, "true");
    else if (strieq(name, "QSearchTT")) optQsTT     = strieq(value, "true");
    else if (strieq(name, "TTEval"))    optTTEval   = strieq(value, "true");
    else if (strieq(name, "Singular"))  optSingular = strieq(value, "true");
    else if (strieq(name, "ContHist"))  optCont     = strieq(value, "true");
    else if (strieq(name, "CorrHist"))  optCorr     = strieq(value, "true");
    else if (strieq(name, "CapHist"))   optCapHist  = strieq(value, "true");
    else if (strieq(name, "TimeManager")) optTimeMgr = strieq(value, "true");
    else if (strieq(name, "Move Overhead")) {
        int v = atoi(value);
        tMoveOverhead = v < 0 ? 0 : (v > 5000 ? 5000 : v);
    }
    else if (strieq(name, "Fifty"))     optFifty    = strieq(value, "true");
    else if (strieq(name, "OwnBook"))   optOwnBook  = strieq(value, "true");
    else if (strieq(name, "BookPick"))  optBookBest = strieq(value, "Best");
    else if (strieq(name, "BookMaxPly")) {
        int v = atoi(value);
        tBookMaxPly = v < 0 ? 0 : (v > 200 ? 200 : v);
    }
    else if (strieq(name, "BookFile")) {
        strncpy(bookPath, value[0] ? value : "<embedded>", sizeof(bookPath) - 1);
        bookPath[sizeof(bookPath) - 1] = 0;
        bookLoad();
    }
    else if (strieq(name, "NetFile")) {
        if (value[0] && strcmp(value, "<empty>") != 0) {
            strncpy(nnPath, value, sizeof(nnPath) - 1);
            nnPath[sizeof(nnPath) - 1] = 0;
            nnFree();               /* drop the old net so the new one really loads */
            netTried = 0;
            ttClear();              /* old scores came from another net */
            evalCacheClear();
            ensureNetReady();
        }
    }
    else {
        int k;
        for (k = 0; k < NTUNES; k++)
            if (strieq(name, tunes[k].name)) {
                int v = atoi(value);
                if (v < tunes[k].lo) v = tunes[k].lo;
                if (v > tunes[k].hi) v = tunes[k].hi;
                *tunes[k].v = v;
                if (tunes[k].v == &tLmrBase || tunes[k].v == &tLmrDiv) initLMR();
            }
    }
}

static void printEval(BOARD* b) {
    int sgn = (b->stm == WHITE) ? 1 : -1;
    ensureNetReady();
    if (nnLoaded && threadsPrepare(1) == 1) {
        SEARCH_INFO proto;
        memset(&proto, 0, sizeof(proto));
        threadSetup(&g_th[0], 0, b, &proto);
        printf("info string eval (White's view): %d cp   net %s (768 -> %dx2 -> 1)\n",
               sgn * evalNet(g_th[0].board), nnPath, nnH);
    } else
        printf("info string eval (White's view): %d cp   material only, no net loaded\n",
               sgn * evalMaterial(b));
}

/* Fixed-depth search over set positions, always 1 thread: same build + same
   net = same node count, so any change in nodes means play changed. */
static void runBench(BOARD* b, int d) {
    static const char* benchFens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 1 9",
        "2rq1rk1/pp1bppbp/3p1np1/4n3/3NP2P/1BN1BP2/PPPQ2P1/2KR3R w - - 1 13",
        "r2q1rk1/1b2bppp/p2ppn2/1p6/3BPP2/2NB4/PPPQ2PP/2KR3R w - - 0 13",
        "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 3 6",
        "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/5N1P/PPBN1PP1/R1BQR1K1 w - - 0 13",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "6k1/5ppp/8/8/8/2B5/5PPP/3R2K1 w - - 0 1",
        "8/8/3k4/3p4/3P4/2K5/8/8 w - - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
        "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    };
    int nf = (int)(sizeof(benchFens) / sizeof(benchFens[0])), i;
    long total = 0;
    double ms = 0.0;
    if (d < 1) d = 1;
    if (d > MAX_PLY - 1) d = MAX_PLY - 1;
    ensureNetReady();
    for (i = 0; i < nf; i++) {
        SEARCH_INFO proto;
        MOVE bm;
        char mv[6] = "0000";
        double t0;
        parseFEN(b, benchFens[i]);
        ttClear();
        evalCacheClear();
        memset(&proto, 0, sizeof(proto));
        proto.depth = d;
        proto.silent = 1;
        t0 = nowMs();
        bm = runSearch(b, &proto, 1);
        ms += nowMs() - t0;
        if (bm) moveToStr(bm, mv);
        printf("bench %2d  %-6s nodes %ld\n", i + 1, mv, proto.nodes);
        total += proto.nodes;
    }
    printf("bench depth %d: %ld nodes  %.0f ms  %ld nps\n", d, total, ms,
           (long)(total / (ms / 1000.0 + 0.001)));
    fflush(stdout);
    parseFEN(b, START_FEN);
    ttClear();
    evalCacheClear();
}

static void uciLoop(BOARD* b) {
    char line[8192];

    setbuf(stdin, NULL);
    setbuf(stdout, NULL);

    for (;;) {
        char token[64] = {0};
        if (g_pending[0]) {                        /* command queued during a search */
            strcpy(line, g_pending);
            g_pending[0] = 0;
        } else if (!fgets(line, sizeof(line), stdin)) {
            break;
        }
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        sscanf(line, "%63s", token);

        if (!strcmp(token, "uci")) {
            printf("id name ChesslabNNet9\nid author K7\n");
            printf("option name Hash type spin default %d min 1 max %d\n",
                   TT_DEFAULT_MB, TT_MAX_MB);
            printf("option name Threads type spin default 1 min 1 max %d\n", MAX_THREADS);
            printf("option name NetFile type string default %s\n", nnPath);
            printf("option name OwnBook type check default %s\n", optOwnBook ? "true" : "false");
            printf("option name BookFile type string default %s\n", bookPath);
            printf("option name BookMaxPly type spin default %d min 0 max 200\n", tBookMaxPly);
            printf("option name BookPick type combo default %s var Best var Weighted\n", optBookBest ? "Best" : "Weighted");
            printf("option name RFP type check default %s\n",       optRFP      ? "true" : "false");
            printf("option name NullMove+ type check default %s\n", optNullPlus ? "true" : "false");
            printf("option name Futility type check default %s\n",  optFutility ? "true" : "false");
            printf("option name LMP type check default %s\n",       optLMP      ? "true" : "false");
            printf("option name LMR+ type check default %s\n",      optLMRPlus  ? "true" : "false");
            printf("option name SEE type check default %s\n",       optSEE      ? "true" : "false");
            printf("option name Delta type check default %s\n",     optDelta    ? "true" : "false");
            printf("option name IIR type check default %s\n",       optIIR      ? "true" : "false");
            printf("option name Ordering+ type check default %s\n", optOrdering ? "true" : "false");
            printf("option name QSearchTT type check default %s\n", optQsTT     ? "true" : "false");
            printf("option name TTEval type check default %s\n",    optTTEval   ? "true" : "false");
            printf("option name Singular type check default %s\n",  optSingular ? "true" : "false");
            printf("option name ContHist type check default %s\n", optCont     ? "true" : "false");
            printf("option name CorrHist type check default %s\n", optCorr     ? "true" : "false");
            printf("option name CapHist type check default %s\n",  optCapHist  ? "true" : "false");
            printf("option name TimeManager type check default %s\n", optTimeMgr ? "true" : "false");
            printf("option name Move Overhead type spin default %d min 0 max 5000\n", tMoveOverhead);
            {
                int k;
                for (k = 0; k < NTUNES; k++)
                    printf("option name %s type spin default %d min %d max %d\n",
                           tunes[k].name, *tunes[k].v, tunes[k].lo, tunes[k].hi);
            }
            printf("option name Fifty type check default %s\n",     optFifty    ? "true" : "false");
            printf("uciok\n");
        }
        else if (!strcmp(token, "isready")) {
            ensureNetReady();
            if (optOwnBook && !bookTried) bookLoad();
            printf("readyok\n");
        }
        else if (!strcmp(token, "setoption")) {
            handleSetOption(line);
        }
        else if (!strcmp(token, "ucinewgame")) {
            int k;
            parseFEN(b, START_FEN);
            ttClear();
            evalCacheClear();
            for (k = 0; k < MAX_THREADS; k++)
                if (g_th[k].corr) memset(g_th[k].corr, 0, sizeof(int) * 2 * CORR_SIZE);
        }
        else if (!strcmp(token, "eval")) {
            printEval(b);
        }
        else if (!strcmp(token, "bench")) {
            int d = 8;
            sscanf(line + 5, "%d", &d);
            runBench(b, d);
        }
        else if (!strcmp(token, "perft")) {
            int d = 1, i;
            sscanf(line + 5, "%d", &d);
            for (i = 1; i <= d; i++) {
                double t0 = nowMs();
                long nds = perft(b, i);
                printf("perft %d : %ld nodes  (%.0f ms)\n", i, nds, nowMs() - t0);
                fflush(stdout);
            }
        }
        else if (!strcmp(token, "position")) {
            char* ptr = strstr(line, "startpos");
            char* fenp = strstr(line, "fen");
            char* mv;

            if (ptr) {
                parseFEN(b, START_FEN);
            } else if (fenp) {
                parseFEN(b, fenp + 4);
            } else {
                parseFEN(b, START_FEN);
            }

            mv = strstr(line, "moves");
            if (mv) {
                char ms[16];
                mv += 5;
                while (sscanf(mv, "%15s", ms) == 1) {
                    MOVE m = strToMove(b, ms);
                    if (!m || b->hisPly >= HIST_SIZE - MAX_PLY - 2) break;
                    makeMove(b, m);
                    mv = strstr(mv, ms) + strlen(ms);
                }
            }
        }
        else if (!strcmp(token, "go")) {
            SEARCH_INFO proto;
            char* p;
            char best[6];
            MOVE bm;
            int wtime = 0, btime = 0, winc = 0, binc = 0, movestogo = 30, infinite = 0;

            ensureNetReady();
            memset(&proto, 0, sizeof(proto));
            proto.depth = MAX_PLY - 1;
            proto.timeLimit = 0;

            if ((p = strstr(line, "depth")))    sscanf(p + 5, "%d", &proto.depth);
            if ((p = strstr(line, "movetime"))) sscanf(p + 8, "%d", &proto.timeLimit);
            if ((p = strstr(line, "wtime")))    sscanf(p + 5, "%d", &wtime);
            if ((p = strstr(line, "btime")))    sscanf(p + 5, "%d", &btime);
            if ((p = strstr(line, "winc")))     sscanf(p + 4, "%d", &winc);
            if ((p = strstr(line, "binc")))     sscanf(p + 4, "%d", &binc);
            if ((p = strstr(line, "movestogo")))sscanf(p + 9, "%d", &movestogo);
            if ((p = strstr(line, "nodes")))    sscanf(p + 5, "%ld", &proto.nodeLimit);
            if (strstr(line, "infinite"))       infinite = 1;

            if (proto.timeLimit > 0) {                  /* movetime: as V3 */
                proto.softLimit = proto.timeLimit / 2;
            } else if (wtime || btime) {
                int mytime = (b->stm == WHITE) ? wtime : btime;
                int myinc  = (b->stm == WHITE) ? winc  : binc;
                int left = mytime - tMoveOverhead;
                if (left < 10) left = 10;
                if (movestogo < 1) movestogo = 30;
                if (optTimeMgr) {
                    /* soft: normal budget; hard: up to 4x when the position needs it */
                    int mtg = movestogo < 25 ? movestogo : 25;
                    int base = left / mtg + myinc * 3 / 4;
                    int hard = base * 4;
                    if (base > left / 2) base = left / 2;
                    if (hard > left / 3 + myinc) hard = left / 3 + myinc;
                    if (hard > left - 10) hard = left - 10;
                    if (hard < base) hard = base;
                    proto.softLimit = base > 5 ? base : 5;
                    proto.timeLimit = hard > 10 ? hard : 10;
                } else {
                    proto.timeLimit = mytime / movestogo + myinc - tMoveOverhead;
                    if (proto.timeLimit < 20) proto.timeLimit = 20;
                    if (proto.timeLimit > mytime / 2) proto.timeLimit = mytime / 2;
                    proto.softLimit = proto.timeLimit / 2;
                }
            }
            if (proto.depth < 1 || proto.depth >= MAX_PLY) proto.depth = MAX_PLY - 1;

            bm = 0;
            /* opening book first (never in "go infinite" = analysis) */
            if (optOwnBook && !infinite && b->hisPly < tBookMaxPly) {
                if (!bookTried) bookLoad();
                bm = bookProbe(b);
                if (bm) {
                    char bs[6];
                    moveToStr(bm, bs);
                    printf("info string book move %s\n", bs);
                    printf("info depth 1 score cp 0 nodes 0 time 0 pv %s\n", bs);
                    proto.stopped = 1;
                }
            }
            if (!bm) {
                g_age++;
                bm = runSearch(b, &proto, g_numThreads);
            }

            /* UCI: in "go infinite" the move is sent only after "stop", even
               if the search already reached its maximum depth. */
            while (infinite && !proto.stopped && !g_quit && !g_pending[0]) {
                if (handleSearchCommand()) break;
            }

            if (bm) {
                moveToStr(bm, best);
                printf("bestmove %s\n", best);
            } else {
                printf("bestmove 0000\n");
            }
            fflush(stdout);
            if (g_quit) break;
        }
        else if (!strcmp(token, "stop")) {
            /* stop with no search running: nothing to do */
        }
        else if (!strcmp(token, "quit")) {
            break;
        }
    }
}

/* ChesslabNNet9.exe                      -> UCI mode
   ChesslabNNet9.exe bench [depth] [net]  -> run bench and exit (for PGO) */
int main(int argc, char** argv) {
    nowMs();
    initZobrist();
    initSq64();
    bookRng ^= (U64)time(NULL) * 0x9E3779B97F4A7C15ULL;
    if (!bookRng) bookRng = 1;
    initBoardTables();
    initLMR();

    gboard = (BOARD*)calloc(1, sizeof(BOARD));
    evalCache = (U64*)calloc(EVAL_CACHE_SIZE, sizeof(U64));
    ttResize(TT_DEFAULT_MB);
    if (!gboard || !tt || !evalCache) return 1;

    parseFEN(gboard, START_FEN);

    if (argc >= 2 && !strcmp(argv[1], "bench")) {
        int d = (argc >= 3) ? atoi(argv[2]) : 8;
        if (argc >= 4) {
            strncpy(nnPath, argv[3], sizeof(nnPath) - 1);
            nnPath[sizeof(nnPath) - 1] = 0;
        }
        runBench(gboard, d);
        return 0;
    }

    uciLoop(gboard);
    return 0;
}
