# Introduction

A chess game is one of the best-documented competitive processes there
is: finite, governed by a complete rule set, and recorded move by move.
It is usually analysed by evaluating positions with an engine, which
says who stands better but not what kind of dynamical system a game is.
Statistical studies have started to ask the second question.
Opening-move frequencies follow a power law whose exponent grows with
game depth, and the pooled distribution follows Zipf's law
[@blasius2009]; move sequences show long-range memory that depends on
player skill [@schaigorodsky2013]; and engine-based measures of how much
each move shifts the expected outcome show skill-dependent decision
complexity [@chacoma2024]. Regan and Haworth [@regan2011] rate players
by comparing each move with an engine's evaluation of all legal moves,
fitting a sensitivity and a consistency parameter. That construction is
close in spirit to the move ensemble used here.

Statistical physics supplies a different and complementary vocabulary.
Two species $A$ and $B$ that vanish on meeting, $A+B\to 0$, are a
classic problem in reaction kinetics, in which density fluctuations
change the decay of the reagents [@ovchinnikov1978; @toussaint1983]. The
same structure appears in finance: the simplest model of Bak, Paczuski
and Shubik maps onto diffusing and annihilating particles [@bak1997].
Information theory contributes the entropy of a choice among
alternatives [@shannon1948], its role in statistical inference
[@jaynes1957], and, in Shannon's own proposal for a chess-playing
program, the idea of evaluating moves by a numerical score
[@shannon1950].

Chess already has the three ingredients of an annihilation lattice gas.
Squares are exclusive sites; two opposing species of pieces occupy them;
and a capture removes a piece from the board. Mover decisions then
determine where and when the reactions happen. What is missing, is a
framework that puts exclusion, mutual annihilation, Boolean evolution
and decision entropy into one set of observables computed directly from
a game record. This paper has two aims. The first is to define the model
and implement it as a reusable tool. The second is to test, on five
classic games and against explicit null ensembles, whether the
observables carry information beyond what the rules and the game length
already imply. The games serve as a demonstration. The longer-term
interest is that the three ingredients are shared by many systems
outside chess (Section [4](#sec:discussion){reference-type="ref"
reference="sec:discussion"}).

We fixed five hypotheses before running the comparisons:

- Annihilations cluster in a window in which the number of available
  captures peaks, and the timing of this window differs between tactical
  and positional games.

- Decision entropy falls over a game, and surprisal exceeds the model
  entropy at sacrificial moves.

- The Boolean control field changes more bits per move than the piece
  field, and more so in sharp games.

- Real games differ from random-play games and from model-sampled games
  in entropy, immune fraction and related observables.

- Survival time by piece type increases with hierarchy rank.

Section [2](#sec:methods){reference-type="ref" reference="sec:methods"}
defines the model, data and statistics,
Section [3](#sec:results){reference-type="ref" reference="sec:results"}
reports the results, Section [4](#sec:discussion){reference-type="ref"
reference="sec:discussion"} discusses them and
Section [5](#sec:conclusion){reference-type="ref"
reference="sec:conclusion"} concludes.

# Methods {#sec:methods}

## Lattice gas, species and annihilation

The $8\times8$ board is a lattice of 64 sites with hard-core exclusion:
at most one piece occupies a site, and the code checks this after every
ply. White and Black are the two species. Each piece type has a
hierarchy rank $h$ (pawn 1, knight 2, bishop 3, rook 4, queen 5, king 6)
and an existence weight $w$, $$\begin{equation}
w_{\mathrm{P}}=1,\; w_{\mathrm{N}}=3,\; w_{\mathrm{B}}=3.25,\; w_{\mathrm{R}}=5,\; w_{\mathrm{Q}}=9,\; w_{\mathrm{K}}=0 .
\end{equation}$$ The king has zero weight and is never annihilated; it
acts as the absorbing boundary of the game. Each species starts with
total weight $W_0=39.5$. The hierarchy rank is used only to order the
survival analysis (H5).

A capture is treated as a reaction $W+B\to$ one particle. The victim's
weight is released as reaction energy, and the capture cell is marked as
a decision-immune site that stays marked for the rest of the game. This
is a bookkeeping convention: in chess the capturing piece survives, so
the reaction is asymmetric. We keep the convention because it gives a
clean conserved quantity, and we return to it in the Discussion. The
condensate (immune) fraction is $$\begin{equation}
\varphi(t)=\frac{\text{weight released up to ply } t}{2W_0},
\end{equation}$$ and the surviving weight fraction of a species $\sigma$
is $$\begin{equation}
S_\sigma(t)=\frac{\sum_{i\in\sigma,\ \text{alive at } t} w_i}{W_0}.
\end{equation}$$ Particle density is the number of pieces divided by 64.

## Boolean evolution

The position is a Boolean tensor $X_t\in\{0,1\}^{12\times 64}$ (six
piece types for each species, 64 sites). A move changes it by an XOR
with its difference pattern $D(m_t)$, $$\begin{equation}
X_{t+1}=X_t\oplus D(m_t),\qquad A_t=\lVert D(m_t)\rVert_{\mathrm{H}},
\end{equation}$$ where $\lVert\cdot\rVert_{\mathrm H}$ is the Hamming
weight (2 for a quiet move, 3 for a capture, 4 for castling). A second
Boolean field $C_t\in\{0,1\}^{2\times64}$ records which sites each side
attacks and is updated the same way. The ratio $R_{\mathrm{c/p}}$ of the
mean control-field flips per ply to the mean piece-field flips per ply
measures how non-locally a single move changes the system.

## Decision entropy and immune information

At ply $t$ the mover has $N_t$ legal moves, each with a heuristic energy
$s_m$ (in pawn-like units): the weight of a captured piece, a small
bonus for central squares and for giving check, and a penalty
proportional to the moved piece's weight if it lands on a square
attacked by the opponent (larger if the square is undefended). Moves are
weighted in a Boltzmann ensemble at inverse temperature $\beta$,
$$\begin{equation}
p_m=\frac{e^{\beta s_m}}{\sum_{m'}e^{\beta s_{m'}}},\qquad
H_t=-\sum_m p_m\log_2 p_m\le\log_2 N_t,\qquad
I_t=-\log_2 p_{m_t},
\end{equation}$$ where $H_t$ is the decision entropy and $I_t$ the
surprisal of the move actually played. Captures and pawn pushes cannot
be undone, so we treat them as ratchets and accumulate their surprisal
as immune information, $$\begin{equation}
I_{\mathrm{imm}}(t)=\sum_{\tau\le t,\ \text{irreversible}}I_\tau .
\end{equation}$$ We report the mean entropy $\langle H\rangle$, the
entropy drop
$\Delta H=\langle H\rangle_{\text{first quarter}}-\langle H\rangle_{\text{last quarter}}$,
the gap $\langle I\rangle-\langle H\rangle$ (zero on average for moves
sampled from the ensemble itself), and the share
$I_{\mathrm{imm}}/\sum I$. Other per-ply observables are the number of
available captures $C_t$ (reaction channels), the mixing index (fraction
of neighbouring occupied pairs that are unlike) and the number of
contested sites. All results use $\beta=1$ unless stated otherwise.

## Games and data verification

We analysed five games: Morphy against the Duke of Brunswick and Count
Isouard (Paris, 1858); Byrne--Fischer (New York, 1956); Karpov--Kasparov
(World Championship, game 16, 1985); Kasparov--Topalov (Wijk aan Zee,
1999); and Aronian--Anand (Wijk aan Zee, 2013). They contrast
sacrificial and positional play and span 155 years. For each game, the
move list was transcribed from two independent web sources
(Section [\[sec:data\]](#sec:data){reference-type="ref"
reference="sec:data"}). The two transcriptions were compared as move
sequences, replayed move by move for legality, and checked against the
recorded result. All five pairs were identical (33, 82, 80, 87 and 46
plies) and replayed without error; Morphy and Byrne--Fischer end in
checkmate and the other three end in resignation.

## Null ensembles and statistics

Real games were compared with two ensembles of games of the same length
as each real game, started from the initial position and analysed with
the same code. *Null A* consists of 1000 games of uniformly random legal
moves. *Null B* consists of 1000 games in which each move is drawn from
the model ensemble ($\beta=1$). Games that ended before the target
length were discarded and replaced. For each observable $x$ and game we
report the $z$-score
$(x_{\mathrm{real}}-\bar x_{\mathrm{null}})/s_{\mathrm{null}}$. With 6
observables, 5 games and 2 nulls (60 comparisons), we treat $|z|\ge 3.3$
(roughly $p<10^{-3}$ for a normal distribution) as a conservative marker
and regard smaller values as descriptive only. Survival order across
piece types was assessed by a Spearman rank correlation of the mean
final surviving fraction against hierarchy rank.

## Software

The model is implemented in Python (`chess_physics.py`) using the
python-chess library [@fiekas2025], NumPy and Matplotlib. It reads PGN
files and writes a text report, a per-ply table and a diagnostic plot
per game. The null-ensemble and statistics scripts used here are
provided with the code
(Section [\[sec:data\]](#sec:data){reference-type="ref"
reference="sec:data"}).

# Results {#sec:results}

## Observables for the five games

Table [\[tab:games\]](#tab:games){reference-type="ref"
reference="tab:games"} and Fig. [1](#fig:games){reference-type="ref"
reference="fig:games"} summarise the baseline run. Density fell
monotonically in every game, as it must, from 0.500 to between 0.172
(Kasparov--Topalov) and 0.344 (Aronian--Anand). There were 10 to 21
annihilations, and the immune fraction $\varphi$ ranged from 0.256 to
0.734; in Kasparov--Topalov, 21 captures released 58.0 of the 79.0 units
of total weight. The first annihilation occurred between ply 6 and ply
16 (plies 7, 12, 6, 16 and 12 in the order of
Table [\[tab:games\]](#tab:games){reference-type="ref"
reference="tab:games"}). The longest chain of captures on one cell was 3
(Morphy and Karpov--Kasparov) and 2 in the other games.

Mean decision entropy was 3.67--4.67 bits against a mean $\log_2 N_t$ of
4.45--5.19 bits. Entropy fell from the first to the last quarter in all
five games, by 0.40 to 2.33 bits (Morphy: 4.70 to 2.37 bits). The piece
field changed 2.23--2.42 bits per ply on average and the control field
3.9--6.8, so $R_{\mathrm{c/p}}$ was 1.70--2.81. The number of available
captures peaked at 5 to 7, at between 0.23 and 0.72 of the game length.

<figure id="fig:games" data-latex-placement="t">
<embed src="fig_games.pdf" />
<figcaption>Immune fraction <span
class="math inline"><em>φ</em>(<em>t</em>)</span> (top), decision
entropy <span
class="math inline"><em>H</em><sub><em>t</em></sub></span>, its upper
bound <span
class="math inline">log<sub>2</sub><em>N</em><sub><em>t</em></sub></span>
and the surprisal <span
class="math inline"><em>I</em><sub><em>t</em></sub></span> of the played
move (middle), and Boolean activity of the control and piece fields
(bottom) for the five games.</figcaption>
</figure>

## Survival of the two species

Figure [2](#fig:survival){reference-type="ref" reference="fig:survival"}
shows the surviving weight fraction of each species. In the three long
games (Byrne--Fischer, Karpov--Kasparov, Kasparov--Topalov) both species
fell below half of their initial weight within 1 to 5 plies of each
other (plies 54 and 59, 78 and 79, 74 and 75), so annihilation was
nearly symmetric in time. In Morphy's game only White crossed the
half-weight line (ply 32), which reflects a one-sided sacrificial
finish; Aronian--Anand ended before either species did. The species with
more surviving weight at the end won in only two of five games
(Byrne--Fischer, Black, 0.49 against 0.35; Kasparov--Topalov, White,
0.30 against 0.23). Final weights were 0.39/0.64 (White/Black) for
Morphy, 0.46/0.43 for Karpov--Kasparov and 0.77/0.72 for Aronian--Anand.

Averaged over the five games, the final surviving fraction was 0.54 for
pawns, 0.35 for knights, 0.40 for bishops, 0.45 for rooks and 0.60 for
queens. The rank correlation with hierarchy was $\rho=0.40$ ($n=5$ piece
types, $p\approx0.5$), so no monotonic ordering can be claimed.

<figure id="fig:survival" data-latex-placement="t">
<embed src="fig_survival.pdf" />
<figcaption>Surviving weight fraction <span
class="math inline"><em>S</em><sub><em>σ</em></sub>(<em>t</em>)</span>
of White (dark) and Black (blue) in the five games.</figcaption>
</figure>

## Surprisal at sacrificial moves

For the four games with a well-known sacrifice we looked at the move
itself (Table [1](#tab:sac){reference-type="ref" reference="tab:sac"}).
In Morphy's game, the queen sacrifice 16.Qb8+ had $H_t=1.49$ bits but
$I_t=13.68$ bits, the most surprising ply of the game. Fischer's
17...Be6 ($I_t=8.04$, $H_t=1.86$) and Kasparov's 24.Rxd4 ($I_t=6.26$,
$H_t=2.29$) ranked at the 98.8th and 94.3rd percentile of $I_t-H_t$
within their games. Anand's 16...Nde5, a knight move in a position with
many reasonable options ($H_t=5.21$), reached only the 73.9th percentile
($I_t=5.74$).

  Game                Move          Ply   $H_t$   $I_t$   Percentile
  ------------------- ----------- ----- ------- ------- ------------
  Morphy--Isouard     16.Qb8+        31    1.49   13.68        100.0
  Byrne--Fischer      17...Be6       34    1.86    8.04         98.8
  Kasparov--Topalov   24.Rxd4        47    2.29    6.26         94.3
  Aronian--Anand      16...Nde5      32    5.21    5.74         73.9

  : Entropy and surprisal at the selected sacrificial move of each game
  ($\beta=1$). Percentile: share of plies in the same game with
  $I_t-H_t$ at or below the value of this move. {#tab:sac}

## Comparison with null ensembles

Table [\[tab:null\]](#tab:null){reference-type="ref"
reference="tab:null"} and Fig. [3](#fig:z){reference-type="ref"
reference="fig:z"} give the $z$-scores of the real games. Only three of
the 60 comparisons reach $|z|\ge3.3$: the immune fraction of Morphy's
game against random play ($z=+5.6$), its entropy drop against
model-sampled play ($z=+4.1$) and the surprisal--entropy gap of
Kasparov--Topalov against model-sampled play ($z=+3.6$).

Against random play, three patterns held in all five games, although
mostly below the marker. The surprisal--entropy gap was smaller in real
games ($0.11$--$0.40$ bits against a null mean of $2.32$--$3.97$;
$z=-1.4$ to $-2.6$), the immune fraction was larger ($z=+1.2$ to $+5.6$)
and the peak number of available captures was smaller ($z=-0.7$ to
$-2.5$). The mean entropy was higher than in random play ($z=+0.6$ to
$+3.0$), and the entropy drop was not distinguishable ($z=-1.1$ to
$+1.0$; the null mean was 1.2--1.5 bits).

Against model-sampled play, the surprisal--entropy gap was positive in
all five games ($z=+0.7$ to $+3.6$), the entropy drop was positive in
four ($z=+1.7$ to $+4.1$; Aronian--Anand: $0.0$), and the immune
fraction was close to the null ($z=-0.8$ to $+0.3$, except Morphy at
$+2.7$). The control-to-piece ratio was close to both nulls in the two
positional games (Karpov--Kasparov and Aronian--Anand, $|z|\le0.8$
against both) and higher in Morphy ($z=+2.5$ against A, $+3.0$ against
B).

<figure id="fig:z" data-latex-placement="t">
<embed src="fig_z.pdf" />
<figcaption><span class="math inline"><em>z</em></span>-scores of the
five real games against the two null ensembles for six observables. The
shaded band marks <span
class="math inline">|<em>z</em>| &lt; 3.3</span>.</figcaption>
</figure>

## Sensitivity to the ensemble temperature

Table [\[tab:beta\]](#tab:beta){reference-type="ref"
reference="tab:beta"} repeats the entropy analysis at $\beta=0.5$, 1 and
2. The surprisal--entropy gap is negative at $\beta=0.5$ ($-0.18$ to
$-0.28$ bits), small and positive at $\beta=1$ and large and positive at
$\beta=2$ ($+1.25$ to $+1.68$ bits). The entropy drop $\Delta H$ grew
with $\beta$ in every game except Byrne--Fischer, where it was nearly
flat (1.37, 1.40, 1.44), and for Aronian--Anand it changed sign (from
$-0.34$ at $\beta=0.5$ to $+2.37$ at $\beta=2$). The null ensembles were
generated at $\beta=1$ only.

## Outcome of the hypotheses

Table [2](#tab:hyp){reference-type="ref" reference="tab:hyp"} states how
each hypothesis fared.

       Verdict            Evidence
  ---- ------------------ -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  H1   Not supported      Peak capture availability occurred at 0.23--0.72 of game length with no common window; clustering was not tested formally.
  H2   Partly supported   $\Delta H>0$ in all five games at $\beta=1$, but about equally large in random play, and its sign for Aronian--Anand depends on $\beta$. Surprisal exceeded entropy most at the sacrifices of Morphy, Fischer and Kasparov (94th--100th percentile), less at Anand's.
  H3   Partly supported   $R_{\mathrm{c/p}}>1$ in all games (1.70--2.81) and higher in the three sharp games (2.58--2.81) than the two positional ones (1.70--2.09), but random and model-sampled games gave similar values (1.8--2.4).
  H4   Weakly supported   3 of 60 comparisons exceed $|z|=3.3$; the most consistent difference is a smaller surprisal--entropy gap than in random play.
  H5   Not supported      Rank correlation $\rho=0.40$ ($n=5$, $p\approx0.5$); queens survived most and knights least.

  : Outcome of the pre-specified hypotheses. {#tab:hyp}

# Discussion {#sec:discussion}

#### What the model measures.

The framework turns a game record into a compact, interpretable set of
observables, and these differ between games in sensible ways: the three
sharp games have a higher control-to-piece ratio than the two positional
ones, and the sacrificial moves of Morphy, Fischer and Kasparov stand
out in surprisal. Both effects are what one would expect from
experience, and we report them as consistency checks rather than
discoveries.

#### Much of the structure is kinematic.

Random play shows monotone density loss, an entropy drop of 1.2--1.5
bits and a control field that is more active than the piece field. These
features follow from the rules and from material leaving the board,
since $N_t$ shrinks as pieces disappear. A statement such as "decision
entropy decays during a game" therefore tells us about the lattice gas
more than about the players. The comparison with model-sampled play is
the more informative one, and there real games lose entropy faster in
four of five cases at $\beta=1$.
Table [\[tab:beta\]](#tab:beta){reference-type="ref"
reference="tab:beta"} shows that this conclusion depends on $\beta$.

#### The surprisal test is partly circular.

The heuristic energy penalises moving a piece onto an attacked square,
and a sacrifice is exactly such a move. Its high surprisal therefore
shows that the model flags moves that give up material, not that
sacrifices are special in a physical sense. Telling a brilliant
sacrifice from a blunder needs engine-based energies, which the model's
structure allows but this study did not use.

#### Survival and outcome.

Surviving weight at the end predicted the winner in only two of five
games. This is expected: the king has zero weight, so a mating attack is
invisible to the weight measure, and Morphy's win leaves the winner with
the lower surviving weight. Material is one of several conserved-like
quantities, and king safety would need its own observable.

#### The annihilation convention.

A chess capture is asymmetric, since the capturing piece survives; our
merge into one particle is a bookkeeping choice. We did not test a
symmetric accounting in which the capturer is also charged part of the
victim's weight, and we cannot say how much the reported values of
$\varphi$ depend on the convention.

#### Limitations.

The study has five games chosen for fame, not drawn at random, so
nothing here is a statement about chess in general. Move energies come
from a simple heuristic, and the weights were fixed at conventional
values with no sweep. The null ensembles used a single $\beta$. The
$z$-scores come from distributions of 1000 games, so the extreme tail is
only roughly resolved, and with 60 comparisons the $|z|\ge3.3$ marker is
a rule of thumb, not a formal correction.

#### Transfer to other systems.

The pipeline needs only a record of which site was occupied by which
species at each time, and which reactions occurred.
Table [3](#tab:map){reference-type="ref" reference="tab:map"} gives
proposed mappings.

  Domain               Species                     Exclusion                   Reaction               Decision
  -------------------- --------------------------- --------------------------- ---------------------- --------------------------
  Reaction kinetics    Two reactants               One particle per site       $A+B\to0$ on contact   None (random walk)
  Ecology              Two competing populations   One individual per patch    Fatal encounter        Foraging, movement
  Order-book markets   Buy and sell orders         One order per price level   Order matching         Where and when to submit
  Immunology           Pathogen and immune cells   One cell per binding site   Neutralisation         Which target to engage
  Conflict             Opposing forces             One unit per location       Attrition              Commit or hold

  : Proposed mappings of the model onto other domains (analogies,
  untested). {#tab:map}

# Conclusion {#sec:conclusion}

Chess can be described as an annihilating Boolean lattice gas whose
observables---density, immune fraction, surviving weight, Boolean
activity, decision entropy and surprisal---are computed directly from a
game record. On five verified classic games the framework gives
consistent and interpretable output and flags sacrificial moves by their
surprisal. At this stage, with heuristic move energies and five games,
it does not show that real games differ strongly from random or
model-sampled play: three of 60 comparisons were clearly outside the
null, several apparent regularities are kinematic, and the entropy
conclusions depend on the ensemble temperature. The value of the work
for now is the measurement framework and the open tool, which are ready
for engine-based energies and larger corpora, and for the first tests
outside chess.

# Data and code availability {#sec:data .unnumbered}

The software (`chess_physics.py`), the study scripts for data
verification, null ensembles and statistics, the verified PGN files,
per-ply tables and figures are provided with this article. The move
records were transcribed from the following pages (accessed 22 October
2026); in each pair the first source was used for analysis and the
second for verification.

- Morphy, 1858: <https://en.wikipedia.org/wiki/Opera_Game>;
  <https://gameknot.com/annotation.pl/the-opera-game?gm=12022>

- Byrne--Fischer, 1956:
  <https://en.wikipedia.org/wiki/Game_of_the_Century_(chess)>;
  <https://en.chessbase.com/news/2006/games/fischer1956g0.htm>

- Karpov--Kasparov, 1985, game 16:
  <https://en.wikipedia.org/wiki/World_Chess_Championship_1985>;
  <https://chess.com/blog/ThummimS/world-chess-championship-1985-game-16-karpov-vs-kasparov>.
  A third record in descriptive notation
  (<https://csmonitor.com/1985/1106/lchs06.html>) agrees in length and
  result but was not machine-compared.

- Kasparov--Topalov, 1999:
  <https://en.wikipedia.org/wiki/Kasparov%27s_Immortal>;
  <https://chess.com/blog/Eshwar360/the-best-chess-game-of-all-time-garry-kasparov-vs-veselin-topalov-1999>

- Aronian--Anand, 2013:
  <https://en.wikipedia.org/wiki/Anand%27s_Immortal>;
  <https://en.chessbase.com/home/TabId/211/PostId/4008776>

# Competing interests {#competing-interests .unnumbered}

The author declares no competing interests.

[^1]: Corresponding author: borthakur.gg@gmail.com
