---
abstract: |
  Flood risk management has become a major area of research for those
  institutions and administrations endorsed with forecasting and damage
  mitigation, not only due to precipitation but also due to
  infrastructure failure. *Deterministic-8* is a standard algorithm in
  the sequence of routines applied to predict the way a volume of fluid
  will distribute over the land accessible to it. The continuous version
  of the problem is a dynamical one with constrictions for which a
  traditional Lagrange multipliers approach is natural. A lattice
  version of it is NP-complete complexity-wise and computationally is
  well suited for a quantum approach the latter being the object of this
  paper.
title: |
  QUBO for deterministic-8 and flow accumulation:\
  another edge between Quantum Computation and Hydrology
authors:
  - name: Jaime Anguiano Olarra
    orcid: 0000-0002-6447-8180
    affiliation: 1
affiliations:
 - name: Independent Researcher, Spain
   index: 1
bibliography: apssamp.bib
date: 2026-05-05
nocite: "`\\nocite{*}`{=latex}"

---

# []{#sec:level1 label="sec:level1"}INTRODUCTION:

Flood risk assessment and subsequent damage prevention is gaining
attention in the recent years. Even more acute is the urge for enhancing
the prediction tools available for the agents watching regions where the
actual models are lacking or outdated due to new trends in precipitation
patterns, land-use, obsolete infrastructures or new legislation (many
plans are requiring longer *return periods*) in many circumstances, all
these issues concurring all at once.\
Dramatic flash-flood events like the one that hit Spain on October 29th,
2024, with over 230 deaths, and which could have been increased by one
or two orders of magnitude had a nearby dam fail, as it seemed plausible
at the time (Calvo-Sancho et al. 2026), are of special interest.
*Deterministic-8* is a widely used routine in the pre-processing stages
of hydrologic studies for flood planning and is provided in industry
standard toolkits like SAGA(Conrad et al. 2015). The goal of the problem
is forecasting how a certain amount of water initially localized over
some area will spread in its surrounding area. Machine Learning Quantum
Computing(Combarro and González-Castillo 2023, 2025) has just not
entered yet fully in this active field of Hydrology but the problems
posed are very suitable for its application. The rest of this manuscript
is devoted to build a general Quadratic Unconstrained Binary
Optimization (QUBO) representation on paper containing the
Deterministic-8 method, applying it to selected cases to showcase its
use, ending with conclusions.

# []{#sec:level2 label="sec:level2"}Theoretical Formulation

We follow notation introduced by O'Malley(O'Malley 2018) whenever
possible.

## []{#sec:level2 label="sec:level2"}Deterministic-8 General QUBO

Deterministic-8 is a method self-described by its own name, being the
first solution anyone can think of when faced with deciding, like in a
boardgame, how to move in a square grid from one square to the next one.
Simply put: explore all the next cells where you can move to check if
doing so brings you closer to your goal, moving to the most favorable
ones. Several at the same time, or many times, when time presses,
shifting to only one just like as a *one hot* problem. This is precisely
how it is done in the SAGA tools(Conrad et al. 2015), an industry
standard in the field of pre-processing Digital Elevation Models for
later use in the modeling software later involved (like HEC-RAS(US Army
Corps of Engineers, n.d.), QGIS, iRIC or Iber). This routine can be
considered a checkpoint before any more detailed hydrologic analysis
takes place.

In our present case, identification with the 2-D model discussed by
O'Malley (O'Malley 2018) is direct. Keeping his notation we just give a
different interpretation to the magnitudes behind the variables. This is
better seen through an example. We start at time $t = 0$ with a 3x3 grid
of vertices which will be our potentially flooded area, and we set on
the vertex $h_{2,2}^0$ our aquifer in (O'Malley 2018) scheme. The
variables are binary, $h_{i,j}^t$, has value 1 if it is filled and value
0 if it is empty, and indexing starts at $1$: $i,j = {1,2,3...}$. Just
like in knapsack, a multiplicative constant $\alpha_{i,j}^t$ accounts
for the height of free surface of water of that cell. The area of each
cell is unitary (a simplification which is also standard in industry as
seen for example in how the accumulated flow is computed in SAGA). We
will call this grid the *run-off layer*. Notice that up to this point
the layer lacks edges. We tackle this problem immediately.

Define a second grid of the same dimensions which serves as a faithful
representation of the terrain, the Digital Elevation Model in jargon.
Position it right below the previous one giving a value of elevation
$z_{i,j}^t$ to each vertex by the same binary variable times a
multiplicative constant assignment but with the difference that in this
case all vertices are non-zero. Setting the value for the edges equal to
the difference between the values of the vertices they connect,
subtracting the arrival vertex value (height) from the departure vertex
value and removing all the vertices of this middle grid, we are left
with a subgraph which can be plugged into the *run-off* grid to supply
the edges to the $h_{i,j}$ grid. These subgraph's values are signed,
they represent the gravitational gradients. We will reuse O'Malley's
label $k_{i,j}^\mu$ even though here it is not permeability. The
subscript $\mu$ stands for the different spatial dimensions along the
water moves in the plane, taking values from $\mu = {x, y}$ whereas the
subscripts indicate which vertex was taken as departure, so f.ex.
$k_{1,2}^x$ uniquely identifies the edge that moving along $x$, connects
from the vertex $h_{1,2}^t$ to the vertex $h_{2,2}^t$. Thus we have
constructed what in Hydrology is called a channel network.

![[]{#fig:wide label="fig:wide"}The three 3x3 grid layers: The *run-off*
layer in blue, on top. Its vertices are binary valued, the value of the
heads are parameters $\alpha_{i,j}$ not shown here. The terrain layer in
green at the bottom and a middle channel network obtained from signed
differences in the aggregated values of head + height values of adjacent
cells. Time subscripts have been omitted for
simplicity.](fig1.jpg){#fig:wide}

The function to optimize is the negative of the potential energy. i.e.
we want $L$ to be as negative as we can: $$\begin{equation}
  L_{free} = -V = -  \sum_{i,j}(\alpha_{i,j}^t + z_{i,j}^t)h_{i,j}^t
\end{equation}$$ subject to two constrains. The first one is mass
conservation: $$\begin{equation}
    \alpha_{i,j}^0h_{i,j}^0 = \sum_{i,j}\alpha_{i,j}^th_{i,j}^t
\end{equation}$$ (water spreads, but no water is lost, this forces on us
to set a lower bound to $\alpha_{i,j}$ where to stop covering more area
which we set to $1$). The second is that inertia alone drives the
dynamics: water covers cells according to gravity, minimizing its
overall gravitational energy, following general principles of
Statistical Mechanics(Costa and Pessoa 2021). This potential to change
vertex will be represented by an interaction term between the content of
neighboring cells We build the later using the following steps. The
direction of movement is signed as we saw, for example for moving along
the $x$ axis: $$\begin{equation}
  \frac{z_{i+1,j}^t - z_{i,j}^t }{|z_{i+1,j}^t - z_{i,j}^t|}=\begin{cases}-1, &  z_{i+1,j}^t < z_{i,j}^t \\0,  &  z_{i+1,j}^t = z_{i,j}^t\\1,  &  z_{i+1,j}^t > z_{i,j}^t 
\end{cases}
\end{equation}$$ This factor serves as a switch to automatically turn
*on* or *off* the inclusion of the interaction term. Only if it is *on*
there is a gateway to lose energy by occupying new cells. Using the
$k_{i,j}^\mu$ notation: $$\begin{equation}
  \frac{k_{i,j}^{\mu, t}}{|k_{i,j}^{\mu, t}|}=\hat{k}_{i,j}^{\mu,t}=\begin{cases}-1, &  k_{i,j}^{\mu,t} < 0\\0,  &  k_{i,j}^{\mu,t} = 0\\1,  &  k_{i,j}^{\mu,t} > 0
\end{cases}
\end{equation}$$ It is not obvious but this factor does not suffice for
its purpose. The problem is that if an adjacent cell is at the same
height it seems there is no reason for water to spill into it. But
imagine that the previous cell has some water on it, it will spill till
it reaches the lower bound for $\alpha_{i,j}^t$. This is realized by
merely replacing the $z_{i,j}^t$ by $\alpha_{i,j}^t + z_{i,j}^t$,
consider this done and accounted for by the $\hat{k}_{i,j}^t$s This
leads to a bare Lagrangian interaction term that turns on when the right
conditions apply allowing movement, which translates in activating new
$h_{i,j}^t$ vertices flipping their value from their initial $0$ to $1$:
$$\begin{equation}
  L_{int}^{bare} = -\frac{1}{2}\Bigl(1-\hat{k}_{i,j}^{\mu,t}\Bigr)=\begin{cases}-1, & \text{if } \hat{k}_{i,j}^{\mu,t} = -1 \\0,  & \text{otherwise } 
\end{cases}
\end{equation}$$ Notice that it embodies a binary variable in itself.
Finally we need to encode by how much does it afflict the bare
Lagrangean. In this case it is the change in overall potential energy:
$$\begin{multline}
  L_{int} = -\frac{1}{2}\Bigl(1-\hat{k}_{i,j}^{\mu,t}\Bigr)\sum_{i,j}[(\alpha_{i,j}^{t+1} + z_{i,j}^t)h_{i,j}^{t+1}-\\(\alpha_{i,j}^{t} + z_{i,j}^t)h_{i,j}^{t}]
\end{multline}$$ And the full QUBO: $$\begin{multline}
  L =  -  \sum_{i,j}(\alpha_{i,j}^t + z_{i,j}^t)h_{i,j}^t -\\\frac{1}{2}\Bigl(1-\hat{k}_{i,j}^{\mu,t}\Bigr)\sum_{i,j}[(\alpha_{i,j}^{t+1} + z_{i,j}^t)h_{i,j}^{t+1}-(\alpha_{i,j}^{t} + z_{i,j}^t)h_{i,j}^{t}] + \\\gamma \sum_{i,j} (\alpha_{i,j}^0h_{i,j}^0 - \sum_{i,j}\alpha_{i,j}^th_{i,j}^t)
\end{multline}$$ which simplifies to:

$$\begin{multline}
  L =  -  \frac{1}{2}\sum_{i,j}\Bigl[\Bigl([(1-\gamma) + (1+\gamma)\hat{k}_{i,j}^{\mu,t}]\alpha_{i,j}^t + (1+\hat{k}_{i,j}^{\mu,t})z_{i,j}^t\Bigr)h_{i,j}^t -\\\Bigl(1-\hat{k}_{i,j}^{\mu,t}\Bigr)\sum_{i,j}[(\alpha_{i,j}^{t+1} + z_{i,j}^t)h_{i,j}^{t+1} + \gamma \alpha_{i,j}^0h_{i,j}^0 \Bigr]
\end{multline}$$

## []{#sec:level3 label="sec:level3"}Physical considerations

In this model no other physical effect is considered. Nonetheless see
that it is legitimate to let the amount of water vary at anytime (*e.g.*
rainfall, release of a dam) or rate by varying the $\alpha_{i,j}^t$
consistently. That the most efficient way to come to the ground-state is
by flowing along the largest $\hat{k}_{i,j}$s was not explicitly written
in anyway and it is expected that this behavior will emerge naturally.
Before moving onto the software implementation *per se*, it is worth
mentioning that the diffusion of water discretized as in here is truly a
good example of an Ising model where the interaction fits very well the
next-neighbor condition as even if we were to address more complex fluid
dynamics, the next-to-next neighbor structure is built-in for the
problem of spreading of water flow, at least when far from supersonic or
high cavitation regimes.

# []{#sec:level4 label="sec:level4"}*In Machina* implementation description

We have developed our model, now we apply it to some showcase examples
growing in difficulty.

## []{#sec:level4 label="sec:level4"}Showcase examples

### []{#sec:level1 label="sec:level1"}**1-D Dam failure**

We remove the $j$ index as there is no need for it. This is the most
simple model of dam failure. We have two units of water on the only cell
(vertex) active and at some indeterminate time between time $t = 0$ and
$t = 1$ we remove the constrain letting the water be able to flood the
adjacent cell.\
\
[**Initial state:**]{.smallcaps}\
[Terrain:]{.smallcaps} $z_0^0 = z_1^0 = 1$ (our *Digital Elevation
Model*)\
$$\begin{equation}
    \left[h_0^{t=0} = 1, \alpha_0^{t=0} = 2\right]
\end{equation}$$ The only
$\hat{k}_0^0 = \frac{(1+0) - (1+1)}{|(1+0)-(1+1)|}=-1$ so even if the
channel network suggest that no flooding should be expected (the terrain
is flat), the fact that water won't stand showing some kind of
self-sustained \"hydraulic jump\" is effectively realized by the
inclusion of the $\alpha_{i,j}^t$ as part of the switch function.

[**Final state:**]{.smallcaps} $$\begin{equation}
    \left[\left[h_0^{t=1}=1, \alpha_0^{t=1} = 1\right], \left[h_1^{t=1}=1, \alpha_1^{t=1} = 1\right]\right]
\end{equation}$$

### []{#sec:level2 label="sec:level2"}**3-D flooding**

[**Initial state:**]{.smallcaps}

[Terrain:]{.smallcaps} This would be imported from Geopandas(Project,
n.d.) for example. $$\begin{equation}
z_{i,j}^0= \begin{bmatrix}
    z_{1,3}^0 = 6 & z_{2,3}^0 =5 & z_{3,3}^0 = 5\\
    z_{1,2}^0 = 7 & z_{2,2}^0 = 2 & z_{2,3}^0 = 5\\
    z_{1,1}^0 = 0 & z_{1,2}^0 = 4 & z_{1,3}^0 = 6
\end{bmatrix}
\end{equation}$$

![[]{#fig:wide label="fig:wide"}Sample
terrain](fig3Dterrain.jpg){#fig:wide}

As we see, the aquifer is located at $h_{2,2}^0$ and filled to three
units, i.e. $\alpha_{2,2}^0 = 3$, all the other $h_{i,j}^0$ are dry
(have value $0$ as well as all the other $\alpha_{i,j}^0$. The channel
network obviously only gives us a path from $z_{2,2}^0$ to $z_{1,2}^0$
to $z_{1,1}^0$. If we include the height of the free surface we get:
$$\begin{equation}
\alpha_{i,j}^0 + z_{i,j}^0= \begin{bmatrix}
     6 & 5 &  5\\
     7 &  5 & 5\\
     0 &  4 &  6
\end{bmatrix}
\end{equation}$$

![[]{#fig:wide label="fig:wide"}Channel
network](fig3channels.jpg){#fig:wide}

We can dramatically enhance movement by breaking a dam changing the
terrain, which we do setting $z_{12}^1 = 1$. So at time $t = 1$ a new
state with lower energy could be: [**State at t = 1:**]{.smallcaps}
$$\begin{equation}
    \left[\left[h_{2,2}^{t=1}= 1, \alpha_{2,2}^{t=1} = 2\right], \left[h_{1,2}^{t=1}= 1, \alpha_{1,2}^{t=1} = 1\right]\right]
\end{equation}$$

![[]{#fig:wide label="fig:wide"}Flow can be perceived in a less abstract
way by means of this figure at an intermediate
step.](fig4_t2.5_along_z21.png){#fig:wide}

And at time $t=\infty$ the final state needs to be the ground-state
which actually can be achieved at t=1 if both cells $z_{11}^1$ and
$z_{12}^1$ are flooded in one step:\
[**Final state:**]{.smallcaps} $$\begin{equation}
    \left[\left[h_{1,1}^{t=\infty}= 1, \alpha_{1,1}^{t=\infty} = 2\right], \left[h_{1,2}^{t=\infty}= 1, \alpha_{1,2}^{t=\infty} = 1\right]\right]
\end{equation}$$

This final time is the initial time of the next iteration of the
algorithm in a larger grid. In fact, given a larger grid and controlling
how much flow can move at each timestep enriches the dynamics notably as
what otherwise are accumulation points, the vertex $z_{1,1}^t$ in our
grid, can be mere parts of a channel even with negligible Strahler order
in an larger catchment area.

## []{#sec:citeref label="sec:citeref"}Conclusions

The simplicity of this proof of concept model makes it suitable to be
coded using almost any framework for quantum computing at the present
time. In D-Wave's Ocean SDK(D-Wave Systems Inc., n.d.) this can be
achieved using a quadratic model and an ExactSolver without requiring
running it on actual quantum architectures. Considering the smallness of
these showcases calls for sticking to the ExactSolver. A version of it
using D-Wave's SimulatedAnnealingSampler() is available(Olarra 2026). On
the other hand, a full-scale model for massive grids will certainly be
of much interest. This has not been pursued during the elaboration of
this first work but it is under development. The arrival of Quantum
Optimization to the field of flood risk assessment is very promising but
it is still not here. May this small step contribute to the collective
work towards this objective.\

::: acknowledgments
The author would like to thank Dr. Pedro Pessoa, at the Arizona State
University, US, and Prof. Juan Julian Merelo Guervos, at the University
of Granada, Spain, for useful discussions and review of this manuscript.
:::

:::::::::::::: {#refs .references .csl-bib-body .hanging-indent}
::: {#ref-CalvoSancho2026 .csl-entry}
Calvo-Sancho, Carlos, Javier Díaz-Fernández, Juan J. González-Alemán, et
al. 2026. "Human-Induced Climate Change Amplification on Storm Dynamics
in [Valencia's]{.nocase} 2024 Catastrophic Flash Flood." *Nature
Communications* 17 (1): 1492.
<https://doi.org/10.1038/s41467-026-68929-9>.
:::

::: {#ref-combarro2023practical .csl-entry}
Combarro, Elías F., and Samuel González-Castillo. 2023. *A Practical
Guide to Quantum Machine Learning and Quantum Optimization: Hands-on
Approach to Modern Quantum Algorithms*. Packt Publishing.
:::

::: {#ref-combarro2025practical .csl-entry}
Combarro, Elías F., and Samuel González-Castillo. 2025. *A Practical
Guide to Quantum Computing: Hands-on Approach to Quantum Computing with
Qiskit*. Packt Publishing.
:::

::: {#ref-Saga .csl-entry}
Conrad, O., B. Bechtel, M. Bock, et al. 2015. "System for Automated
Geoscientific Analyses (SAGA) v. 2.1.4." *Geoscientific Model
Development* 8 (7): 1991--2007.
<https://doi.org/10.5194/gmd-8-1991-2015>.
:::

::: {#ref-Stat_Mec_Unconf2021 .csl-entry}
Costa, Bruno Arderucio, and Pedro Pessoa. 2021. "Statistical Mechanics
of Unconfined Systems: Challenges and Lessons." *Physical Sciences
Forum* 3 (1). <https://doi.org/10.3390/psf2021003008>.
:::

::: {#ref-dwave_ocean_sdk .csl-entry}
D-Wave Systems Inc. n.d. "Ocean Software SDK." In *GitHub Repository*.
GitHub.
:::

::: {#ref-OMalley2018 .csl-entry}
O'Malley, Daniel. 2018. "An Approach to Quantum-Computational Hydrologic
Inverse Analysis." *Scientific Reports* 8 (1): 6919.
<https://doi.org/10.1038/s41598-018-25206-0>.
:::

::: {#ref-qubodet8 .csl-entry}
Olarra, Jaime Anguiano. 2026. *quboDet8*. Self.
<https://github.com/jaimeangola/jaimeangola.github.io>.
:::

::: {#ref-geopandas .csl-entry}
Project, The Geopandas. n.d. *Geopandas: Geospatial Analysis Toolkit*.
Self. <http://geopandas.org>.
:::

::: {#ref-Qiskit .csl-entry}
Qiskit contributors. n.d. *Qiskit: An Open-Source Framework for Quantum
Computing*. <https://doi.org/10.5281/zenodo.2562110>.
:::

::: {#ref-hec_ras_manual .csl-entry}
US Army Corps of Engineers. n.d. *HEC-RAS, River Analysis System*.
Hydrologic Engineering Center (HEC).
<https://www.hec.usace.army.mil/software/hec-ras/>.
:::
::::::::::::::

