# Minority game definition

## Static normal-form version

We define the single-shot minority game as a normal-form game

\[
\Gamma = \langle N, (S_i)_{i \in N}, (u_i)_{i \in N} \rangle
\]

where:

- \(N = \{1,\dots,n\}\) is the set of players
- each player \(i\) chooses a strategy \(s_i \in S_i = \{0,1\}\)
- \(s_i = 1\) means attend
- \(s_i = 0\) means stay home

Let aggregate attendance be

\[
A(s) = \sum_{i=1}^{n} s_i
\]

and let \(L\) be the attendance threshold.

The pay-off for player \(i\) is

\[
u_i(s)=
\begin{cases}
1 & \text{if } s_i = 1 \text{ and } A(s) \le L, \\
1 & \text{if } s_i = 0 \text{ and } A(s) > L, \\
-1 & \text{otherwise.}
\end{cases}
\]

This project uses the El Farol threshold version with typical example values \(n=101\) and \(L=60\).

## Mixed strategies

A mixed strategy for player \(i\) is a probability distribution over \(\{0,1\}\). In the simplest case,

\[
\sigma_i(1)=p_i, \qquad \sigma_i(0)=1-p_i
\]

with \(p_i \in [0,1]\).

## Repeated-game bridge

The repeated version uses the same stage game over multiple rounds \(t=1,\dots,m\), with cumulative pay-off

\[
U_i = \sum_{t=1}^{m} u_i^{(t)}.
\]

Later sections can then introduce adaptive or inductive strategy updates between rounds.
