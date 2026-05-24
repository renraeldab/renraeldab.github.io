---
title: "GMM and EM"
date: "none"
tags: ["machine-learning"]
---

Consider this hero level statistics from an RPG game: instead of an expected bell-shaped distribution, we observe a 
bimodal distribution with high counts at both ends and a dip in the middle.

![](rpg_heros.png)

This pattern tells a compelling story about player psychology. Players tend to categorize heroes into two groups: meta 
heroes that they prioritize leveling up to maximum potential, and weak heroes that receive minimal investment. Each 
group follows its own normal distribution, and when combined, these two distributions create the bimodal pattern we 
observe.

This is precisely where Gaussian Mixture Models (GMM) come into play, a powerful statistical tool that allows us to 
model complex data as a combination of multiple Gaussian distributions, revealing hidden structures and patterns that 
single-distribution models would miss.

## Gaussian Mixture Models

A Gaussian Mixture Model (GMM) is a probabilistic model that assumes data are generated from a finite mixture of $k$ Gaussian components. Each component has its own mean $\mu_j$ and covariance $\Sigma_j$, and the overall distribution is a weighted combination of these individual Gaussians.

### Definition

Let $z$ denote the hidden (latent) variable indicating which component generated a data point. The prior probability of belonging to component $j$ is

$$
p(z = j) = \pi_j, \text{ with } \sum_{j = 1}^k \pi_j = 1.
$$

Given $z = j$, the observed variable $x$ follows a Gaussian distribution parameterized by that component:

$$
p(x \mid z = j) = \mathcal{N}(x \mid \mu_j, \Sigma_j).
$$

Because $z$ is unobserved, the marginal distribution of $x$ is obtained by summing over all possible components, yielding a weighted sum of Gaussians:

$$
p(x) = \sum_{j = 1}^k \pi_j \, \mathcal{N}(x \mid \mu_j, \Sigma_j).
$$

### Generative Process vs. Observation

We can view a GMM as defining a simple two-step generative process:

1. First, sample a component assignment $z \sim p(z)$.
2. Then, sample the observation conditioned on that assignment: $x \mid z \sim p(x \mid z)$.

In practice, however, we only observe the data $x$; the latent assignments $z$ are hidden from us. We need to infer these hidden states and estimate the model parameters from observed data.

## Maximum Likelihood Parameter Estimation

Suppose we have a dataset $X = \{x_1, \dots, x_n\}$ drawn from a mixture of $k$ Gaussian distributions. If we knew the cluster assignment $z_i \in \{1, \dots, k\}$ for each data point, we could treat the assignments as parameters and maximize the joint log-likelihood of the complete data $(x_i, z_i)$:

$$
\underset{\mu, \Sigma, z}{\mathrm{argmax}} \sum_{i = 1}^n \log p(x_i, z_i)
= \underset{\mu, \Sigma, z}{\mathrm{argmax}} \sum_{i = 1}^n \bigl(\log \pi_{z_i} + \log \mathcal{N}(x_i \mid \mu_{z_i}, \Sigma_{z_i})\bigr).
$$

To make the optimization more explicit, we can rewrite each $z_i$ using a one-hot encoding $z_{ij}$, where

$$
z_{ij} = \begin{cases}
1 & \text{if } z_i = j, \\
0 & \text{otherwise}.
\end{cases}
$$

Since every data point must belong to exactly one cluster, we have $\sum_{j=1}^k z_{ij} = 1$ for each $i$. Using this representation, the mixture probability and Gaussian density for a single point can be expressed as products over all components:

$$
\pi_{z_i} = \prod_{j = 1}^k \pi_j^{z_{ij}}, \qquad
\mathcal{N}(x_i \mid \mu_{z_i}, \Sigma_{z_i}) = \prod_{j = 1}^k \mathcal{N}(x_i \mid \mu_j, \Sigma_j)^{z_{ij}}.
$$

Substituting these into the objective gives a single double summation:

$$
\underset{\mu, \Sigma, z}{\mathrm{argmax}} \sum_{i = 1}^n \sum_{j = 1}^k z_{ij}\bigl(\log \pi_j + \log \mathcal{N}(x_i \mid \mu_j, \Sigma_j)\bigr).
$$

Unfortunately, we cannot maximize this directly because the latent assignments $z_{ij}$ are unknown. Optimizing over both the parameters and the discrete assignments simultaneously leads to a combinatorial problem with many local optima.

## Expectation-Maximization

Expectation-Maximization (EM) algorithm** is an iterative method for
finding maximum likelihood estimates (MLE) or maximum a posteriori (MAP)
estimates when data contains latent (hidden) variables or is
incomplete.

### Overview

Given observed data $X$, latent variables $Z$, and parameters $\theta$, we
want to maximize the observed-data log-likelihood:
$$
\log p(X \mid \theta) = \log \sum_{Z} p(X, Z \mid \theta)
$$

Introduce an arbitrary probability distribution $q(Z)$ over the latent
variables with $q(Z) > 0$. We can rewrite:

$$
\log p(X \mid \theta) = \log \sum_{Z} q(Z) \frac{p(X, Z \mid \theta)}{q(Z)}
= \log \mathbb{E}_Z[\frac{p(X, Z \mid \theta)}{q(Z)}]
$$

According to Jensen's Inequality, for a concave function $f$

$$
f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)]
$$

Since $\log$ is concave, we have:

$$
\log p(X \mid \theta) \ge \mathbb{E}_Z[\log \frac{p(X, Z \mid \theta)}{q(Z)}]
= \sum_{Z} q(Z) \log \frac{p(X, Z \mid \theta)}{q(Z)}
\triangleq \mathcal{L}(q, \theta)
$$

$\mathcal{L}(q, \theta)$ is called the **Evidence Lower BOund (ELBO)**.
The gap between the true log-likelihood and the ELBO is a KL divergence:

$$\log p(X|\theta) = \mathcal{L}(q, \theta) + \text{KL}\big(q(Z) \,\|\,
p(Z|X,\theta)\big)$$

Since $\text{KL} \geq 0$, maximizing $\mathcal{L}$ pushes us closer to
maximizing $\log p(X|\theta)$.



### ￿ 3. The Two Steps
EM alternates between **tightening the bound** and **pushing it upward**.
#### ￿ E-Step (Expectation)
**Fix $\theta = \theta^{(t)}$. Maximize $\mathcal{L}(q, \theta^{(t)})$
w.r.t $q(Z)$.**
From the KL decomposition, $\mathcal{L}$ is maximized when the KL
divergence is zero:
$$q(Z) = p(Z|X, \theta^{(t)})$$
This makes the bound **tight**: $\mathcal{L}(q, \theta^{(t)}) = \log
p(X|\theta^{(t)})$.
In practice, we don’t keep $q$ as a full distribution. Instead, we compute
the **expected complete-data log-likelihood** (often called the $Q$-
function):
$$Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z \sim p(Z|X,\theta^{(t)})} \big[
\log p(X, Z|\theta) \big]$$
*(The term $-\mathbb{E}[\log q(Z)]$ drops out in the M-step since it
doesn’t depend on $\theta$.)*
#### ￿ M-Step (Maximization)
**Fix $q(Z) = p(Z|X, \theta^{(t)})$. Maximize $\mathcal{L}(q, \theta)$
w.r.t $\theta$.**
Since $q$ is fixed, maximizing $\mathcal{L}$ is equivalent to maximizing
$Q(\theta, \theta^{(t)})$:
$$\theta^{(t+1)} = \arg\max_{\theta} Q(\theta, \theta^{(t)})$$
This is usually a standard MLE problem, but now the "data" includes soft
assignments from the E-step.
---
### ￿ 4. Why It Works (Monotonic Improvement)
Each EM iteration guarantees **non-decreasing observed log-likelihood**:
1. **E-step**: Tightens the lower bound to touch $\log
p(X|\theta^{(t)})$.
2. **M-step**: Increases the lower bound to $\mathcal{L}(q,
\theta^{(t+1)})$.
3. Since the true likelihood is always $\geq$ the ELBO, we get:
$$\log p(X|\theta^{(t)}) \leq \mathcal{L}(q, \theta^{(t+1)}) \leq \log
p(X|\theta^{(t+1)})$$
Thus, $\log p(X|\theta)$ climbs monotonically until convergence.
---
### ￿￿ 5. Key Properties & Caveats
| Property | Detail |
|----------|--------|
| **Convergence** | Guaranteed to converge to a **local optimum** (or
saddle point) of the likelihood. |
| **Global optimum** | Not guaranteed. Sensitive to initialization.
Multiple random starts are common. |
| **Tractability** | Requires: (1) closed-form or computable posterior
$p(Z|X,\theta)$, (2) easy maximization of $Q$. |
| **Speed** | Often slow near convergence (linear convergence rate).
Accelerated variants exist. |
| **Identifiability** | Label switching, degenerate solutions (e.g., zero
variance in GMMs) can occur. |
---
### ￿ 6. Canonical Example: Gaussian Mixture Models (GMM)
- **Observed**: $X = \{x_1, \dots, x_N\}$
- **Latent**: $z_n \in \{1,\dots,K\}$ (cluster assignment for $x_n$)
- **Parameters**: $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$
**E-step**: Compute *responsibilities* (posterior cluster probabilities):
$$\gamma_{nk} = p(z_n=k | x_n, \theta^{(t)}) = \frac{\pi_k^{(t)}
\mathcal{N}(x_n | \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_j \pi_j^{(t)}
\mathcal{N}(x_n | \mu_j^{(t)}, \Sigma_j^{(t)})}$$
**M-step**: Update parameters using weighted MLE:
$$\pi_k^{(t+1)} = \frac{1}{N}\sum_n \gamma_{nk}, \quad \mu_k^{(t+1)} =
\frac{\sum_n \gamma_{nk} x_n}{\sum_n \gamma_{nk}}, \quad \Sigma_k^{(t+1)}
= \frac{\sum_n \gamma_{nk} (x_n-\mu_k^{(t+1)})(x_n-
\mu_k^{(t+1)})^\top}{\sum_n \gamma_{nk}}$$
Repeat until $\|\theta^{(t+1)} - \theta^{(t)}\| < \epsilon$.
---
### ￿ Summary
- **Goal**: Maximize $\log p(X|\theta)$ with latent $Z$.
- **Trick**: Introduce $q(Z)$, use Jensen’s inequality to get ELBO.
- **E-step**: Set $q(Z) = p(Z|X,\theta^{(t)})$ → compute expected
complete log-likelihood.
- **M-step**: Maximize that expectation w.r.t $\theta$.
- **Result**: Monotonic likelihood increase, converges to a local
optimum.