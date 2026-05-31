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

The Expectation-Maximization (EM) algorithm is an iterative method for
finding maximum likelihood estimates (MLE) or maximum a posteriori (MAP)
estimates when data contain latent (hidden) variables or are
incomplete.

### Overview

Given observed data $X$, latent variables $Z$, and parameters $\theta$, we
want to maximize the observed-data log-likelihood:
$$
\log p(X \mid \theta) = \log \sum_{Z} p(X, Z \mid \theta).
$$

Because the summation sits inside the logarithm, the parameters are tightly
coupled and this objective is difficult to optimize directly. The key idea
behind EM is to construct a tractable lower bound and maximize the bound
instead.

To derive that bound, introduce an arbitrary distribution $q(Z)$ over the latent
variables. We can then rewrite:

$$
\log p(X \mid \theta) = \log \sum_{Z} q(Z) \frac{p(X, Z \mid \theta)}{q(Z)}
= \log \mathbb{E}_{q(Z)}\!\left[\frac{p(X, Z \mid \theta)}{q(Z)}\right].
$$

According to Jensen's inequality, for a concave function $f$,

$$
f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)].
$$

![](jensens_inequality.png)

Since $\log$ is concave, we have:

$$
\log p(X \mid \theta) \ge \mathbb{E}_{q(Z)}\!\left[\log \frac{p(X, Z \mid \theta)}{q(Z)}\right]
= \sum_{Z} q(Z) \log \frac{p(X, Z \mid \theta)}{q(Z)}
\triangleq \mathcal{L}(q, \theta).
$$

$\mathcal{L}(q, \theta)$ is called the **Evidence Lower Bound (ELBO)**.
The gap between the true log-likelihood and the ELBO is the KL divergence:

$$\log p(X \mid \theta) = \mathcal{L}(q, \theta) + \mathrm{KL}\big(q(Z) \,\|\,
p(Z \mid X, \theta)\big).$$

Since $\mathrm{KL} \geq 0$, maximizing $\mathcal{L}$ pushes us closer to
maximizing $\log p(X \mid \theta)$.

EM alternates between **tightening the bound** and **pushing it upward**.

### E-Step

**Fix $\theta = \theta^{(t)}$ and maximize $\mathcal{L}(q, \theta^{(t)})$
with respect to $q(Z)$.**
From the KL decomposition, $\mathcal{L}$ is maximized when the KL
divergence vanishes:
$$q(Z) = p(Z \mid X, \theta^{(t)}).$$
This makes the bound **tight**: $\mathcal{L}(q, \theta^{(t)}) = \log
p(X \mid \theta^{(t)})$.
In practice, we do not need to represent $q$ explicitly. Instead, we compute
the **expected complete-data log-likelihood**, often called the $Q$-
function:
$$Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z \sim p(Z \mid X,\theta^{(t)})} \big[
\log p(X, Z \mid \theta) \big].$$
Because the entropy term $-\mathbb{E}[\log q(Z)]$ does not depend on $\theta$,
maximizing $\mathcal{L}$ over $\theta$ is equivalent to maximizing $Q$.

### M-Step

**Fix $q(Z) = p(Z \mid X, \theta^{(t)})$ and maximize $\mathcal{L}(q, \theta)$
with respect to $\theta$.**
Since $q$ is fixed, maximizing $\mathcal{L}$ is equivalent to maximizing
the $Q$-function:
$$\theta^{(t+1)} = \arg\max_{\theta} Q(\theta, \theta^{(t)}).$$
This is usually a standard MLE problem, but the "data" now include soft
assignments from the E-step.

### Monotonic Improvement

Each EM iteration guarantees a **non-decreasing observed log-likelihood**, illustrated below:

![](em_elbo.png)

1. **E-step**: Tightens the lower bound to touch $\log
p(X \mid \theta^{(t)})$.
2. **M-step**: Increases the lower bound to $\mathcal{L}(q,
\theta^{(t+1)})$.
3. Since the true likelihood is always at least the ELBO, we get:
$$\log p(X \mid \theta^{(t)}) \leq \mathcal{L}(q, \theta^{(t+1)}) \leq \log
p(X \mid \theta^{(t+1)}).$$
Thus, $\log p(X \mid \theta)$ climbs monotonically until convergence.

> Complete-data MLE pretends we know the latent
> assignments $z_i$ and optimizes $\log p(X, Z \mid \theta)$ directly.
> Observed-data MLE, which is what EM targets, instead integrates over that
> uncertainty via marginalization.

## GMM EM

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
