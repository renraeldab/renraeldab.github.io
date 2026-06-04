---
title: "GMM and EM"
date: "2026-06-04"
tags: ["machine-learning"]
---

> **TL;DR** — This post explains Gaussian Mixture Models and the Expectation-Maximization algorithm from the ground up, 
> then applies them to a real customer segmentation problem. If you want to implement GMM from scratch, see the 
> [NumPy implementation](#implementation-in-numpy); if you prefer a ready-to-use pipeline, see the [sklearn wrapper](#complete-experiment-pipeline).

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

A Gaussian Mixture Model (GMM) is a probabilistic model that assumes data are generated from a finite mixture of $k$ 
Gaussian components. Each component has its own mean $\mu_j$ and covariance $\Sigma_j$, and the overall distribution is 
a weighted combination of these individual Gaussians.

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

In practice, however, we only observe the data $x$; the latent assignments $z$ are hidden from us. We need to infer these 
hidden states and estimate the model parameters from observed data.

### Sensitivity to Outliers

GMM is sensitive to outliers. The model uses maximum likelihood estimation with Gaussian distributions, which have light tails. 
Outliers can inflate covariance estimates and shift component means because the squared Mahalanobis distance penalizes outliers quadratically, 
but they still contribute to the likelihood. A single extreme outlier can distort an entire Gaussian component, pulling its mean toward the outlier and inflating its covariance.

![](gmm_outliers.png)

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

Unfortunately, we cannot maximize this directly because the latent assignments $z_{ij}$ are unknown. Optimizing over 
both the parameters and the discrete assignments simultaneously leads to a combinatorial problem with many local optima.

## Expectation-Maximization

The Expectation-Maximization (EM) algorithm is an iterative method for finding maximum likelihood estimates (MLE) or 
maximum a posteriori (MAP) estimates when data contain latent (hidden) variables or are incomplete.

### Overview

Given observed data $X$, latent variables $Z$, and parameters $\theta$, we want to maximize the marginal log-likelihood of the observed data:

$$
\log p(X \mid \theta) = \log \sum_{Z} p(X, Z \mid \theta).
$$

Because the summation sits inside the logarithm, the parameters are tightly coupled and this objective is difficult to optimize directly. 
The key idea behind EM is to construct a tractable lower bound and maximize the bound instead.

To derive that bound, introduce an arbitrary distribution $q(Z)$ over the latent variables. We can then rewrite:

$$
\log p(X \mid \theta) = \log \sum_{Z} q(Z) \frac{p(X, Z \mid \theta)}{q(Z)}
= \log \mathbb{E}_Z\!\left[\frac{p(X, Z \mid \theta)}{q(Z)}\right].
$$

According to Jensen's inequality, for a concave function $f$,

$$
f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)].
$$

![](jensens_inequality.png)

Since $\log$ is concave, we have:

$$
\log p(X \mid \theta) \ge \mathbb{E}_Z\!\left[\log \frac{p(X, Z \mid \theta)}{q(Z)}\right]
\triangleq \mathcal{L}(q, \theta).
$$

$\mathcal{L}(q, \theta)$ is called the **Evidence Lower Bound (ELBO)**.

Why is this a lower bound? The gap between the true log-likelihood and the ELBO turns out to be exactly the KL divergence 
between $q(Z)$ and the posterior $p(Z \mid X, \theta)$. To see why, take the difference:

$$
\log p(X \mid \theta) - \mathcal{L}(q, \theta)
= \mathbb{E}_Z\!\big[\log p(X \mid \theta)\big]
- \mathbb{E}_Z\!\left[\log \frac{p(X, Z \mid \theta)}{q(Z)}\right].
$$

The first expectation is just $\log p(X \mid \theta)$, because it does not depend on $Z$. Using the chain rule 
$p(X, Z \mid \theta) = p(Z \mid X, \theta)\,p(X \mid \theta)$, we have $\log p(X, Z \mid \theta) = \log p(Z \mid X, \theta) + \log p(X \mid \theta)$. 
Substituting this in and simplifying:

$$
\begin{aligned}
\log p(X \mid \theta) - \mathcal{L}(q, \theta)
&= \mathbb{E}_Z\!\left[
\log p(X \mid \theta)
- \log p(Z \mid X, \theta)
- \log p(X \mid \theta)
+ \log q(Z)
\right] \\
&= \mathbb{E}_Z\!\left[
\log q(Z) - \log p(Z \mid X, \theta)
\right] \\
&= \mathrm{KL}\big(q(Z) \,\|\, p(Z \mid X, \theta)\big).
\end{aligned}
$$

Rearranging gives the decomposition:

$$
\log p(X \mid \theta) = \mathcal{L}(q, \theta) + \mathrm{KL}\big(q(Z) \,\|\, p(Z \mid X, \theta)\big).
$$

The objective $\mathcal{L}(q, \theta)$ depends on two sets of variables: the model parameters $\theta$ and the distribution $q(Z)$. 
Optimizing them jointly is still difficult, but optimizing one while holding the other fixed is tractable. This suggests a coordinate-ascent strategy.

If we fix $\theta$, then $\log p(X \mid \theta)$ is constant. From the decomposition above, maximizing $\mathcal{L}(q, \theta)$ 
with respect to $q$ is equivalent to minimizing $\mathrm{KL}\big(q(Z) \,\|\, p(Z \mid X, \theta)\big)$. Because a KL divergence 
is always non-negative, the minimum is $0$, achieved when $q(Z) = p(Z \mid X, \theta)$. Setting $q$ to the posterior makes the 
bound tight: $\mathcal{L}(q, \theta) = \log p(X \mid \theta)$.

If we fix $q$, maximizing $\mathcal{L}(q, \theta)$ with respect to $\theta$ pushes the lower bound upward. Even though the 
bound may no longer be tight after $\theta$ changes, raising the bound is guaranteed to raise the true log-likelihood.

EM therefore alternates between two steps:

- **E-Step**: tightening the bound by optimizing $q$;
- **M-Step**: pushing the bound upward by optimizing $\theta$.

### $Q$-Function

First, let us expand the ELBO:

$$
\mathcal{L}(q, \theta)
= \mathbb{E}_Z\!\big[\log p(X, Z \mid \theta)\big] - \mathbb{E}_Z\!\big[\log q(Z)\big].
$$

The second term is the entropy of $q$, which does not depend on $\theta$ in the M-step because $q$ is fixed. Therefore, 
maximizing $\mathcal{L}$ with respect to $\theta$ is equivalent to maximizing only the first term.

Also note that the KL divergence vanishes in the E-step to make the bound tight:

$$
q(Z) = p(Z \mid X, \theta^{(t)}).
$$

Therefore, our goal becomes maximizing an expected complete-data log-likelihood, called **$Q$-function**:

$$
Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z \mid X,\theta^{(t)}} \big[\log p(X, Z \mid \theta) \big].
$$

### E-Step

Fix $\theta = \theta^{(t)}$ and compute the $Q$-function.

How do we evaluate this expectation when $Z$ appears inside the logarithm? Because the expectation is over $Z$, the latent variable is averaged outside the log. 
For discrete $Z$, the expectation is a concrete weighted sum over all possible assignments:

$$
Q(\theta, \theta^{(t)}) = \sum_{Z} p(Z \mid X, \theta^{(t)}) \, \log p(X, Z \mid \theta).
$$

Each term evaluates $\log p(X, Z \mid \theta)$ at a particular assignment $Z$ and weights it by the posterior probability of that assignment. 
We never need to manipulate $q$ as an abstract distribution; we only need these posterior probabilities to form the weighted sum.

### M-Step

With $q(Z)$ fixed at the posterior from the E-step, maximize the $Q$-function with respect to $\theta$:

$$
\theta^{(t+1)} = \arg\max_{\theta} Q(\theta, \theta^{(t)}).
$$

This is usually a standard MLE problem, but the "data" now include soft assignments from the E-step.

### Monotonic Improvement

Each EM iteration guarantees a **non-decreasing observed log-likelihood**, illustrated below:

![](em_elbo.png)

1. **E-step**: Tightens the lower bound to touch $\log p(X \mid \theta^{(t)})$, and constructs a better ELBO.
2. **M-step**: Increases the lower bound to $\mathcal{L}(q, \theta^{(t+1)})$ (red arrows in the picture).

After an iteration:

$$
\mathcal{L}(q, \theta^{(t+1)}) \geq \mathcal{L}(q, \theta^{(t)}) = \log p(X \mid \theta^{(t)})
$$

The gap between the true likelihood and the ELBO is a KL divergence, which is always non-negative. Therefore:

$$
\log p(X \mid \theta^{(t+1)}) = \mathcal{L}(q, \theta^{(t+1)}) + \underbrace{\mathrm{KL}\big(q(Z) \,\|\, p(Z \mid X, \theta^{(t+1)})\big)}_{\geq 0} \geq \mathcal{L}(q, \theta^{(t+1)}).
$$

Chaining these together gives:

$$
\log p(X \mid \theta^{(t+1)}) \geq \log p(X \mid \theta^{(t)}).
$$

Thus, $\log p(X \mid \theta)$ climbs monotonically until convergence.

One caveat is that EM is only guaranteed to reach a *local* optimum, so initialization matters. The algorithm can also 
converge arbitrarily slowly near saddle points or flat regions of the likelihood surface.

## Applying EM to GMM

We can now return to the mixture model from the beginning of this post. In the "Maximum Likelihood Parameter Estimation" section, 
we wrote the complete-data log-likelihood using hard one-hot assignments $z_{ij}$, but optimizing over both the parameters and 
these discrete indicators leads to a combinatorial problem with many local optima. EM solves exactly this difficulty.

For a Gaussian Mixture Model, the observed data are $X = \{x_1, \dots, x_N\}$, the latent assignments are $z_n \in \{1, \dots, K\}$, 
and the parameters are $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$. Using the same one-hot encoding $z_{nk} \in \{0, 1\}$ with $\sum_{k=1}^K z_{nk} = 1$ 
that we introduced earlier, the complete-data log-likelihood is:

$$
\log p(X, Z \mid \theta) = \sum_{n=1}^N \sum_{k=1}^K z_{nk} \bigl(\log \pi_k + \log \mathcal{N}(x_n \mid \mu_k, \Sigma_k)\bigr).
$$

Plugging this into the general $Q$-function gives the GMM-specific objective:

$$
\begin{aligned}
Q(\theta, \theta^{(t)}) &= \mathbb{E}_{Z \mid X,\theta^{(t)}} \big[\sum_{n=1}^N \sum_{k=1}^K z_{nk} \bigl(\log \pi_k + \log \mathcal{N}(x_n \mid \mu_k, \Sigma_k)\bigr)\big] \\
&= \sum_{n=1}^N \sum_{k=1}^K \mathbb{E}_{Z \mid X,\theta^{(t)}}[z_{nk}] \bigl(\log \pi_k + \log \mathcal{N}(x_n \mid \mu_k, \Sigma_k)\bigr).
\end{aligned}
$$

> Complete-data MLE pretends we know the latent assignments $z_{nk}$ and optimizes $\log p(X, Z \mid \theta)$ directly. Observed-data MLE, which is what EM targets, instead integrates over that uncertainty via marginalization: EM replaces each hard $z_{nk} \in \{0,1\}$ with its soft posterior expectation $\mathbb{E}[z_{nk}]$.

### E-Step: Compute Responsibilities

The E-step evaluates the expectations that appear in the $Q$-function above. Because $z_{nk}$ is an indicator, its conditional 
expectation is simply the posterior probability of belonging to component $k$:

$$
\mathbb{E}_{Z \mid X,\theta^{(t)}}[z_{nk}] = p(z_n = k \mid x_n, \theta^{(t)}) = \frac{\pi_k^{(t)} \, \mathcal{N}(x_n \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \, \mathcal{N}(x_n \mid \mu_j^{(t)}, \Sigma_j^{(t)})}.
$$

These posterior expectations are called responsibilities. They play the same role that the hard assignments $z_{nk}$ played in the complete-data MLE, but now as soft, fractional weights.

### M-Step: Weighted Maximum Likelihood

With the responsibilities fixed, maximizing $Q$ reduces to a weighted MLE problem for each Gaussian component:

$$
\begin{aligned}
N_k &= \sum_{n=1}^N \mathbb{E}[z_{nk}], \\[6pt]
\pi_k^{(t+1)} &= \frac{N_k}{N}, \\[6pt]
\mu_k^{(t+1)} &= \frac{1}{N_k} \sum_{n=1}^N \mathbb{E}[z_{nk}] \, x_n, \\[6pt]
\Sigma_k^{(t+1)} &= \frac{1}{N_k} \sum_{n=1}^N \mathbb{E}[z_{nk}] \, (x_n - \mu_k^{(t+1)})(x_n - \mu_k^{(t+1)})^\top.
\end{aligned}
$$

The algorithm alternates between these two steps until the log-likelihood or the parameters change by less than a small threshold $\epsilon$.

### Implementation in NumPy

Below is a compact, self-contained implementation using only NumPy.

```python
import numpy as np


class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.K = n_components      # number of mixture components
        self.max_iter = max_iter   # maximum EM iterations
        self.tol = tol             # convergence threshold for log-likelihood

    def _gaussian_pdf(self, X, mu, Sigma):
        """Evaluate the multivariate Gaussian N(mu, Sigma) for each row of X."""
        d = X.shape[1]
        det = np.linalg.det(Sigma)
        inv = np.linalg.inv(Sigma)
        diff = X - mu
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        return np.exp(exponent) / np.sqrt((2 * np.pi) ** d * det)

    def fit(self, X, init_means=None):
        N, D = X.shape

        # Initialize mixture weights uniformly and by default pick K random data points as means.
        # In practice, K-means centroids give a better starting point.
        self.pi = np.ones(self.K) / self.K
        if init_means is not None:
            self.mu = np.array(init_means)
        else:
            self.mu = X[np.random.choice(N, self.K, replace=False)]
        self.Sigma = np.array([np.eye(D) for _ in range(self.K)])

        log_likelihoods = []

        for _ in range(self.max_iter):
            # --- E-step: compute responsibilities E[z_nk] ---
            # gamma[n, k] = pi_k * N(x_n | mu_k, Sigma_k) / sum_j(...)
            gamma = np.zeros((N, self.K))
            for k in range(self.K):
                gamma[:, k] = self.pi[k] * self._gaussian_pdf(
                    X, self.mu[k], self.Sigma[k]
                )
            gamma /= gamma.sum(axis=1, keepdims=True)

            # --- M-step: weighted maximum likelihood ---
            # N_k = sum_n E[z_nk]  (effective number of points in component k)
            Nk = gamma.sum(axis=0)

            # Update mixture weights: pi_k = N_k / N
            self.pi = Nk / N

            for k in range(self.K):
                # Weighted mean: mu_k = (1/N_k) sum_n E[z_nk] * x_n
                self.mu[k] = (gamma[:, k][:, None] * X).sum(axis=0) / Nk[k]

                diff = X - self.mu[k]
                # Weighted covariance: Sigma_k = (1/N_k) sum_n E[z_nk] * (x_n - mu_k)(x_n - mu_k)^T
                self.Sigma[k] = (gamma[:, k][:, None] * diff).T @ diff / Nk[k]

                # Small ridge to prevent singular covariance if a component collapses.
                self.Sigma[k] += 1e-6 * np.eye(D)

            # --- Compute observed-data log-likelihood for convergence check ---
            # log p(X | theta) = sum_n log sum_k pi_k * N(x_n | mu_k, Sigma_k)
            likelihood = np.zeros(N)
            for k in range(self.K):
                likelihood += self.pi[k] * self._gaussian_pdf(
                    X, self.mu[k], self.Sigma[k]
                )
            log_likelihoods.append(np.sum(np.log(likelihood)))

            if len(log_likelihoods) > 1:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                    break

        return self

    def predict(self, X):
        """Assign each point to the component with highest responsibility."""
        gamma = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            gamma[:, k] = self.pi[k] * self._gaussian_pdf(
                X, self.mu[k], self.Sigma[k]
            )
        return gamma.argmax(axis=1)
```

## Experiment: Mall Customer Segmentation

To see how GMM performs on real data, we apply it to the [Mall Customer Segmentation dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python), which contains demographic and behavioral information for 200 customers of a shopping mall.

| Feature | Description |
|---------|-------------|
| `CustomerID` | Unique identifier (not predictive) |
| `Gender` | Male or Female |
| `Age` | Customer age in years |
| `Annual Income (k$)` | Annual income in thousands of US dollars |
| `Spending Score (1-100)` | Mall-assigned score based on spending behavior |

The objective is **unsupervised customer segmentation**: grouping customers into distinct clusters based on similarities without predefined labels.

### Exploratory Data Analysis

| Statistic | Age | Annual Income (k$) | Spending Score (1-100) |
|-----------|-----|--------------------|------------------------|
| Mean | 38.85 | 60.56 | 50.20 |
| Std | 13.97 | 26.26 | 25.82 |
| Min | 18 | 15 | 1 |
| Max | 70 | 137 | 99 |

![](eda_distributions.png)

The pairwise correlations in the heatmap are weak. If clusters exist, they are likely determined by interactions between variables rather than by any single dominant axis.

![](eda_income_vs_spending.png)

The scatter plot of Annual Income versus Spending Score reveals no simple linear relationship. Instead, customers appear to separate into distinct regions: 
high earners with low scores, low earners with high scores, and various intermediate groups. This suggests that **customer segments exist as overlapping clouds** 
rather than crisp, separable classes — exactly the scenario where GMM shines.

**Why GMM for this problem?**

1. **Soft Clustering.** K-Means assigns each point to exactly one cluster. In reality, a customer might sit on the boundary between two segments. GMM provides probabilistic memberships, acknowledging that uncertainty.
2. **Elliptical Clusters.** K-Means assumes spherical clusters of equal size. The income vs. spending plot suggests clusters may be elongated or oriented at angles. GMM captures covariance structure, allowing for elliptical, variably-sized clusters.
3. **Generative Model.** Because GMM models the data as a mixture of Gaussians, we can later generate synthetic customer profiles or compute the likelihood of new customers under the learned model.

### First Attempt: 2D GMM

We start with a simple two-feature model using Annual Income and Spending Score. After standardizing the features, we fit a GMM with $K=5$ and 
visualize the resulting clusters together with their 2-standard-deviation covariance ellipses.

![](2d_first_attempt.png)

The model captures several visible groupings, but it doesn't seem to make sense.

### Choosing the Number of Components

A practical question when fitting a GMM is selecting the number of components $K$.

#### Information Criteria: AIC and BIC

For a model with $p$ parameters fit to $n$ samples, the information criteria are:

$$
\text{AIC} = 2p - 2\ln(\hat{L}), \qquad
\text{BIC} = \ln(n)\,p - 2\ln(\hat{L}),
$$

where $\hat{L}$ is the maximized likelihood of the data. Both criteria reward likelihood while penalizing complexity; 
BIC imposes a heavier penalty and tends to favor simpler models. Lower is better.

#### Silhouette Score

For each sample, the silhouette coefficient compares its average distance to points in its own cluster (cohesion) with 
its average distance to the nearest neighboring cluster (separation). The overall score ranges from $-1$ to $+1$:

- **$+1$**: the sample is far from neighboring clusters (well-clustered).
- **$0$**: the sample lies near the boundary between two clusters.
- **$-1$**: the sample may have been assigned to the wrong cluster.

*Caveat:* Silhouette uses Euclidean distance. After standardizing features this is reasonable, but it may still favor 
spherical clusters even when GMM has learned an elongated covariance structure.

#### Calinski-Harabasz Score

Also known as the Variance Ratio Criterion, the Calinski-Harabasz (CH) score is the ratio of between-cluster dispersion 
to within-cluster dispersion:

$$
\text{CH} = \frac{\operatorname{tr}(B_k) / (k - 1)}{\operatorname{tr}(W_k) / (n - k)},
$$

where $B_k$ is the between-cluster scatter matrix, $W_k$ is the within-cluster scatter matrix, $k$ is the number of 
clusters, and $n$ is the number of samples. Higher values indicate better-defined clusters.

Because the denominator grows with $k$, CH penalizes overly complex models. For a business use-case such as targeted 
marketing campaigns, you generally want a small number of interpretable segments; a high CH score at a moderate $k$ is 
a strong signal that you have found a good grouping.

*Caveat:* Like silhouette, CH assumes Euclidean space and can be less reliable when clusters have very different 
densities or highly anisotropic covariances.

> [!NOTE]
> Silhouette and Calinski-Harabasz assume Euclidean geometry and crisp boundaries, whereas GMM uses Mahalanobis distances and soft assignments. While these metrics are still informative, they are not native to GMM. This is why we also rely heavily on likelihood-based criteria (AIC/BIC) derived directly from the model. Comparing all four criteria together gives a more robust basis for choosing $K$.

#### Selecting the Best $K$

We fit GMMs with $K = 2, \dots, 10$ and record AIC, BIC, silhouette score, and Calinski-Harabasz score for each.

![](metrics_old_init.png)

**Best $K$ by each metric:**
- AIC: $K=10$
- BIC: $K=5$
- Silhouette: $K=3$
- Calinski-Harabasz: $K=8$

The four metrics do not agree. AIC favors complexity ($K=10$), while BIC penalizes it more strongly and settles on $K=5$. 
Silhouette prefers $K=3$, and Calinski-Harabasz prefers $K=8$. We select $K=5$ based on BIC, because BIC provides a stronger 
penalty for model complexity and is generally more reliable for GMM model selection.

### Better Initialization with K-Means

EM is guaranteed to improve the likelihood at each step, but only with respect to the starting point. Random initialization 
can land in poor local optima. A common remedy is to initialize the Gaussian means with K-Means centroids. Re-running the 
same $K$ sweep with K-means initialization gives:

**Best $K$ by each metric (K-means init):**
- AIC: $K=9$
- BIC: $K=5$
- Silhouette: $K=5$
- Calinski-Harabasz: $K=5$

With K-means initialization, three of the four metrics now agree on $K=5$. To see the qualitative difference, we compare the best $K=5$ models side by side.

![](comparison_init.png)

K-means initialization improves all four metrics at $K=5$:

| Metric | Random Init | K-means Init |
|--------|-------------|--------------|
| AIC | 977.64 | 962.89 |
| BIC | 1073.29 | 1058.54 |
| Silhouette | 0.246 | 0.554 |
| Calinski-Harabasz | 40.94 | 244.41 |

Both AIC and BIC decrease, indicating a better fit with the same complexity. More dramatically, the Silhouette score more 
than doubles and Calinski-Harabasz increases six-fold. This confirms that K-means initialization helps EM converge to a 
much better local optimum with more coherent, well-separated clusters.

> Why not just use K-means and stop there? K-means is excellent for finding a quick, rough partition of the data, which is why it works so well as an initializer. But it makes three restrictive assumptions that GMM relaxes:
>
> **1. Hard vs. Soft Clustering.** K-means assigns every point to exactly one cluster. GMM treats cluster membership as a probability: a customer can be 70 % in segment A and 30 % in segment B. This matters at boundaries, where forcing a hard assignment discards useful uncertainty.
>
> **2. Cluster Shape and Size.** K-means assumes clusters are spherical and equally sized because it minimizes Euclidean distance to centroids. GMM learns a full covariance matrix for each component, so it can capture elliptical, stretched, or variably-sized clusters.
>
> **3. Partitioning vs. Generative Modeling.** K-means is a distance-based partitioning algorithm: it slices the input space into Voronoi cells. GMM is a generative probabilistic model: it describes how the data could have been produced. That means you can compute the likelihood of new customers, sample synthetic profiles, or embed the model inside a larger probabilistic pipeline.

### Extending to Four Dimensions

So far we have used only Income and Spending Score. We now add Age and Gender to see whether richer features produce better clusters, again using K-means initialization.

![](metrics_2d_4d_compare.png)

Both AIC and BIC drop sharply once $K > 2$, and for every $K \geq 3$ the 4D model scores better than the 2D model on these likelihood-based criteria. 
This suggests that the extra features do carry information that improves the probabilistic fit.

Because the 4D clusters cannot be visualized directly, we project the best model (by BIC, $K=4$) into the first three principal components.

![](4d_pca.png)

The four clusters appear well separated in PCA space, but a closer look reveals a problem: every cluster is long and narrow. 
They form two distant pairs, and within each pair the two clusters lie very close together. A point at the sharp tip of one 
cluster can actually be closer to the center of its neighbor than to its own center. This is exactly the geometry that silhouette 
and Calinski-Harabasz penalize, so it is no surprise that these Euclidean-based metrics prefer the 2D model: by their standards, 
4D with $K=4$ is worse than 2D with $K=5$.

Which model should we choose? This is not a purely statistical question. If the mall wants the best possible probabilistic fit 
and can afford tailored strategies for different segments, the 4D model is justified by AIC and BIC. If the marketing team needs 
simple, actionable rules, the 2D model is far easier to explain: high or low income crossed with high or low spending score 
naturally gives four quadrants, plus a middle group, for five clear customer types.

The 4D clusters offer a different lens. Their centers are:

| Cluster | Age | Annual Income (k$) | Spending Score (1-100) | Gender |
|---------|-----|--------------------|------------------------|--------|
| 1 | 41.6 | 57.5 | 39.2 | Female |
| 2 | 28.1 | 62.2 | 72.0 | Male |
| 3 | 29.6 | 63.4 | 81.3 | Female |
| 4 | 49.2 | 62.3 | 29.7 | Male |

Notice that income is almost the same across all four groups — close to the overall average. Instead of separating customers by wealth, 
the 4D model partitions them by age, spending behavior, and gender. This is a valid but very different story from the 2D income-versus-spending view.

Ultimately, the choice between 2D and 4D depends on what the business values more: simplicity and visual clarity, or a richer, more nuanced segmentation.

### Outliers

With only 200 customers, this dataset is small, so we should be cautious about labeling any point an outlier simply because it stands out from the rest. 
A seemingly unusual customer — for example, someone with very high income and very low spending — might represent a legitimate but under-sampled segment in the population. 
Removing it could erase a real customer group rather than clean the data.

Still, it is worth examining the data for extreme values and considering one of the following strategies if the dataset were larger or noisier:

- **Robust preprocessing:** Remove or cap extreme values before fitting.
- **Robust GMM variants:** Use t-distributions (t-mixture models) instead of Gaussians, which are more robust to outliers due to heavier tails.
- **Down-weighting:** Apply weights to data points or use a trimmed likelihood approach.

### Complete Experiment Pipeline

The following pipeline wraps `sklearn.mixture.GaussianMixture` into a reusable class that handles scaling, fitting, prediction, and metric collection.

```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd
import numpy as np


class GMMPipeline:
    def __init__(self, features, n_components, init="kmeans", random_state=42):
        self.features = features
        self.n_components = n_components
        self.init = init
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def _prepare(self, df, fit=False):
        X = df[self.features].values
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def fit(self, df):
        X = self._prepare(df, fit=True)
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            init_params=self.init,
            random_state=self.random_state,
            max_iter=200,
        )
        self.model.fit(X)
        return self

    def predict(self, df):
        X = self._prepare(df)
        return self.model.predict(X)

    def metrics(self, df):
        X = self._prepare(df)
        labels = self.model.predict(X)
        return {
            "AIC": self.model.aic(X),
            "BIC": self.model.bic(X),
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
        }

    def cluster_centers(self):
        return pd.DataFrame(
            self.scaler.inverse_transform(self.model.means_),
            columns=self.features,
        )


df = pd.read_csv("Mall_Customers.csv")

# Encode Gender as a numeric feature
df["Gender_encoded"] = df["Gender"].map({"Male": 0, "Female": 1})

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender_encoded"]

pipeline = GMMPipeline(features=features, n_components=4)
pipeline.fit(df)

labels = pipeline.predict(df)
print(pipeline.metrics(df))
print(pipeline.cluster_centers())
```