# ---   IMPORTS   --- #
# ------------------- #
from src.utils.preds.pred_reduce_strategy import PredReduceStrategy
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class EntropicPredReduceStrategy(PredReduceStrategy):
    r"""
    :author: Alberto M. Esmoris Pena

    Reduce many predictions per point to a single one by considering the
    point-wise entropies.

    The reduced likelihoods for the :math:`n_c` classes of the :math:`i`-th
    point will be as shown below, assuming :math:`K_i` values for the
    reduction.

    .. math::

        \pmb{\hat{z}}_{i} = \dfrac{
			\displaystyle \sum_{k=1}^{K_i}{(1-\hat{e}_{i_k}) \pmb{z}_{i_k}}
		}{
			\displaystyle \sum_{k=1}^{K_i}{(1-\hat{e}_{i_k})}
		} \in (0, 1)^{n_c}

    In the above equation, :math:`\pmb{z}_{i_k} \in (0, 1)^{n_c}` represents
    the :math:`k` vector of likelihoods for the :math:`i`-th point where
    :math:`k=1,\ldots,K_i` (:math:`K_i` is the number of neighborhoods where
    the :math:`i`-th point appears).

    To understand :math:`\hat{e}_{i_k}` it is necessary to define the entropy
    for a classification task of :math:`n_c` classes first such that:

    .. math::

        \mathcal{E}(\pmb{z}_{i_k}) =
            - \sum_{c=1}^{n_c}{z_{i_kc} \log_2(z_{i_kc})} =
            \sum_{c=1}^{n_c}{f(z_{i_kc})}

    Now, consider the derivatives of :math:`f(z) = -z \log_2(z)`:

    .. math::

        \dfrac{df}{dz} = - \dfrac{1+\ln(z)}{\ln(2)} \;,\quad
        \dfrac{d^2f}{dz^2} = - \dfrac{1}{z\ln(2)}

    Note that :math:`\dfrac{df}{dz} = 0 \iff z = e^{-1}` and
    :math:`\dfrac{d^2f}{dz^2} < 0`. For then, :math:`f(e^{-1})` will be a
    maximum, leading to an upper bound :math:`e^* = -n_ce^{-1}\log_2(e^{-1})`
    such that :math:`\mathcal{E}(\pmb{z}_{i_k}) \leq e^*`. Consequently, it
    is possible to obtain normalized entropies such that:

    .. math::

        \hat{e}_{i_k} = \dfrac{\mathcal{E}(\pmb{z}_{i_k})}{e^*} \in (0, 1)

    NOTE that a min clip value :math:`\eps_*` will be considered to replace
    all values below it to avoid logarithm of zero or division by zero cases.

    See :class:`.PredReduceStrategy`.

    :ivar min_clip_value: The minimum value that will be allowed. Values below
        it will be replaced to be the min clip value itself. This is useful to
        avoid division by zero and logarithm of zero cases. By default it is
        :math:`10^{-6}`.
    :vartype min_clip_value: float
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a mean prediction reduction strategy.

        :param kwargs: The attributes for the EntropicPredReduceStrategy.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.min_clip_value = kwargs.get('min_clip_value', 1e-6)

    # ---  REDUCE METHODS  --- #
    # ------------------------ #
    def reduce(self, reducer, npoints, nvals, Z, I):
        """
        See :class:`.PredReduceStrategy` and
        :meth:`.PredReduceStrategy.reduce`.
        """
        if nvals == 1:
            return self.reduce_single(reducer, npoints, nvals, Z, I)
        return self.reduce_many(reducer, npoints, nvals, Z, I)

    def reduce_single(self, reducer, npoints, nvals, Z, I):
        """
        Handle the reduce operations when nvals=1.
        See :meth:`.EntropicPredReduceStrategy.reduce`.
        """
        # Initialize final likelihoods
        Zhat_type = Z[0].dtype
        Zhat = np.zeros(npoints, dtype=Zhat_type)
        # Initialize final denominators
        denoms = np.zeros(npoints, dtype=Zhat_type)
        # Compute entropic norm
        Enorm = - np.exp(-1)*np.log2(np.exp(-1))
        # For each neighborhood
        for i, Zi in enumerate(Z):
            # Clip values
            Zi = np.maximum(Zi, self.min_clip_value)
            # Compute normalized entropies
            E = - Zi * np.log2(Zi) / Enorm
            # Compute weights from normalized entropies
            w = 1 - E
            # Aggregate final likelihoods
            Zhat[I[i]] += w*Zi
            denoms[I[i]] += w
        # Normalize final likelihoods
        non_zero_mask = denoms != 0
        Zhat[non_zero_mask] = Zhat[non_zero_mask] / denoms[non_zero_mask]
        return Zhat

    def reduce_many(self, reducer, npoints, nvals, Z, I):
        """
        Handle the reduce operation when nvals>1.
        See :meth:`.EntropicPredReduceStrategy.reduce`.
        """
        # Initialize final likelihoods
        Zhat_type = Z[0].dtype
        Zhat = np.zeros((npoints, nvals), dtype=Zhat_type)
        # Initialize final denominators
        denoms = np.zeros(npoints, dtype=Zhat_type)
        # Compute entropic norm
        Enorm = -nvals * np.exp(-1)*np.log2(np.exp(-1))
        # For each neighborhood
        for i, Zi in enumerate(Z):
            # Clip values
            Zi = np.maximum(Zi, self.min_clip_value)
            # Compute normalized entropies
            E = - np.sum(Zi * np.log2(Zi), axis=-1)/Enorm
            # Compute weights from normalized entropies
            w = 1 - E
            # Aggregate final likelihoods
            Zhat[I[i]] += np.expand_dims(w, axis=-1)*Zi
            denoms[I[i]] += w
        # Normalize final likelihoods
        non_zero_mask = denoms != 0
        Zhat[non_zero_mask] = Zhat[non_zero_mask] / np.expand_dims(
            denoms[non_zero_mask], axis=-1
        )
        return Zhat
