{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pylab import *\n",
    "from Ising_Knockoffs import Ising_Knockoffs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Load Ising data and the respective coupling matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "Z = np.array(pd.read_csv(\"data\\\\Z.csv\"))\n",
    "Theta = np.array(pd.read_csv(\"data\\\\Theta.csv\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Create Instance of sampler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "sampler = Ising_Knockoffs(Z, Theta)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Sample knockoffs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "Z_tilde = sampler.sample_knockoffs()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Visualize results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x1440 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = subplots(1, 2, figsize=(10, 20))\n",
    "ax[0].imshow(Z)\n",
    "ax[0].set_xlabel(\"original features\")\n",
    "ax[1].set_xlabel(\"knockoff features\")\n",
    "ax[1].imshow(Z_tilde)\n",
    "show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# Ising-Knockoffs
Gibbs-Sampler, generating Knockoffs according to https://arxiv.org/abs/1807.00931
