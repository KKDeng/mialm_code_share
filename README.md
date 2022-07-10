 # Code of paper "[A manifold inexact augmented Lagrangian method for nonsmooth optimization on Riemannian manifold](https://arxiv.org/pdf/1911.09900)"

Our code is based on "[manopt toolbox](https://www.manopt.org/index.html)" and code in paper "[Proximal Gradient Method for Nonsmooth Optimization over the Stiefel Manifold](https://epubs.siam.org/doi/abs/10.1137/18M122457X)" 



MIALM is a algorithm developed for solving the nonsmooth optimization problems on Riemannian manifold: $$ \min\limits_{X \in \mathcal{M}}   f (X) + h(\mathcal{A}(X)) .$$  Here, $f(X): \mathcal{M} \rightarrow \mathbb{R}$ is smoooth, $h(X):\mathbb{R}^m \rightarrow \mathbb{R}$ is  convex but nonsmooth, $\mathcal{A}:\mathcal{M}\to \mathbb{R}^m$ is a linear operator, and the feasible set $\mathcal{M}$ is a Riemannian manifold.






## References
If you use these codes in an academic paper, please cite the following papers.

 Kangkang, Deng, and Peng Zheng, [*An inexact augmented Lagrangian method for nonsmooth optimization on Riemannian manifold,IMA Journal of Numerical Analysis*](https://doi.org/10.1093/imanum/drac018)).

## Authors

- [Kangkang Deng](https://kangkang-deng.github.io/) (dengkangkang@pku.edu.cn)
- Zheng Peng (pzheng@fzu.edu.cn)



## To Contact Us
If you have any questions or find any bugs, please feel free to contact us by email (dengkangkang@pku.edu.cn).




## License
 You can redistribute it and/or modify it under the terms of the [GNU Lesser General Public License](https://www.gnu.org/licenses/lgpl-3.0.en.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

MIALM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Please see the [GNU Lesser General Public License](https://www.gnu.org/licenses/lgpl-3.0.en.html) for more details.
 



