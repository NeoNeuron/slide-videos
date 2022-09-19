# Animation of Probability and Statistics

This repo provides scripts as generators for slides animation for probability and statistics courses.

## Requirements

Install the following packages to ensure the successful excution of scripts
```bash
conda install -y numpy matplotlib scipy ffmpeg ipython jupyter notebook

pip install svgpathtools svgpath2mpl
```

## Contents of topics and related scripts

1. 几何概率模型
    - `geo_prob_1d.py`
    - `geo_prob_3d.py`
    - `buffen_needle.py`
    - `buffen_measure.py`

2. 全概率公式（选自第一章：随机事件和概率）
   - `pcr.py`
   - `goat_problem.py`

3. 贝叶斯公式（选自第一章：随机事件和概率）
   - `med_test.py`

4. 随机事件独立性（选自第一章：随机事件和概率）
   - `tetrahedron_rotation.py`
   - `cube_morph.py`
   - `missile_hit.py`
   - `missile_shot.py`

7. 正态分布
    - `evolve_norm.py`
    - `normal_figures.py`

8. 随机变量的函数
    - `3d_projection.py`
    - `height_histogram.py`
    - `stick_density.py`
    - `random_gen_demo.py`

9. 二维正态分布
    - `gaussian2d.py`
    - `gaussian2d_evolve.py`
    - `gaussian2d_rotate.py`
    - `2d_gaussian_margin.py`
    - `gaussian2d_evolve_new.py`

10. 数学期望（选自第四章：随机变量的数字特征）
    - `discretize_func.py`
    - `simulate_central_limit.py`
    - `general_lln.py`
    - `previous_topics/car.ipynb`
    - `previous_topics/plot_curve.ipynb`
    - `SPitts.py`

11. 方差（选自第四章：随机变量的数字特征）
    - `previous_topics/variance.ipynb`
    - `bm.py`

12. 协方差和相关系数（选自第四章：随机变量的数字特征）
    - `corr_v1.py`
    - `corr_v2.py`
    - `dot_product.py`
    - `cov_fig.py`
    - `3d_bar.py`
    - `3d_bar_v2.py`
    - `previous_topics/covariance.ipynb`

13. 大数定律（选自第五章：大数定律和中心极限定理）
    - `flipping_coin.py`
    - `Chebyshev.py`
    - `previous_topics/chebyshev.ipynb`
    - `previous_topics/chebyshev2.ipynb`
    - `previous_topics/chebyshev3.ipynb`
    - `lln.py`
    - `simulate_central_limit.py`
    - `general_lln.py`
    - `car.py`

14. 中心极限定理（选自第五章：大数定律和中心极限定理）
    - `flight_test.py`
    - `flight_profit_test.py`
    - `general_distributions.py`
    - `general_distributions2.py`

17. 点估计的评价标准
    - `point_estimation.py`

19. 正态总体的假设检验（选自第八章：假设检验）
    - `previous_topics/hypothesis_test.ipynb`
    - `previous_topics/variance.ipynb`
    - `Hypothesis.py`
    - `Hypothesis1.py`
    - `hypothesis_sampling.py`

## Others Chapters

1. 概率的定义（选自第一章：随机事件和概率）
    - `flipping_coin.py`
    - `rcp.py`