# Animation of Probability and Statistics

This repo provides scripts as generators for slides animation for probability and statistics courses.

## Requirements

Install the following packages to ensure the successful excution of scripts

```bash
conda env create -y -f requirements.yml
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
    - `standardize_norm.py`
    - `normal_distribution_figs.py`

8. 随机变量的函数
    - `incubation.py`
    - `incubation_anim.py`
    - `stock_price.py`
    - `3d_projection.py`
    - `height_histogram.py`
    - `stick_density.py`
    - `random_gen_demo.py`

9. 二维正态分布
    - `cpu_ram.py`
    - `cpu_ram_3d.py`
    - `gaussian2d.py`
    - `gaussian2d_evolve.py`
    - `gaussian2d_evolve_new.py`
    - `gaussian2d_morph.py`
    - `2d_gaussian_proj.py`
    - `2d_gaussian_margin.py`
    - `2d_gaussian_margin_snapshot.py`
    - `gaussian2d_rotate.py`

10. 数学期望（选自第四章：随机变量的数字特征）
    - `discretize_func.py`
    - `simulate_central_limit.py`
    - `general_lln.py`
    - `previous_topics/car.ipynb`
    - `pcr_expectation.py`
    - `SPitts.py`

11. 方差（选自第四章：随机变量的数字特征）
    - `previous_topics/variance.ipynb`
    - `bm.py`
    - `dot_motion.py`
    - `stock_optimization.py`

12. 协方差和相关系数（选自第四章：随机变量的数字特征）
    - `corr_v1.py`
    - `corr_v2.py`
    - `dot_product.py`
    - `cov_fig.py`
    - `3d_bar.py`
    - `3d_bar_v2.py`
    - `previous_topics/covariance.ipynb`
    - `corr_dep.py`
    - `corr_morph.py`

13. 大数定律（选自第五章：大数定律和中心极限定理）
    - `flipping_coin.py`
    - `Chebyshev.py`
    - `previous_topics/chebyshev3.ipynb`
    - `lln.py`
    - `simulate_central_limit.py`
    - `general_lln.py`
    - `car.py`
    - `integrate.py`
    - `company_income.py`

14. 中心极限定理（选自第五章：大数定律和中心极限定理）
    - `flight_test.py`
    - `flight_profit_test.py`
    - `general_distributions.py`
    - `general_distributions2.py`

15. 矩估计
    - `multi_moment.py`
    - `moment_estimation.py`
    - `evolve_moment.py`

17. 点估计的评价标准
    - `point_estimation.py`
    - `bulb_lifetime.py`
    - `lln.py`

18. 区间估计
    - `intervel_estimation_var_known.py`
    - `intervel_estimation_var_unknown.py`

19. 正态总体的假设检验（选自第八章：假设检验）
    - `previous_topics/hypothesis_test.ipynb`
    - `previous_topics/variance.ipynb`
    - `Hypothesis.py`
    - `Hypothesis1.py`
    - `hypothesis_sampling.py`

20. 线性回归
    - `lr_examples.py`
    - `lr_height_CI.py`
    - `lr_height_CI_movie.py`
    - `least_square_3d.py`
    - `Galton_height_regression.py`

## Others Chapters

1. 概率的定义（选自第一章：随机事件和概率）
    - `flipping_coin.py`
    - `rcp.py`