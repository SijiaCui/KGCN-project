@echo off
::===============================================================
:: script for training
:: default para: aggregator:sum, neighbor_sample_size:8, dim:16, n_iter:1
:: exp1: different aggregate mode: --aggregator=['sum', 'concat', 'neighbor']
:: exp2: neighbor_sample_size=[2,4,8,16,32]
:: exp3: dim=[8,16,32,64]
:: exp4: n_iter=[1,2,3,4]
::===============================================================

call conda activate pattern

:: exp1
python main.py > exp1_aggregator_sum.log --aggregator='sum'
python main.py > exp1_aggregator_concat.log --aggregator='concat'
python main.py > exp1_aggregator_neighbor.log --aggregator='neighbor'

:: exp2
python main.py > exp2_neighbor_size_2.log --neighbor_sample_size=2
python main.py > exp2_neighbor_size_4.log --neighbor_sample_size=4
python main.py > exp2_neighbor_size_8.log --neighbor_sample_size=8
python main.py > exp2_neighbor_size_16.log --neighbor_sample_size=16
python main.py > exp2_neighbor_size_32.log --neighbor_sample_size=32

:: exp3
python main.py > exp3_dim_8.log --dim=8
python main.py > exp3_dim_16.log --dim=16
python main.py > exp3_dim_32.log --dim=32
python main.py > exp3_dim_64.log --dim=64

:: exp4
python main.py > exp4_n_iter_1.log --n_iter=1 --n_epochs=50
python main.py > exp4_n_iter_2.log --n_iter=2 --n_epochs=50
python main.py > exp4_n_iter_3.log --n_iter=3 --n_epochs=50
python main.py > exp4_n_iter_4.log --n_iter=4 --n_epochs=50
