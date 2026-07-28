import runpy
import sys
sys.path.insert(0, r'D:\Codes\Quantum\DriftAdaptiveQEC')
sys.stdout = open(r'D:\Codes\Quantum\DriftAdaptiveQEC\runs\t_risk_20260728_03_cutoff36_44_repair_fresh1\operator_logs\preflight.stdout.log', 'w', encoding='utf-8', buffering=1)
sys.stderr = open(r'D:\Codes\Quantum\DriftAdaptiveQEC\runs\t_risk_20260728_03_cutoff36_44_repair_fresh1\operator_logs\preflight.stderr.log', 'w', encoding='utf-8', buffering=1)
sys.argv = ['phase9_cutoff36_44_repair', '--preflight-only']
runpy.run_module('cnn_fpga.benchmark.phase9_cutoff36_44_repair', run_name='__main__')
