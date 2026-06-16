# Full split (baseline comparison — no filtering)
python eval_results.py \
    ./config/only_vehicles.py \
    results/attack_results.pkl

# Attacked samples only, attacked class only  
python eval_results.py \
    ./config/only_vehicles.py \
    results/attack_results.pkl \
    --attack-filter sample_class

# Instance-level
python eval_results.py \
    ./config/only_vehicles.py \
    results/attack_results.pkl \
    --attack-filter instance