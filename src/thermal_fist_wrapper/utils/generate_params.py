#!/usr/bin/env python3
import pandas as pd
import yaml
from pathlib import Path

# Load particle list
def load_particle_list(file_path: Path):
    return pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, 
                       names=['pdgid', 
                              'name',
                              'stable',
                              'mass[GeV]',
                              'degeneracy',
                              'statistics',
                              'B',
                              'Q',
                              'S',
                              'C',
                              '|S|',
                              '|C|',
                              'width[GeV]',
                              'threshold[GeV]'])

# Load rules from YAML
def load_rules(rules_path: Path):
    with open(rules_path, 'r') as file:
        return yaml.safe_load(file)['rules']

# Evaluate rules
def get_params_for_particle(particle, rules):
    for rule in rules:
        if 'condition' in rule:
            condition = rule['condition']
            match = True
            for key, value in condition.items():
                if key == 'id_range':
                    if not (value[0] <= particle['pdgid'] <= value[1]):
                        match = False
                        break
                elif particle.get(key, None) != value:
                    match = False
                    break
            if match:
                return rule['params']['a'], rule['params']['b']
    return rules[-1]['default']['a'], rules[-1]['default']['b']

# Generate parameter files
def generate_param_file(particle_list: pd.DataFrame, rules, output_path: Path):
    with open(output_path, 'w') as f:
        f.write("# pdg_i pdg_j b_{ij}[fm^3] a_{ij}[GeV*fm^3]\n")
        for i, particle1 in particle_list.iterrows():
            for j, particle2 in particle_list.iterrows():
                # Determine parameters based on first particle (customizable logic)
                a, b = get_params_for_particle(particle1, rules)
                f.write(f"{particle1['pdgid']} {particle2['pdgid']} {b:.3f} {a:.3f}\n")

if __name__ == "__main__":
    particle_list_path = Path("input/list.dat")
    rules_path = Path("input/rules.yml")
    output_file_path = Path("input/param_files/params_generated.txt")

    particle_list = load_particle_list(particle_list_path)
    rules = load_rules(rules_path)

    generate_param_file(particle_list, rules, output_file_path)

    print(f"Parameter file created at {output_file_path}")
