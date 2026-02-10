#!/usr/bin/env python3
"""
Command line utility to add permitted values to schema
Usage:
    python add_permitted_value.py phenotype HEART_DEFECT "Heart development defect"
    python add_permitted_value.py treatment BMP4 "Bone morphogenetic protein 4"
    python add_permitted_value.py flag OUT_OF_FOCUS "Image out of focus" --level image --severity warning
"""

import argparse
from permitted_values_manager import PermittedValuesManager

def main():
    parser = argparse.ArgumentParser(description="Add permitted values to schema")
    parser.add_argument("category", choices=["phenotype", "genotype", "treatment", "flag"])
    parser.add_argument("name", help="Name of the value")
    parser.add_argument("description", help="Description of the value")
    parser.add_argument("--level", help="Flag level (for flags only)")
    parser.add_argument("--severity", default="warning", help="Flag severity")
    parser.add_argument("--aliases", nargs="*", help="Aliases for genotypes")
    parser.add_argument("--exclusive", action="store_true", help="Phenotype is exclusive")
    parser.add_argument("--terminal", action="store_true", help="Phenotype is terminal")
    
    args = parser.parse_args()
    
    manager = PermittedValuesManager()
    
    try:
        if args.category == "phenotype":
            manager.add_phenotype(args.name, args.description, args.exclusive, args.terminal)
        elif args.category == "genotype":
            manager.add_genotype(args.name, args.description, args.aliases)
        elif args.category == "treatment":
            manager.add_treatment(args.name, args.description)
        elif args.category == "flag":
            if not args.level:
                print("Error: --level required for flags")
                return
            manager.add_flag(args.level, args.name, args.description, args.severity)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
