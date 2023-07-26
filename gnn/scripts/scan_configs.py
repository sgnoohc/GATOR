#!/bin/env python
import argparse
import itertools

from utils import GatorConfig

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def filename_str(value):
    if type(value) == str:
        return value
    else:
        return str(value).replace(".", "p").replace("-", "m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make standard plots")
    parser.add_argument("config_json", type=str, help="base config JSON")
    parser.add_argument("scans", nargs="*")
    parser.add_argument(
        "--individual", action="store_true",
        help="make a new config for each unique setting"
    )
    parser.add_argument(
        "--constrained", action="store_true",
        help="make a new config following the specific ordering of the settings provided"
    )
    parser.add_argument(
        "--combinatoric", action="store_true",
        help="make a new config for every combination of settings"
    )
    args = parser.parse_args()

    # Get base config
    base_config = GatorConfig.from_json(args.config_json)

    # Parse CLI for scan configurations
    scans = {}
    for scan in args.scans:
        scan = scan.replace(" ", "")
        if ":" in scan:
            key, values = scan.split(":")
            values = values.split(",")
        elif "=" in scan:
            key, values = scan.split("=")
            values = values.split(",")
        else:
            raise Exception(
                f"{scan} is not properly formatted (must be key:v1,v2,... or key=v1,...)"
            )

        if is_number(values[0]):
            scans[key] = [int(v) if v.isdigit() else float(v) for v in values]
        else:
            scans[key] = values

    # Get all settings for individual configs
    new_settings = []
    if args.individual:
        for key, values in scans.items():
            for value in values:
                new_settings.append({key: value})

    # Get all settings for constrained configs
    if args.constrained:
        n_values = set([len(values) for values in scans.values()])
        if len(n_values) != 1:
            raise Exception("all scans must be the same length for a constrained scan")
        else:
            for value_i in range(n_values.pop()):
                new_settings.append({key: values[value_i] for key, values in scans.items()})

    # Get all settings for combinatoric configs
    if args.combinatoric:
        combinations = itertools.product(*[[(k, v) for v in scans[k]] for k in scans])
        for combination in combinations:
            new_settings.append({key: value for key, value in combination})

    # Create new configs
    new_configs = []
    for settings in new_settings:
        new_config = base_config.copy(extra=settings)

        new_name = base_config.name
        for key, value in settings.items():
            new_name += f"_{key}-{filename_str(value)}"
        new_config.set_name(new_name)

        new_configs.append(new_config)

    if len(new_configs) > 5:
        # Check if user really wants to make a lot of configs
        print("Preparing new configs:")
        print("\n".join([new_config.name for new_config in new_configs[:10]]))
        print(f"+ {len(new_configs) - 10} more...\n")
        resp = None
        while resp not in ["Y", "y", "N", "n"]:
            resp = input(f"Create {len(new_configs)} new configs? (y/n): ")
        if resp == "N" or resp == "n":
            print("Aborted.")
            exit()
    elif len(new_configs) > 0:
        # Print the configs that are about to be written
        print("Writing new configs:")
        print("\n".join([new_config.name for new_config in new_configs]))
        
    # Write new configs
    for new_config in new_configs:
        with open(f"configs/{new_config.name}.json", "w") as f:
            new_config.dump(f)
    print("Done.")
